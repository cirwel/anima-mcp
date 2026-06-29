defmodule AnimaBroker.Sensors.BMP280 do
  @moduledoc """
  BMP280 barometric pressure + temperature sensor (I2C 0x76/0x77).

  The compensation is the error-prone part: the trimming parameters are a
  **mix of signed and unsigned** 16-bit words. `dig_T1` and `dig_P1` are
  unsigned; everything else (`dig_T2..T3`, `dig_P2..P9`) is signed two's
  complement. Getting one wrong throws temperature and pressure off by tens of
  degrees / hundreds of hPa. The parser below pins each field's signedness, and
  `bmp280_test.exs` verifies it (and the full compensation) against the
  datasheet's Appendix-A reference vector.

  Compensation uses the datasheet's floating-point reference formulas
  (`bmp280_compensate_T_double` / `bmp280_compensate_P_double`), matching what
  `adafruit_bmp280` returns. Pressure is reported in hPa (Pa / 100), temperature
  in °C — the Python contract fields `pressure_hpa`, `pressure_temp_c`.
  """
  use GenServer
  require Logger
  import Bitwise

  alias AnimaBroker.Hardware.I2C
  alias AnimaBroker.State.Store

  # BMP280 responds at 0x76 (SDO low) or 0x77 (SDO high). The BrainCraft HAT —
  # like the adafruit_bmp280 default (`Adafruit_BMP280_I2C(i2c)` → 0x77) — wires
  # it at 0x77; the live Pi confirmed it (reads at 0x77, EIO at 0x76). Override
  # with the `:address` option for a board strapped to 0x76.
  @address 0x77

  # Calibration block: 24 bytes starting at 0x88.
  @reg_calib 0x88
  @reg_calib_len 24
  # Raw measurement block: 6 bytes starting at 0xF7 (press[3], temp[3]).
  @reg_data 0xF7
  @reg_data_len 6
  # ctrl_meas (0xF4) / config (0xF5): normal mode, osrs_t x1, osrs_p x1.
  #   ctrl_meas = osrs_t(111?) ... we use x1/x1: osrs_t=001, osrs_p=001, mode=11
  #   = 0b001_001_11 = 0x27
  @reg_ctrl_meas 0xF4
  @ctrl_meas_normal 0x27

  @type calibration :: %{
          required(:dig_t1) => non_neg_integer(),
          required(:dig_t2) => integer(),
          required(:dig_t3) => integer(),
          required(:dig_p1) => non_neg_integer(),
          required(:dig_p2) => integer(),
          required(:dig_p3) => integer(),
          required(:dig_p4) => integer(),
          required(:dig_p5) => integer(),
          required(:dig_p6) => integer(),
          required(:dig_p7) => integer(),
          required(:dig_p8) => integer(),
          required(:dig_p9) => integer()
        }

  # ---- Pure: calibration parse (signed/unsigned mix) ----

  @doc """
  Parse the 24-byte calibration block (little-endian). `dig_T1`/`dig_P1` are
  unsigned; the rest are signed. This signedness split is the whole ballgame.
  """
  @spec parse_calibration(binary()) :: {:ok, calibration()} | {:error, term()}
  def parse_calibration(
        <<t1::little-unsigned-16, t2::little-signed-16, t3::little-signed-16,
          p1::little-unsigned-16, p2::little-signed-16, p3::little-signed-16,
          p4::little-signed-16, p5::little-signed-16, p6::little-signed-16, p7::little-signed-16,
          p8::little-signed-16, p9::little-signed-16>>
      ) do
    {:ok,
     %{
       dig_t1: t1,
       dig_t2: t2,
       dig_t3: t3,
       dig_p1: p1,
       dig_p2: p2,
       dig_p3: p3,
       dig_p4: p4,
       dig_p5: p5,
       dig_p6: p6,
       dig_p7: p7,
       dig_p8: p8,
       dig_p9: p9
     }}
  end

  def parse_calibration(_), do: {:error, :bad_calibration}

  # ---- Pure: raw ADC extraction ----

  @doc "Extract 20-bit `{adc_p, adc_t}` from the 6-byte measurement block."
  @spec parse_raw(binary()) :: {:ok, {non_neg_integer(), non_neg_integer()}} | {:error, term()}
  def parse_raw(<<pm, pl, px, tm, tl, tx>>) do
    adc_p = pm <<< 12 ||| pl <<< 4 ||| px >>> 4
    adc_t = tm <<< 12 ||| tl <<< 4 ||| tx >>> 4
    {:ok, {adc_p, adc_t}}
  end

  def parse_raw(_), do: {:error, :short_read}

  # ---- Pure: compensation (datasheet double formulas) ----

  @doc "Compensate temperature. Returns `{t_fine, temp_c}` (t_fine feeds pressure)."
  @spec compensate_temperature(calibration(), integer()) :: {float(), float()}
  def compensate_temperature(c, adc_t) do
    var1 = (adc_t / 16384.0 - c.dig_t1 / 1024.0) * c.dig_t2

    var2 =
      (adc_t / 131_072.0 - c.dig_t1 / 8192.0) *
        (adc_t / 131_072.0 - c.dig_t1 / 8192.0) * c.dig_t3

    t_fine = var1 + var2
    {t_fine, t_fine / 5120.0}
  end

  @doc "Compensate pressure given `t_fine`. Returns pressure in **Pa** (0.0 if degenerate)."
  @spec compensate_pressure(calibration(), integer(), float()) :: float()
  def compensate_pressure(c, adc_p, t_fine) do
    var1 = t_fine / 2.0 - 64_000.0
    var2 = var1 * var1 * c.dig_p6 / 32_768.0
    var2 = var2 + var1 * c.dig_p5 * 2.0
    var2 = var2 / 4.0 + c.dig_p4 * 65_536.0
    var1 = (c.dig_p3 * var1 * var1 / 524_288.0 + c.dig_p2 * var1) / 524_288.0
    var1 = (1.0 + var1 / 32_768.0) * c.dig_p1

    if var1 == 0.0 do
      0.0
    else
      p = 1_048_576.0 - adc_p
      p = (p - var2 / 4096.0) * 6250.0 / var1
      var1 = c.dig_p9 * p * p / 2_147_483_648.0
      var2 = p * c.dig_p8 / 32_768.0
      p + (var1 + var2 + c.dig_p7) / 16.0
    end
  end

  @doc """
  Full compensation from calibration + raw block to
  `%{temp_c:, pressure_hpa:}`.
  """
  @spec compute(calibration(), non_neg_integer(), non_neg_integer()) ::
          %{temp_c: float(), pressure_hpa: float()}
  def compute(calibration, adc_p, adc_t) do
    {t_fine, temp_c} = compensate_temperature(calibration, adc_t)
    pressure_pa = compensate_pressure(calibration, adc_p, t_fine)
    %{temp_c: temp_c, pressure_hpa: pressure_pa / 100.0}
  end

  # ---- API ----

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc "Read calibration once (call at init)."
  @spec read_calibration(module(), I2C.bus(), I2C.address()) ::
          {:ok, calibration()} | {:error, term()}
  def read_calibration(impl, bus, address \\ @address) do
    case I2C.read_register(impl, bus, address, @reg_calib, @reg_calib_len) do
      {:ok, bin} -> parse_calibration(bin)
      {:error, _} = err -> err
    end
  end

  @doc "Configure normal mode (osrs_t x1, osrs_p x1)."
  @spec configure(module(), I2C.bus(), I2C.address()) :: :ok | {:error, term()}
  def configure(impl, bus, address \\ @address),
    do: I2C.write_register(impl, bus, address, @reg_ctrl_meas, <<@ctrl_meas_normal>>)

  @doc "Read + compensate one sample. Needs the calibration read at init."
  @spec read_once(module(), I2C.bus(), I2C.address(), calibration()) ::
          {:ok, %{temp_c: float(), pressure_hpa: float()}} | {:error, term()}
  def read_once(impl, bus, address \\ @address, calibration) do
    with {:ok, bin} <- I2C.read_register(impl, bus, address, @reg_data, @reg_data_len),
         {:ok, {adc_p, adc_t}} <- parse_raw(bin) do
      {:ok, compute(calibration, adc_p, adc_t)}
    end
  end

  # ---- Callbacks ----

  @impl true
  def init(opts) do
    base = %{
      impl: Keyword.get(opts, :bus_impl, AnimaBroker.Hardware.I2C.Circuits),
      bus: Keyword.fetch!(opts, :bus),
      address: Keyword.get(opts, :address, @address),
      store: Keyword.get(opts, :store, Store),
      interval_ms: Keyword.get(opts, :interval_ms, 2000)
    }

    _ = configure(base.impl, base.bus, base.address)

    case read_calibration(base.impl, base.bus, base.address) do
      {:ok, calib} ->
        send(self(), :read)
        {:ok, Map.put(base, :calibration, calib)}

      {:error, reason} ->
        Logger.error("[BMP280] calibration read failed: #{inspect(reason)} — sensor disabled")
        # No calibration => can't compensate. Stay up (so the supervisor doesn't
        # thrash) but don't schedule reads.
        {:ok, Map.put(base, :calibration, nil)}
    end
  end

  @impl true
  def handle_info(:read, %{calibration: nil} = state), do: {:noreply, state}

  def handle_info(:read, state) do
    case read_once(state.impl, state.bus, state.address, state.calibration) do
      {:ok, %{temp_c: t, pressure_hpa: p}} ->
        state.store.merge(%{"readings" => %{"pressure_hpa" => p, "pressure_temp_c" => t}})

      {:error, reason} ->
        Logger.warning("[BMP280] read failed: #{inspect(reason)}")
    end

    Process.send_after(self(), :read, state.interval_ms)
    {:noreply, state}
  end
end
