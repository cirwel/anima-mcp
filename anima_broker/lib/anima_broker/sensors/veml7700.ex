defmodule AnimaBroker.Sensors.VEML7700 do
  @moduledoc """
  VEML7700 ambient light sensor (I2C 0x10).

  Matches the current Python config (`anima_mcp/sensors/pi.py`): **gain 1x,
  200ms integration** — chosen for indoor precision (50–500 lux) since the
  sensor sits next to the DotStar LEDs.

  ## Conversion

  Lux is the raw 16-bit ALS count times a resolution that depends on gain and
  integration time. This mirrors the `adafruit_veml7700` `.lux` property so the
  soak cross-check lines up:

      resolution = 0.0036 * (800 / integration_ms) * (2 / gain)
      lux        = raw_als * resolution

  At gain 1x / 200ms that is `0.0036 * 4 * 2 = 0.0288` lx/count.

  EMA smoothing is intentionally **not** done here — in the Python broker it
  lives one layer up (PiSensors), so the raw driver stays pure and the smoother
  can be ported separately.
  """
  use GenServer
  require Logger

  alias AnimaBroker.Hardware.I2C
  alias AnimaBroker.State.Store

  @address 0x10

  # Registers (16-bit, little-endian on the wire).
  @reg_als_conf 0x00
  @reg_als 0x04

  # ALS_CONF for gain 1x, 200ms, persistence 1, interrupt disabled, powered on:
  #   bits 12:11 (gain) = 00  -> 1x
  #   bits  9:6  (IT)   = 0001 -> 200ms
  #   bit   0    (SD)   = 0    -> powered on
  # => 0x0040
  @conf_gain_1x_200ms 0x0040

  @gain 1.0
  @integration_ms 200.0
  @integration_tolerance 0.30
  # Continuous mode exposes the most recently completed conversion. At an
  # arbitrary I2C read, its phase is unknown by up to one integration period;
  # Vishay also specifies +/-30% integration-time tolerance. This interval
  # conservatively bounds every possible photon-capture time represented by
  # the returned register value.
  @capture_support_ms round(2 * @integration_ms * (1 + @integration_tolerance))
  @capture_support_seconds @capture_support_ms / 1000.0

  # ---- Pure conversion (unit-tested) ----

  @doc "Resolution in lux/count for the given gain and integration time (ms)."
  @spec resolution(float(), float()) :: float()
  def resolution(gain \\ @gain, integration_ms \\ @integration_ms) do
    0.0036 * (800.0 / integration_ms) * (2.0 / gain)
  end

  @doc "Convert a raw 16-bit ALS count to lux at the configured gain/integration."
  @spec lux_from_raw(non_neg_integer(), float(), float()) :: float()
  def lux_from_raw(raw, gain \\ @gain, integration_ms \\ @integration_ms) do
    raw * resolution(gain, integration_ms)
  end

  @doc "The 16-bit value written to ALS_CONF for the configured mode."
  def conf_word, do: @conf_gain_1x_200ms

  @doc "Bound the capture-time support of a continuous-mode sensor read."
  def capture_provenance(completed_at \\ NaiveDateTime.local_now()) do
    window_start =
      NaiveDateTime.add(completed_at, -@capture_support_ms, :millisecond)

    %{
      "light_observed_at" => NaiveDateTime.to_iso8601(window_start),
      "light_observed_precision_seconds" => @capture_support_seconds
    }
  end

  # ---- API ----

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc "Read lux once. Returns `{:ok, lux}` or `{:error, reason}`. Testable with a fake bus."
  @spec read_once(module(), I2C.bus(), I2C.address()) :: {:ok, float()} | {:error, term()}
  def read_once(impl, bus, address \\ @address) do
    case I2C.read_register(impl, bus, address, @reg_als, 2) do
      {:ok, <<raw::little-16>>} -> {:ok, lux_from_raw(raw)}
      {:ok, other} -> {:error, {:short_read, other}}
      {:error, _} = err -> err
    end
  end

  @doc "Write the gain/integration config to the sensor."
  @spec configure(module(), I2C.bus(), I2C.address()) :: :ok | {:error, term()}
  def configure(impl, bus, address \\ @address) do
    # 16-bit little-endian: LSB first.
    I2C.write_register(impl, bus, address, @reg_als_conf, <<@conf_gain_1x_200ms::little-16>>)
  end

  # ---- Callbacks ----

  @impl true
  def init(opts) do
    state = %{
      impl: Keyword.get(opts, :bus_impl, AnimaBroker.Hardware.I2C.Circuits),
      bus: Keyword.fetch!(opts, :bus),
      address: Keyword.get(opts, :address, @address),
      store: Keyword.get(opts, :store, Store),
      interval_ms: Keyword.get(opts, :interval_ms, 2000)
    }

    case configure(state.impl, state.bus, state.address) do
      :ok -> :ok
      {:error, reason} -> Logger.warning("[VEML7700] configure failed: #{inspect(reason)}")
    end

    send(self(), :read)
    {:ok, state}
  end

  @impl true
  def handle_info(:read, state) do
    case read_once(state.impl, state.bus, state.address) do
      {:ok, lux} ->
        readings = Map.put(capture_provenance(), "light_lux", lux)
        state.store.merge(%{"readings" => readings})

      {:error, reason} ->
        # Publish absence, not the last good value (as BMP280 does). The
        # capture provenance goes too, so nothing downstream aligns a stale
        # lux against a fresh envelope.
        state.store.merge(%{
          "readings" => %{
            "light_lux" => nil,
            "light_observed_at" => nil,
            "light_observed_precision_seconds" => nil
          }
        })

        Logger.warning("[VEML7700] read failed: #{inspect(reason)} — publishing absent")
    end

    Process.send_after(self(), :read, state.interval_ms)
    {:noreply, state}
  end
end
