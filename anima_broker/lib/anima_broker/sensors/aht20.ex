defmodule AnimaBroker.Sensors.AHT20 do
  @moduledoc """
  AHT20 temperature + humidity sensor (I2C 0x38).

  Protocol (datasheet): initialize once (0xBE 0x08 0x00), then per reading
  trigger a measurement (0xAC 0x33 0x00), wait ~80ms, and read 6+ bytes:

      [status, h[19:12], h[11:4], h[3:0]|t[19:16], t[15:8], t[7:0], (crc)]

      humidity_pct = raw_h / 2^20 * 100
      temp_c       = raw_t / 2^20 * 200 - 50

  Output fields match the Python contract: `ambient_temp_c`, `humidity_pct`.
  """
  use GenServer
  require Logger
  import Bitwise

  alias AnimaBroker.Hardware.I2C
  alias AnimaBroker.State.Store

  @address 0x38

  @cmd_init <<0xBE, 0x08, 0x00>>
  @cmd_trigger <<0xAC, 0x33, 0x00>>
  @status_busy 0x80
  @scale 1_048_576.0

  # ---- Pure decode (unit-tested) ----

  @doc """
  Decode a raw AHT20 reading (6 or 7 bytes) into `%{temp_c:, humidity_pct:}`.
  Returns `{:error, :busy}` if the status byte still has the busy bit set.
  """
  @spec decode(binary()) :: {:ok, %{temp_c: float(), humidity_pct: float()}} | {:error, term()}
  def decode(<<status, b1, b2, b3, b4, b5, _rest::binary>>) do
    if (status &&& @status_busy) != 0 do
      {:error, :busy}
    else
      raw_h = b1 <<< 12 ||| b2 <<< 4 ||| b3 >>> 4
      raw_t = (b3 &&& 0x0F) <<< 16 ||| b4 <<< 8 ||| b5

      {:ok,
       %{
         humidity_pct: raw_h / @scale * 100.0,
         temp_c: raw_t / @scale * 200.0 - 50.0
       }}
    end
  end

  def decode(_), do: {:error, :short_read}

  # ---- API ----

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc "Initialize (calibrate) the sensor."
  @spec init_sensor(module(), I2C.bus(), I2C.address()) :: :ok | {:error, term()}
  def init_sensor(impl, bus, address \\ @address), do: impl.write(bus, address, @cmd_init)

  @doc """
  Trigger a measurement, wait `delay_ms`, then read and decode. `delay_ms` is
  injectable so tests can pass 0 against a fake bus.
  """
  @spec read_once(module(), I2C.bus(), I2C.address(), non_neg_integer()) ::
          {:ok, %{temp_c: float(), humidity_pct: float()}} | {:error, term()}
  def read_once(impl, bus, address \\ @address, delay_ms \\ 80) do
    with :ok <- impl.write(bus, address, @cmd_trigger),
         _ <- if(delay_ms > 0, do: Process.sleep(delay_ms)),
         {:ok, bytes} <- impl.read(bus, address, 7) do
      decode(bytes)
    end
  end

  # ---- Callbacks ----

  @impl true
  def init(opts) do
    state = %{
      impl: Keyword.get(opts, :bus_impl, AnimaBroker.Hardware.I2C.Circuits),
      bus: Keyword.fetch!(opts, :bus),
      address: Keyword.get(opts, :address, @address),
      store: Keyword.get(opts, :store, Store),
      interval_ms: Keyword.get(opts, :interval_ms, 2000),
      delay_ms: Keyword.get(opts, :delay_ms, 80)
    }

    case init_sensor(state.impl, state.bus, state.address) do
      :ok -> :ok
      {:error, reason} -> Logger.warning("[AHT20] init failed: #{inspect(reason)}")
    end

    send(self(), :read)
    {:ok, state}
  end

  @impl true
  def handle_info(:read, state) do
    case read_once(state.impl, state.bus, state.address, state.delay_ms) do
      {:ok, %{temp_c: t, humidity_pct: h}} ->
        state.store.merge(%{"readings" => %{"ambient_temp_c" => t, "humidity_pct" => h}})

      {:error, reason} ->
        Logger.warning("[AHT20] read failed: #{inspect(reason)}")
    end

    Process.send_after(self(), :read, state.interval_ms)
    {:noreply, state}
  end
end
