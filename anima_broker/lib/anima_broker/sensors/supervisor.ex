defmodule AnimaBroker.Sensors.Supervisor do
  @moduledoc """
  Supervises sensor reader processes — one GenServer per device, each reading
  via the `AnimaBroker.Hardware.I2C` behaviour and merging its slice into
  `AnimaBroker.State.Store`:

    * `AnimaBroker.Sensors.VEML7700` — light (gain 1x, 200ms integration)
    * `AnimaBroker.Sensors.AHT20`    — temperature / humidity
    * `AnimaBroker.Sensors.BMP280`   — pressure / temperature

  Sensors only start when an I2C bus is configured (`:i2c_bus`, e.g. `"i2c-1"`
  on the Pi). With no bus configured — dev, CI, tests — the tree starts empty,
  so the broker still boots and writes the (sensorless) envelope. This keeps
  Phase 1 hardware-optional: the conversion math is unit-tested with a fake bus
  regardless of whether real hardware is present.
  """
  use Supervisor
  require Logger

  def start_link(_opts), do: Supervisor.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil) do
    Supervisor.init(sensor_children(), strategy: :one_for_one)
  end

  defp sensor_children do
    bus_name = Application.get_env(:anima_broker, :i2c_bus)
    impl = Application.get_env(:anima_broker, :i2c_impl, AnimaBroker.Hardware.I2C.Circuits)

    if is_nil(bus_name) do
      []
    else
      case impl.open(bus_name) do
        {:ok, bus} ->
          Logger.info("[Sensors] I2C bus #{inspect(bus_name)} open via #{inspect(impl)}")
          common = [bus_impl: impl, bus: bus]

          [
            {AnimaBroker.Sensors.VEML7700, common},
            {AnimaBroker.Sensors.AHT20, common},
            {AnimaBroker.Sensors.BMP280, common}
          ]

        {:error, reason} ->
          Logger.error(
            "[Sensors] I2C bus #{inspect(bus_name)} open failed: #{inspect(reason)} — no sensors"
          )

          []
      end
    end
  end
end
