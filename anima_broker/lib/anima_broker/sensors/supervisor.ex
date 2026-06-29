defmodule AnimaBroker.Sensors.Supervisor do
  @moduledoc """
  Supervises sensor reader processes. Empty in Phase 0.

  Phase 1 adds one GenServer per device, each reading via `Circuits.I2C` and
  merging its slice into `AnimaBroker.State.Store`:

    * `AnimaBroker.Sensors.VEML7700` — light (gain 1x, 200ms integration)
    * `AnimaBroker.Sensors.AHT20`    — temperature / humidity
    * `AnimaBroker.Sensors.BMP280`   — pressure / temperature
  """
  use Supervisor

  def start_link(_opts), do: Supervisor.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil), do: Supervisor.init([], strategy: :one_for_one)
end
