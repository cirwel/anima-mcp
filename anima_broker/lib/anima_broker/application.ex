defmodule AnimaBroker.Application do
  @moduledoc """
  OTP application root. Boots the broker supervision tree.

  Phase 0 children:
    * `State.Store`        — holds the current `data` payload (the SHM body)
    * `Sensors.Supervisor` — empty stub; Phase 1 adds I2C sensor readers
    * `Shm.Writer`         — writes the envelope atomically to the SHM path
    * `Tick`               — the broker tick loop (gated by :autostart)

  `:one_for_one` so a crashed child is restarted in isolation — this is the
  whole point of moving the broker onto OTP.
  """
  use Application

  @impl true
  def start(_type, _args) do
    children =
      [
        AnimaBroker.State.Store,
        AnimaBroker.Sensors.Supervisor,
        AnimaBroker.Shm.Writer
      ] ++ tick_child()

    Supervisor.start_link(children, strategy: :one_for_one, name: AnimaBroker.Supervisor)
  end

  defp tick_child do
    if Application.get_env(:anima_broker, :autostart, true), do: [AnimaBroker.Tick], else: []
  end
end
