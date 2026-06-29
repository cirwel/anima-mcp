defmodule AnimaBroker.Tick do
  @moduledoc """
  The broker tick loop. Each tick bumps a heartbeat counter and flushes the
  current state to shared memory. Phase 1 will also drive sensor reads here
  (or sensors self-schedule and Tick just flushes).

  Uses `Process.send_after/3` rather than a busy loop — crash-safe and cheap.
  """
  use GenServer
  require Logger

  def start_link(_opts), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil) do
    schedule()
    {:ok, %{ticks: 0}}
  end

  @impl true
  def handle_info(:tick, %{ticks: n}) do
    ticks = n + 1

    AnimaBroker.State.Store.merge(%{
      "broker" => %{"impl" => "elixir", "phase" => 1, "ticks" => ticks}
    })

    _ = AnimaBroker.Shm.Writer.flush()
    schedule()
    {:noreply, %{ticks: ticks}}
  end

  defp schedule do
    interval = Application.get_env(:anima_broker, :tick_interval_ms, 2000)
    Process.send_after(self(), :tick, interval)
  end
end
