defmodule AnimaBroker.State.Store do
  @moduledoc """
  Holds the current broker state — the `data` payload that gets written to
  shared memory. Sensors and subsystems merge their slices in; the SHM writer
  snapshots it. Keys are strings so the JSON envelope matches the Python
  contract exactly (see docs/plans/2026-06-29-elixir-broker-migration.md).
  """
  use GenServer

  # ---- API ----

  def start_link(_opts), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)

  @doc """
  Deep-merge a slice into the current state, e.g.
  `merge(%{"readings" => %{"light_lux" => 882.0}})`.
  """
  def merge(slice) when is_map(slice), do: GenServer.cast(__MODULE__, {:merge, slice})

  @doc "Return the full `data` payload snapshot."
  def snapshot, do: GenServer.call(__MODULE__, :snapshot)

  # ---- Callbacks ----

  @impl true
  def init(seed), do: {:ok, deep_merge(initial(), seed)}

  @impl true
  def handle_cast({:merge, slice}, state), do: {:noreply, deep_merge(state, slice)}

  @impl true
  def handle_call(:snapshot, _from, state), do: {:reply, state, state}

  # ---- Internal ----

  # Mirrors the documented `data` shape so a reader sees a recognizable payload
  # even before real readings exist in Phase 1.
  defp initial do
    %{
      "readings" => %{},
      "anima" => %{},
      "wifi_connected" => nil,
      "activity" => %{"level" => "unknown", "reason" => "boot"},
      "learning" => %{},
      "broker" => %{"impl" => "elixir", "phase" => 1, "ticks" => 0}
    }
  end

  defp deep_merge(a, b) do
    Map.merge(a, b, fn
      _k, %{} = va, %{} = vb -> deep_merge(va, vb)
      _k, _va, vb -> vb
    end)
  end
end
