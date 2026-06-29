defmodule AnimaBroker.State.StoreTest do
  use ExUnit.Case, async: false

  alias AnimaBroker.State.Store

  setup do
    # Store is a named singleton started by the application. Start it here if
    # the app didn't (autostart is false in :test, but the supervision tree
    # still starts Store + Sensors.Supervisor + Writer).
    case Process.whereis(Store) do
      nil -> start_supervised!(Store)
      _ -> :ok
    end

    :ok
  end

  test "snapshot returns the seeded data shape" do
    snap = Store.snapshot()
    assert Map.has_key?(snap, "readings")
    assert Map.has_key?(snap, "anima")
    assert Map.has_key?(snap, "broker")
  end

  test "merge deep-merges nested slices" do
    Store.merge(%{"readings" => %{"light_lux" => 123.0}})
    Store.merge(%{"readings" => %{"cpu_temp_c" => 50.0}})
    # cast is async; snapshot is a call, which serializes after the casts.
    snap = Store.snapshot()
    assert snap["readings"]["light_lux"] == 123.0
    assert snap["readings"]["cpu_temp_c"] == 50.0
  end
end
