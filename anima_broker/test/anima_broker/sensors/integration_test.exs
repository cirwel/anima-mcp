defmodule AnimaBroker.Sensors.IntegrationTest do
  @moduledoc """
  Wiring test: a sensor GenServer driven by a fake bus must land its reading in
  the shared `State.Store` under the Python-contract key. async: false because
  Store is a named singleton.
  """
  use ExUnit.Case, async: false

  alias AnimaBroker.Sensors.AHT20
  alias AnimaBroker.Sensors.VEML7700
  alias AnimaBroker.State.Store
  alias AnimaBroker.Test.FakeI2C

  setup do
    case Process.whereis(Store) do
      nil -> start_supervised!(Store)
      _ -> :ok
    end

    # Store is the application's shared singleton, so other modules leave
    # readings in it (store_test merges light_lux 123.0). Clear the keys these
    # tests poll, or a test can pick up another module's value instead of its
    # sensor's. Snapshot is a call, so it serializes after this cast.
    Store.merge(%{
      "readings" => %{
        "light_lux" => nil,
        "light_observed_at" => nil,
        "light_observed_precision_seconds" => nil,
        "ambient_temp_c" => nil,
        "humidity_pct" => nil
      }
    })

    _ = Store.snapshot()
    :ok
  end

  test "VEML7700 GenServer merges light_lux into the Store" do
    bus = %{responses: %{{:write_read, 0x04, 2} => {:ok, <<0x88, 0x13>>}}}
    expected = 5000 * 0.0288

    start_supervised!({VEML7700, bus_impl: FakeI2C, bus: bus, interval_ms: 50})

    # first read fires immediately (init -> send(self(), :read)); poll briefly
    lux =
      Enum.reduce_while(1..50, nil, fn _, _ ->
        case Store.snapshot()["readings"]["light_lux"] do
          nil -> Process.sleep(10) && {:cont, nil}
          v -> {:halt, v}
        end
      end)

    assert is_float(lux)
    assert_in_delta lux, expected, 1.0e-6
    readings = Store.snapshot()["readings"]
    assert {:ok, _observed_at} = NaiveDateTime.from_iso8601(readings["light_observed_at"])
    assert readings["light_observed_precision_seconds"] == 0.52
  end

  # A dead sensor must publish absence, not keep its last good value. The bus
  # below has no canned read, so every read fails.
  test "VEML7700 read failure replaces the last good light reading with nil" do
    Store.merge(%{
      "readings" => %{
        "light_lux" => 123.4,
        "light_observed_at" => "2026-11-12T09:00:00",
        "light_observed_precision_seconds" => 0.52
      }
    })

    start_supervised!({VEML7700, bus_impl: FakeI2C, bus: %{responses: %{}}, interval_ms: 50})

    readings =
      wait_for_readings(fn r ->
        Map.fetch(r, "light_lux") == {:ok, nil}
      end)

    assert Map.fetch!(readings, "light_lux") == nil
    assert Map.fetch!(readings, "light_observed_at") == nil
    assert Map.fetch!(readings, "light_observed_precision_seconds") == nil
  end

  test "AHT20 read failure replaces the last good temp and humidity with nil" do
    Store.merge(%{"readings" => %{"ambient_temp_c" => 21.5, "humidity_pct" => 40.0}})

    start_supervised!(
      {AHT20, bus_impl: FakeI2C, bus: %{responses: %{}}, interval_ms: 50, delay_ms: 0}
    )

    readings =
      wait_for_readings(fn r ->
        Map.fetch(r, "ambient_temp_c") == {:ok, nil}
      end)

    assert Map.fetch!(readings, "ambient_temp_c") == nil
    assert Map.fetch!(readings, "humidity_pct") == nil
  end

  test "capture provenance bounds continuous-read phase and integration tolerance" do
    completed_at = ~N[2026-08-23 12:00:00.500000]

    assert VEML7700.capture_provenance(completed_at) == %{
             "light_observed_at" => "2026-08-23T11:59:59.980000",
             "light_observed_precision_seconds" => 0.52
           }
  end

  defp wait_for_readings(pred) do
    Enum.reduce_while(1..50, nil, fn _, _ ->
      readings = Store.snapshot()["readings"]

      if pred.(readings) do
        {:halt, readings}
      else
        Process.sleep(10)
        {:cont, readings}
      end
    end)
  end
end
