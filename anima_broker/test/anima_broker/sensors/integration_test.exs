defmodule AnimaBroker.Sensors.IntegrationTest do
  @moduledoc """
  Wiring test: a sensor GenServer driven by a fake bus must land its reading in
  the shared `State.Store` under the Python-contract key. async: false because
  Store is a named singleton.
  """
  use ExUnit.Case, async: false

  alias AnimaBroker.Sensors.VEML7700
  alias AnimaBroker.State.Store
  alias AnimaBroker.Test.FakeI2C

  setup do
    case Process.whereis(Store) do
      nil -> start_supervised!(Store)
      _ -> :ok
    end

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
  end
end
