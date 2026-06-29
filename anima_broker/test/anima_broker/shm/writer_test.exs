defmodule AnimaBroker.Shm.WriterTest do
  @moduledoc """
  Phase 0 acceptance: the Elixir writer must emit an envelope the Python
  `SharedMemoryReader` can parse. These tests assert the contract shape and the
  atomic-write behavior. Cross-language validation (feeding the file to the
  actual Python reader) is documented in the project README as a manual check.
  """
  use ExUnit.Case, async: true

  alias AnimaBroker.Shm.Writer

  @data %{
    "readings" => %{"cpu_temp_c" => 56.4, "light_lux" => 882.0},
    "anima" => %{"warmth" => 0.45, "clarity" => 0.74, "stability" => 0.75, "presence" => 0.74},
    "wifi_connected" => true,
    "activity" => %{"level" => "active", "reason" => "engaged"},
    "learning" => %{"preferences" => %{"satisfaction" => 0.87}}
  }

  defp tmp_path do
    Path.join(System.tmp_dir!(), "anima_state_#{System.unique_integer([:positive])}.json")
  end

  test "writes a contract-shaped envelope" do
    path = tmp_path()
    on_exit(fn -> File.rm(path); File.rm(path <> ".tmp") end)

    assert :ok = Writer.write(path, @data)

    envelope = path |> File.read!() |> Jason.decode!()
    assert Map.has_key?(envelope, "updated_at")
    assert Map.has_key?(envelope, "pid")
    assert Map.has_key?(envelope, "data")
    assert is_integer(envelope["pid"])
    assert envelope["data"] == @data
  end

  test "leaves no temp file behind (atomic rename completed)" do
    path = tmp_path()
    on_exit(fn -> File.rm(path); File.rm(path <> ".tmp") end)

    assert :ok = Writer.write(path, @data)
    assert File.exists?(path)
    refute File.exists?(path <> ".tmp")
  end

  test "updated_at is ISO8601 and naive (no timezone suffix, matching Python isoformat)" do
    path = tmp_path()
    on_exit(fn -> File.rm(path) end)

    assert :ok = Writer.write(path, @data)
    envelope = path |> File.read!() |> Jason.decode!()

    assert {:ok, _naive} = NaiveDateTime.from_iso8601(envelope["updated_at"])
    refute String.ends_with?(envelope["updated_at"], "Z")
  end

  test "overwrites an existing file atomically" do
    path = tmp_path()
    on_exit(fn -> File.rm(path); File.rm(path <> ".tmp") end)

    assert :ok = Writer.write(path, %{"v" => 1})
    assert :ok = Writer.write(path, %{"v" => 2})

    envelope = path |> File.read!() |> Jason.decode!()
    assert envelope["data"] == %{"v" => 2}
  end
end
