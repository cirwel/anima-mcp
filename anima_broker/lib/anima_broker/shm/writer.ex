defmodule AnimaBroker.Shm.Writer do
  @moduledoc """
  Writes the shared-memory envelope to disk so the Python MCP server can read
  it unchanged. Contract (see `anima_mcp/shared_memory.py`):

      {"updated_at": <iso8601>, "pid": <int>, "data": {...}}

  The Python reader is safe against torn reads because the write is atomic:
  write to `<path>.tmp`, fsync, then `rename()` over the target (atomic on
  POSIX). For a single writer, atomic rename alone is sufficient — the advisory
  `flock` the Python writer also takes is belt-and-suspenders and is deferred
  (only needed if a second writer is ever introduced).
  """
  use GenServer
  require Logger

  # ---- API ----

  def start_link(_opts), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)

  @doc "Snapshot `State.Store` and write it to the configured SHM path."
  def flush, do: GenServer.call(__MODULE__, :flush)

  @doc "Write an explicit `data` payload to `path`. Used by tests."
  def write(path, data), do: write_envelope(path, data)

  # ---- Callbacks ----

  @impl true
  def init(nil), do: {:ok, %{path: shm_path()}}

  @impl true
  def handle_call(:flush, _from, %{path: path} = st) do
    {:reply, write_envelope(path, AnimaBroker.State.Store.snapshot()), st}
  end

  # ---- Internal ----

  defp shm_path do
    Application.get_env(:anima_broker, :shm_path, "/dev/shm/anima_state.shadow.json")
  end

  @doc false
  def write_envelope(path, data) do
    envelope = %{"updated_at" => timestamp(), "pid" => os_pid(), "data" => data}
    json = Jason.encode!(envelope)
    tmp = path <> ".tmp"

    with :ok <- atomic_write(tmp, json),
         :ok <- File.rename(tmp, path) do
      :ok
    else
      {:error, reason} = err ->
        Logger.error("[Shm.Writer] write failed: #{inspect(reason)}")
        _ = File.rm(tmp)
        err
    end
  end

  # Write + fsync, mirroring the Python writer's flush() + os.fsync(). The
  # `:sync` open flag plus `:file.sync/1` ensures bytes hit disk before rename.
  defp atomic_write(tmp, json) do
    case :file.open(tmp, [:write, :raw, :binary, :sync]) do
      {:ok, fd} ->
        result =
          case :file.write(fd, json) do
            :ok -> :file.sync(fd)
            err -> err
          end

        :file.close(fd)
        result

      {:error, _} = err ->
        err
    end
  end

  # Python writes `datetime.now().isoformat()` — naive LOCAL time, no tz suffix.
  # `NaiveDateTime.local_now/0` matches that: it reads the OS clock in the
  # system-local timezone (the same one Python's naive `now()` uses), so on the
  # Pi both writers agree and the server's `updated_at` freshness checks don't
  # skew. (Phase 0 emitted UTC; harmless only because it wrote a shadow path.)
  # Second precision is fine — freshness is compared in whole seconds.
  defp timestamp, do: NaiveDateTime.to_iso8601(NaiveDateTime.local_now())

  defp os_pid, do: String.to_integer(System.pid())
end
