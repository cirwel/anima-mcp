defmodule AnimaBroker.Test.FakeI2C do
  @moduledoc """
  In-memory `AnimaBroker.Hardware.I2C` for tests — no hardware, no NIF.

  The "bus" is a plain map of canned responses, keyed by the operation:

      %{
        responses: %{
          {:write_read, register_byte, count} => {:ok, <<...>>},
          {:read, count}                       => {:ok, <<...>>}
        },
        trace: self()   # optional: receive {:i2c_write, address, data}
      }

  Writes always succeed (and are optionally traced). Reads/write_reads return
  the canned response, or `{:error, {:no_canned_response, key}}` if absent — so
  a driver asking for something the test didn't stub fails loudly.
  """
  @behaviour AnimaBroker.Hardware.I2C

  @impl true
  def open(config) when is_map(config), do: {:ok, config}
  def open(_), do: {:ok, %{responses: %{}}}

  @impl true
  def write(bus, address, data) do
    if pid = bus[:trace], do: send(pid, {:i2c_write, address, data})
    :ok
  end

  @impl true
  def read(bus, _address, count), do: fetch(bus, {:read, count})

  @impl true
  def write_read(bus, _address, <<register, _::binary>>, count),
    do: fetch(bus, {:write_read, register, count})

  defp fetch(bus, key) do
    case Map.fetch(bus[:responses] || %{}, key) do
      {:ok, resp} -> resp
      :error -> {:error, {:no_canned_response, key}}
    end
  end
end
