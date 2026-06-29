defmodule AnimaBroker.Hardware.I2C.Circuits do
  @moduledoc """
  Real I2C bus, backed by `Circuits.I2C`. Thin pass-through so the rest of the
  broker only ever depends on the `AnimaBroker.Hardware.I2C` behaviour.

  On non-Linux hosts `Circuits.I2C` loads a stub backend (no buses), so this
  module still compiles in dev/CI; it just won't find a real bus to open.
  """
  @behaviour AnimaBroker.Hardware.I2C

  @impl true
  def open(bus_name), do: Circuits.I2C.open(bus_name)

  @impl true
  def read(bus, address, count), do: Circuits.I2C.read(bus, address, count)

  @impl true
  def write(bus, address, data), do: Circuits.I2C.write(bus, address, data)

  @impl true
  def write_read(bus, address, write_data, read_count),
    do: Circuits.I2C.write_read(bus, address, write_data, read_count)
end
