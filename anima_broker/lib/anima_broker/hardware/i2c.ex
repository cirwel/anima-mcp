defmodule AnimaBroker.Hardware.I2C do
  @moduledoc """
  Injectable I2C bus behaviour.

  Sensor drivers depend on this behaviour, not on `Circuits.I2C` directly, so
  they can be unit-tested with a fake bus and never touch hardware. The real
  implementation (`AnimaBroker.Hardware.I2C.Circuits`) wraps `Circuits.I2C`;
  tests inject a stub module that returns canned register bytes.

  All addresses are 7-bit. `read_register/4` / `write_register/4` cover the
  common "select register, then read/write" pattern; `read/3` and `write/3`
  are the raw primitives.
  """

  @type bus :: term()
  @type address :: 0..0x7F

  @callback open(bus_name :: String.t()) :: {:ok, bus()} | {:error, term()}
  @callback read(bus(), address(), count :: pos_integer()) ::
              {:ok, binary()} | {:error, term()}
  @callback write(bus(), address(), data :: binary()) :: :ok | {:error, term()}
  @callback write_read(bus(), address(), write_data :: binary(), read_count :: pos_integer()) ::
              {:ok, binary()} | {:error, term()}

  @doc "Read `count` bytes from `register` on `address` (write reg, then read)."
  @spec read_register(module(), bus(), address(), byte(), pos_integer()) ::
          {:ok, binary()} | {:error, term()}
  def read_register(impl, bus, address, register, count) do
    impl.write_read(bus, address, <<register>>, count)
  end

  @doc "Write `data` (a binary) to `register` on `address`."
  @spec write_register(module(), bus(), address(), byte(), binary()) :: :ok | {:error, term()}
  def write_register(impl, bus, address, register, data) do
    impl.write(bus, address, <<register>> <> data)
  end
end
