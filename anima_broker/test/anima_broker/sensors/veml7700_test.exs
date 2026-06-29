defmodule AnimaBroker.Sensors.VEML7700Test do
  use ExUnit.Case, async: true

  alias AnimaBroker.Sensors.VEML7700
  alias AnimaBroker.Test.FakeI2C

  describe "resolution / lux conversion" do
    test "gain 1x, 200ms integration gives 0.0288 lx/count (matches adafruit)" do
      assert_in_delta VEML7700.resolution(1.0, 200.0), 0.0288, 1.0e-9
    end

    test "resolution scales inversely with gain and integration time" do
      # max gain (2x) + max integration (800ms) is the datasheet base, 0.0036
      assert_in_delta VEML7700.resolution(2.0, 800.0), 0.0036, 1.0e-9
      # halving integration doubles lx/count
      assert_in_delta VEML7700.resolution(1.0, 100.0), 0.0576, 1.0e-9
    end

    test "lux_from_raw multiplies raw count by resolution" do
      # 1000 counts at 0.0288 lx/count = 28.8 lux
      assert_in_delta VEML7700.lux_from_raw(1000), 28.8, 1.0e-6
      assert VEML7700.lux_from_raw(0) == 0.0
    end
  end

  describe "conf_word" do
    test "encodes gain 1x, 200ms, powered on" do
      # bits 12:11 = 00 (gain 1x), bits 9:6 = 0001 (200ms), bit 0 = 0 (on)
      assert VEML7700.conf_word() == 0x0040
    end
  end

  describe "read_once with a fake bus" do
    test "reads the ALS register (little-endian) and converts to lux" do
      # raw ALS = 5000 (0x1388) little-endian on the wire: <<0x88, 0x13>>
      bus = %{responses: %{{:write_read, 0x04, 2} => {:ok, <<0x88, 0x13>>}}}
      assert {:ok, lux} = VEML7700.read_once(FakeI2C, bus)
      assert_in_delta lux, 5000 * 0.0288, 1.0e-6
    end

    test "propagates a bus error" do
      bus = %{responses: %{{:write_read, 0x04, 2} => {:error, :i2c_nak}}}
      assert {:error, :i2c_nak} = VEML7700.read_once(FakeI2C, bus)
    end
  end

  describe "configure" do
    test "writes the 16-bit conf word little-endian to ALS_CONF" do
      bus = %{trace: self()}
      assert :ok = VEML7700.configure(FakeI2C, bus, 0x10)
      # write_register prepends the register byte 0x00, then LSB, MSB of 0x0040
      assert_receive {:i2c_write, 0x10, <<0x00, 0x40, 0x00>>}
    end
  end
end
