defmodule AnimaBroker.Sensors.AHT20Test do
  use ExUnit.Case, async: true
  import Bitwise

  alias AnimaBroker.Sensors.AHT20
  alias AnimaBroker.Test.FakeI2C

  describe "decode" do
    test "midscale raw -> 50% RH, 50C" do
      # raw_h = 0x80000 (half of 2^20) -> 50%; raw_t = 0x80000 -> 200*0.5-50 = 50C
      # bytes: status, h[19:12]=0x80, h[11:4]=0x00, h[3:0]|t[19:16]=0x08, t[15:8], t[7:0]
      bytes = <<0x18, 0x80, 0x00, 0x08, 0x00, 0x00>>
      assert {:ok, %{humidity_pct: h, temp_c: t}} = AHT20.decode(bytes)
      assert_in_delta h, 50.0, 1.0e-6
      assert_in_delta t, 50.0, 1.0e-6
    end

    test "all-zero raw -> 0% RH, -50C (datasheet floor)" do
      bytes = <<0x1C, 0x00, 0x00, 0x00, 0x00, 0x00>>
      assert {:ok, %{humidity_pct: +0.0, temp_c: -50.0}} = AHT20.decode(bytes)
    end

    test "splits the shared nibble byte correctly across humidity and temp" do
      # h nibble low = 0xA, t nibble high = 0x5 in the shared byte 0xA5
      # raw_h = 0x12_3_A? build explicitly: b1=0x12 b2=0x34 b3=0xA5
      # raw_h = 0x12<<12 | 0x34<<4 | 0xA = 0x12340 | ... compute in assertion
      bytes = <<0x18, 0x12, 0x34, 0xA5, 0x67, 0x89>>
      raw_h = 0x12 <<< 12 ||| 0x34 <<< 4 ||| 0xA5 >>> 4
      raw_t = (0xA5 &&& 0x0F) <<< 16 ||| 0x67 <<< 8 ||| 0x89
      assert {:ok, %{humidity_pct: h, temp_c: t}} = AHT20.decode(bytes)
      assert_in_delta h, raw_h / 1_048_576.0 * 100.0, 1.0e-9
      assert_in_delta t, raw_t / 1_048_576.0 * 200.0 - 50.0, 1.0e-9
    end

    test "busy status bit -> error" do
      assert {:error, :busy} = AHT20.decode(<<0x80, 0x80, 0x00, 0x08, 0x00, 0x00>>)
    end

    test "short read -> error" do
      assert {:error, :short_read} = AHT20.decode(<<0x18, 0x00>>)
    end

    test "tolerates a trailing CRC byte (7-byte read)" do
      bytes = <<0x18, 0x80, 0x00, 0x08, 0x00, 0x00, 0xAB>>
      assert {:ok, %{humidity_pct: h}} = AHT20.decode(bytes)
      assert_in_delta h, 50.0, 1.0e-6
    end
  end

  describe "read_once with a fake bus" do
    test "triggers, reads 7 bytes, decodes" do
      bus = %{
        trace: self(),
        responses: %{{:read, 7} => {:ok, <<0x18, 0x80, 0x00, 0x08, 0x00, 0x00, 0x00>>}}
      }

      assert {:ok, %{humidity_pct: h, temp_c: t}} = AHT20.read_once(FakeI2C, bus, 0x38, 0)
      assert_in_delta h, 50.0, 1.0e-6
      assert_in_delta t, 50.0, 1.0e-6
      # the trigger command was issued
      assert_receive {:i2c_write, 0x38, <<0xAC, 0x33, 0x00>>}
    end
  end
end
