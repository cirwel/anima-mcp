defmodule AnimaBroker.Sensors.BMP280Test do
  use ExUnit.Case, async: true
  import Bitwise

  alias AnimaBroker.Sensors.BMP280
  alias AnimaBroker.Test.FakeI2C

  # Datasheet Appendix-A reference trimming values (BST-BMP280-DS001).
  @t1 27504
  @t2 26435
  @t3 -1000
  @p1 36477
  @p2 -10685
  @p3 3024
  @p4 2855
  @p5 140
  @p6 -7
  @p7 15500
  @p8 -14600
  @p9 6000

  # Reference raw ADC values from the same appendix.
  @adc_t 519_888
  @adc_p 415_148

  # 24-byte calibration block as it appears on the wire (little-endian, with the
  # correct signedness baked into the bytes). Parsing this back MUST recover the
  # signed negatives — that is the signed/unsigned trap.
  defp calib_bin do
    <<@t1::little-unsigned-16, @t2::little-signed-16, @t3::little-signed-16,
      @p1::little-unsigned-16, @p2::little-signed-16, @p3::little-signed-16,
      @p4::little-signed-16, @p5::little-signed-16, @p6::little-signed-16, @p7::little-signed-16,
      @p8::little-signed-16, @p9::little-signed-16>>
  end

  # 6-byte measurement block encoding @adc_p (first) then @adc_t.
  defp raw_bin do
    <<@adc_p >>> 12 &&& 0xFF, @adc_p >>> 4 &&& 0xFF, @adc_p <<< 4 &&& 0xF0,
      @adc_t >>> 12 &&& 0xFF, @adc_t >>> 4 &&& 0xFF, @adc_t <<< 4 &&& 0xF0>>
  end

  describe "parse_calibration (signed/unsigned mix)" do
    test "recovers unsigned T1/P1 and signed everything else" do
      assert {:ok, c} = BMP280.parse_calibration(calib_bin())
      assert c.dig_t1 == @t1
      assert c.dig_t2 == @t2
      # the trap: T3 is signed; an unsigned parse would yield 64536, not -1000
      assert c.dig_t3 == @t3
      assert c.dig_p1 == @p1
      assert c.dig_p2 == @p2
      assert c.dig_p6 == @p6
      assert c.dig_p8 == @p8
      assert c.dig_p9 == @p9
    end

    test "rejects a short calibration block" do
      assert {:error, :bad_calibration} = BMP280.parse_calibration(<<0, 0, 0>>)
    end
  end

  describe "parse_raw" do
    test "extracts 20-bit adc_p and adc_t" do
      assert {:ok, {adc_p, adc_t}} = BMP280.parse_raw(raw_bin())
      assert adc_p == @adc_p
      assert adc_t == @adc_t
    end

    test "rejects a short measurement block" do
      assert {:error, :short_read} = BMP280.parse_raw(<<0, 0, 0>>)
    end
  end

  describe "compensation against the datasheet reference vector" do
    setup do
      {:ok, c} = BMP280.parse_calibration(calib_bin())
      %{calib: c}
    end

    test "temperature compensates to ~25.08 C", %{calib: c} do
      {_t_fine, temp_c} = BMP280.compensate_temperature(c, @adc_t)
      assert_in_delta temp_c, 25.08, 0.01
    end

    test "pressure compensates to ~100653 Pa (~1006.5 hPa)", %{calib: c} do
      {t_fine, _} = BMP280.compensate_temperature(c, @adc_t)
      pa = BMP280.compensate_pressure(c, @adc_p, t_fine)
      # datasheet double reference: ~100653.27 Pa
      assert_in_delta pa, 100_653.27, 0.5
    end

    test "compute/3 returns both, pressure in hPa", %{calib: c} do
      result = BMP280.compute(c, @adc_p, @adc_t)
      assert_in_delta result.temp_c, 25.08, 0.01
      assert_in_delta result.pressure_hpa, 1006.53, 0.01
    end
  end

  describe "read_once with a fake bus" do
    test "reads the measurement block and compensates" do
      {:ok, calib} = BMP280.parse_calibration(calib_bin())
      bus = %{responses: %{{:write_read, 0xF7, 6} => {:ok, raw_bin()}}}

      assert {:ok, %{temp_c: t, pressure_hpa: p}} = BMP280.read_once(FakeI2C, bus, 0x76, calib)
      assert_in_delta t, 25.08, 0.01
      assert_in_delta p, 1006.53, 0.01
    end
  end

  describe "read_calibration with a fake bus" do
    test "reads 24 bytes from 0x88 and parses" do
      bus = %{responses: %{{:write_read, 0x88, 24} => {:ok, calib_bin()}}}
      assert {:ok, c} = BMP280.read_calibration(FakeI2C, bus, 0x76)
      assert c.dig_t3 == @t3
    end
  end
end
