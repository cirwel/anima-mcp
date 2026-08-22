defmodule AnimaBroker.Governance.EisvMapperTest do
  use ExUnit.Case, async: true

  alias AnimaBroker.Governance.EisvMapper

  @anima %{"warmth" => 0.4, "clarity" => 0.8, "stability" => 0.7, "presence" => 0.6}

  test "eisv without neural bands mirrors the Python physical-only path" do
    eisv = EisvMapper.anima_to_eisv(@anima, %{})
    assert eisv["E"] == 0.4
    assert eisv["I"] == 0.8
    assert_in_delta eisv["S"], 0.3, 1.0e-9
    assert_in_delta eisv["V"], -0.4, 1.0e-9
  end

  test "eisv with neural bands blends 0.7 physical / 0.3 neural into E only" do
    readings = %{
      "eeg_alpha_power" => 0.9,
      "eeg_beta_power" => 0.5,
      "eeg_gamma_power" => 0.2
    }

    eisv = EisvMapper.anima_to_eisv(@anima, readings)
    # E = 0.7*0.4 + 0.3*(0.5*0.6 + 0.2*0.4) = 0.28 + 0.3*0.38 = 0.394
    assert_in_delta eisv["E"], 0.394, 1.0e-9
    # I = clarity. Alpha is NOT blended in (anima-mcp #141).
    assert_in_delta eisv["I"], 0.8, 1.0e-9
    assert_in_delta eisv["V"], eisv["E"] - eisv["I"], 1.0e-9
  end

  test "alpha never reaches I — CPU% must not land on both sides of V = E - I" do
    # alpha = 1 - beta by construction (computational_neural.py): both bands are
    # the same CPU reading. Blending alpha into I while beta feeds E double-counts
    # it and inflates |V|. Vary alpha across its whole range with beta/gamma held
    # fixed: E and I must both be unmoved, so V is unmoved.
    base = %{"eeg_beta_power" => 0.5, "eeg_gamma_power" => 0.2}

    eisvs =
      for a <- [0.0, 0.25, 0.5, 0.75, 1.0] do
        EisvMapper.anima_to_eisv(@anima, Map.put(base, "eeg_alpha_power", a))
      end

    assert Enum.map(eisvs, & &1["I"]) |> Enum.uniq() |> length() == 1
    assert Enum.map(eisvs, & &1["E"]) |> Enum.uniq() |> length() == 1
    assert Enum.map(eisvs, & &1["V"]) |> Enum.uniq() |> length() == 1
    assert_in_delta hd(eisvs)["I"], @anima["clarity"], 1.0e-9
  end

  test "alpha alone does not invent a zero activation signal" do
    # Alpha is the inverse of beta in computational proprioception, not an
    # independent activation band. Without beta or gamma, the physical warmth
    # reading remains authoritative instead of being damped by an imaginary 0.
    eisv = EisvMapper.anima_to_eisv(@anima, %{"eeg_alpha_power" => 0.9})
    assert_in_delta eisv["E"], 0.4, 1.0e-9
    assert_in_delta eisv["I"], 0.8, 1.0e-9
  end

  test "gamma alone is a real activation signal" do
    eisv = EisvMapper.anima_to_eisv(@anima, %{"eeg_gamma_power" => 0.5})
    assert_in_delta eisv["E"], 0.7 * 0.4 + 0.3 * (0.4 * 0.5), 1.0e-9
  end

  test "complexity matches the Python weights and clamps" do
    # (1-0.8)*0.25 + (1-0.7)*0.35 = 0.05 + 0.105 = 0.155
    assert_in_delta EisvMapper.estimate_complexity(@anima, %{}), 0.155, 1.0e-9

    # + cpu 50% * 0.10 + mem 40% * 0.05 = 0.155 + 0.05 + 0.02 = 0.225
    readings = %{"cpu_percent" => 50.0, "memory_percent" => 40.0}
    assert_in_delta EisvMapper.estimate_complexity(@anima, readings), 0.225, 1.0e-9
  end

  test "confidence penalizes rapid transitions, floors at 0.05" do
    # base = 0.8*0.5 + 0.7*0.3 + 0.6*0.2 = 0.73
    assert_in_delta EisvMapper.compute_confidence(@anima, nil), 0.73, 1.0e-9

    # big transition: delta = 0.3+0.3+0.3 = 0.9 > 0.15 → penalty min(0.75, 0.3) = 0.3
    prev = %{"warmth" => 0.7, "clarity" => 0.5, "stability" => 0.4, "presence" => 0.6}
    assert_in_delta EisvMapper.compute_confidence(@anima, prev), 0.43, 1.0e-9
  end

  test "ethical drift is zero on first check-in, scaled 3x after" do
    assert EisvMapper.compute_ethical_drift(@anima, nil, %{}, %{}) == [0.0, 0.0, 0.0]

    prev = %{"warmth" => 0.35, "clarity" => 0.85, "stability" => 0.7, "presence" => 0.6}
    [dw, dc, ds] = EisvMapper.compute_ethical_drift(@anima, prev, %{}, %{})
    assert_in_delta dw, 0.15, 1.0e-9
    assert_in_delta dc, -0.15, 1.0e-9
    assert_in_delta ds, 0.0, 1.0e-9
  end

  test "environment must not amplify drift — the deltas already carry it once" do
    # This test used to assert the amplifier (d_warmth 0.2 ×3 ×1.5 → clamp 0.5).
    # That encoded the double-count as the contract: post-#173 warmth IS thermal
    # state, so a temperature change amplifying the warmth delta counted one
    # quantity quadratically; and the lux path read Lumen's own LED glow, so its
    # own activity transitions amplified the drift about themselves.
    prev = %{"warmth" => 0.2, "clarity" => 0.8, "stability" => 0.7, "presence" => 0.6}

    calm =
      {%{"ambient_temp_c" => 23.1, "light_lux" => 100.0},
       %{"ambient_temp_c" => 23.0, "light_lux" => 100.0}}

    stormy =
      {%{"ambient_temp_c" => 28.0, "light_lux" => 500.0},
       %{"ambient_temp_c" => 23.0, "light_lux" => 100.0}}

    {r1, p1} = calm
    {r2, p2} = stormy
    drift_calm = EisvMapper.compute_ethical_drift(@anima, prev, r1, p1)
    drift_stormy = EisvMapper.compute_ethical_drift(@anima, prev, r2, p2)

    # Same anima deltas ⇒ same drift, whatever the environment did between
    # check-ins — the environment reaches drift through the deltas only.
    assert drift_calm == drift_stormy
    # d_warmth = 0.2 ×3 = 0.6 → clamp 0.5 (clamp retained; it is a cap, not a gate)
    assert hd(drift_calm) == 0.5
  end

  test "status text carries anima and EISV readouts" do
    eisv = EisvMapper.anima_to_eisv(@anima, %{})
    text = EisvMapper.status_text(@anima, eisv)
    assert text =~ "Warmth: 0.40"
    assert text =~ "EISV: E=0.40"
  end
end
