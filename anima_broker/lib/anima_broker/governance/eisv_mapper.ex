defmodule AnimaBroker.Governance.EisvMapper do
  @moduledoc """
  Anima → EISV mapping for governance check-ins.

  Faithful port of `src/anima_mcp/eisv_mapper.py` (the Python bridge's
  formulas). Since the Phase-2 cutover this module is the SOLE producer of
  Lumen's governance EISV -- the Python bridge runs in passthrough mode
  (`ANIMA_GOVERNANCE_FROM_SHM`) and reads the verdict this client writes, so a
  formula that drifts here is not a shadow discrepancy, it is what governance
  records. Formula changes belong in BOTH files until the Python bridge
  retires; `eisv_mapper_test.exs` pins the shared algebra.

  Inputs are plain maps read from the live SHM envelope: `anima` with
  "warmth"/"clarity"/"stability"/"presence" floats, `readings` with optional
  "eeg_alpha_power"/"eeg_beta_power"/"eeg_gamma_power"/"cpu_percent"/
  "memory_percent"/"ambient_temp_c"/"cpu_temp_c"/"light_lux".
  """

  @neural_weight 0.3
  @physical_weight 0.7

  @doc "Map anima + readings to %{\"E\"|\"I\"|\"S\"|\"V\" => float}."
  def anima_to_eisv(anima, readings) do
    warmth = Map.get(anima, "warmth", 0.5)
    clarity = Map.get(anima, "clarity", 0.5)
    stability = Map.get(anima, "stability", 0.5)

    beta = Map.get(readings, "eeg_beta_power")
    gamma = Map.get(readings, "eeg_gamma_power")
    has_activation = beta != nil or gamma != nil

    e =
      if has_activation do
        neural_energy = (beta || 0) * 0.6 + (gamma || 0) * 0.4
        @physical_weight * warmth + @neural_weight * neural_energy
      else
        warmth
      end

    # Integrity (I): clarity only. Alpha is deliberately NOT mixed in --
    # alpha = 1 - beta by construction (computational_neural.py), so feeding
    # alpha into I while beta feeds E puts CPU% on both sides of V = E - I,
    # the double-count CLAUDE.md warns neural consumers about. Kept in sync
    # with the Python bridge's eisv_mapper.py (anima-mcp #141).
    i = clarity

    e = clamp(e, 0.0, 1.0)
    i = clamp(i, 0.0, 1.0)
    s = clamp(1.0 - stability, 0.0, 1.0)
    v = clamp(e - i, -1.0, 1.0)

    %{"E" => e, "I" => i, "S" => s, "V" => v}
  end

  @doc "Task-complexity estimate in [0, 1]."
  def estimate_complexity(anima, readings) do
    clarity = Map.get(anima, "clarity", 0.5)
    stability = Map.get(anima, "stability", 0.5)

    complexity = (1.0 - clarity) * 0.25 + (1.0 - stability) * 0.35

    complexity =
      case Map.get(readings, "cpu_percent") do
        nil -> complexity
        cpu -> complexity + cpu / 100.0 * 0.10
      end

    complexity =
      case Map.get(readings, "memory_percent") do
        nil -> complexity
        mem -> complexity + mem / 100.0 * 0.05
      end

    clamp(complexity, 0.0, 1.0)
  end

  @doc "Confidence in [0.05, 1.0]; penalizes rapid anima transitions."
  def compute_confidence(anima, prev_anima) do
    confidence =
      Map.get(anima, "clarity", 0.5) * 0.5 +
        Map.get(anima, "stability", 0.5) * 0.3 +
        Map.get(anima, "presence", 0.5) * 0.2

    confidence =
      if prev_anima do
        total_delta =
          abs(Map.get(anima, "warmth", 0.5) - Map.get(prev_anima, "warmth", 0.5)) +
            abs(Map.get(anima, "clarity", 0.5) - Map.get(prev_anima, "clarity", 0.5)) +
            abs(Map.get(anima, "stability", 0.5) - Map.get(prev_anima, "stability", 0.5))

        if total_delta > 0.15 do
          confidence - min(total_delta - 0.15, 0.3)
        else
          confidence
        end
      else
        confidence
      end

    clamp(confidence, 0.05, 1.0)
  end

  @doc """
  Ethical drift [Δη₀, Δη₁, Δη₂] from anima deltas, scaled 3x, clamped ±0.5.
  Returns [0.0, 0.0, 0.0] on the first check-in (no previous state).

  The environmental amplifier (1 + ΔT/10 on >2°C, up to 2x on >30% lux change)
  was removed 2026-08-14 in BOTH mappers: post-#173 warmth IS thermal state, so
  temperature entered emotional drift as the signal and again as its own
  amplifier — quadratic in one quantity — and the lux term read raw light
  including Lumen's own LED glow, so the creature's activity transitions
  amplified the drift about themselves. The deltas carry the environment once.
  """
  def compute_ethical_drift(_anima, nil, _readings, _prev_readings), do: [0.0, 0.0, 0.0]

  # readings args unused since the amplifier removal; underscore-prefixed to
  # keep the 4-arity call sites (client.ex, tests) unchanged.
  def compute_ethical_drift(anima, prev_anima, _readings, _prev_readings) do
    d_warmth = Map.get(anima, "warmth", 0.5) - Map.get(prev_anima, "warmth", 0.5)
    d_clarity = Map.get(anima, "clarity", 0.5) - Map.get(prev_anima, "clarity", 0.5)
    d_stability = Map.get(anima, "stability", 0.5) - Map.get(prev_anima, "stability", 0.5)

    scale = 3.0

    [d_warmth * scale, d_clarity * scale, d_stability * scale]
    |> Enum.map(&clamp(&1, -0.5, 0.5))
  end

  @doc "Human-readable status text for the check-in."
  def status_text(anima, eisv) do
    "Elixir broker shadow check-in. " <>
      "Warmth: #{fmt(Map.get(anima, "warmth"))}. Clarity: #{fmt(Map.get(anima, "clarity"))}. " <>
      "Stability: #{fmt(Map.get(anima, "stability"))}. Presence: #{fmt(Map.get(anima, "presence"))}. " <>
      "EISV: E=#{fmt(eisv["E"])}, I=#{fmt(eisv["I"])}, S=#{fmt(eisv["S"])}, V=#{fmt(eisv["V"])}."
  end

  defp clamp(x, lo, hi), do: max(lo, min(hi, x))

  defp fmt(nil), do: "?"
  defp fmt(x) when is_number(x), do: :erlang.float_to_binary(x / 1, decimals: 2)
end
