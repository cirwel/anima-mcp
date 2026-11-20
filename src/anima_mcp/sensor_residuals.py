"""Shadow-only per-sensor self-relative residual tracker (#79, #175 item 2).

Observes Lumen's own raw/derived sensor channels alongside the existing
fixed-cut ``sense_self()`` pipeline without changing a single behavior, cut,
projection, or governance check-in. The recorded baselines, occupancy counts,
``|z|`` histograms, and online correlations provide the empirical evidence
needed before considering any future cutover from absolute sensor ranges.

Design invariants (CLAUDE.md):
  * Shadow observation only: zero coupling into ``anima.py``,
    ``eisv_mapper.py``, or UNITARES governance check-ins.
  * Fail toward unknown: ``None`` / non-finite readings, warm-up ticks before
    ``min_samples``, and stuck physical I2C readings yield ``z = None`` and an
    explicit non-warm status rather than masquerading as healthy.
  * Pre-update scoring: every ``z`` is scored against the prior ``(mean, var)``
    before folding the current sample into the baseline.
  * Broker is the sole writer of ``~/.anima/sensor_residuals.json``; the MCP
    server reads SHM (and optionally reads the persisted JSON read-only).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .atomic_write import atomic_json_write

SCHEMA_VERSION = 1
COMPARISON_VERSION = 1
AUTHORITY = "shadow_observation_only"
MOVES_NO_GATE = True

DEFAULT_PERSISTENCE_PATH = Path.home() / ".anima" / "sensor_residuals.json"
DEFAULT_MIN_SAMPLES = 43200      # ~24h diurnal cycle at 2s broker ticks
DEFAULT_WINDOW_SAMPLES = 302400  # ~7d exponential memory horizon at 2s ticks
DEFAULT_FROZEN_RUN_THRESHOLD = 60  # ~2m of bit-identical physical readings
DEFAULT_SAVE_INTERVAL_S = 60.0
STABILITY_MIN_CONTRIBUTORS = 3

HISTOGRAM_BINS = ("lt_1", "1_to_2", "2_to_3", "ge_3")
CORRELATION_KEYS = (
    "warmth_signed_z_vs_current_warmth",
    "stability_mean_abs_z_vs_current_stability",
    "clarity_light_z_vs_current_clarity",
)
STABILITY_CHANNELS = (
    "humidity_pct",
    "pressure_hpa",
    "memory_percent",
    "eeg_delta_power",
    "eeg_theta_power",
)
STABILITY_OVERLAP_NOTE = (
    "eeg_delta_power partially derives from cpu_percent and cpu_temp_c"
)


@dataclass(frozen=True)
class ChannelSpec:
    """Metadata and noise-floor policy for a tracked sensor channel."""

    name: str
    transform: str
    min_sigma: float
    physical_frozen_guard: bool
    composite: Optional[str] = None
    non_stationary: bool = False
    conditional: Optional[str] = None


CHANNEL_SPECS: Tuple[ChannelSpec, ...] = (
    ChannelSpec(
        name="ambient_temp_c",
        transform="identity",
        min_sigma=0.25,
        physical_frozen_guard=True,
        composite="warmth",
    ),
    ChannelSpec(
        name="cpu_temp_c",
        transform="identity",
        min_sigma=0.50,
        physical_frozen_guard=True,
        composite=None,
    ),
    ChannelSpec(
        name="cpu_temp_above_ambient_c",
        transform="derived_diff",
        min_sigma=0.50,
        physical_frozen_guard=True,
        composite="warmth",
    ),
    ChannelSpec(
        name="humidity_pct",
        transform="identity",
        min_sigma=1.0,
        physical_frozen_guard=True,
        composite="stability",
    ),
    ChannelSpec(
        name="pressure_hpa",
        transform="identity",
        min_sigma=0.5,
        physical_frozen_guard=True,
        composite="stability",
    ),
    ChannelSpec(
        name="external_light_lux",
        transform="log1p",
        min_sigma=0.15,
        physical_frozen_guard=True,
        composite="clarity_light",
        conditional="daytime_attributable_only",
    ),
    ChannelSpec(
        name="memory_percent",
        transform="identity",
        min_sigma=1.0,
        physical_frozen_guard=False,
        composite="stability",
    ),
    ChannelSpec(
        name="cpu_percent",
        transform="identity",
        min_sigma=2.0,
        physical_frozen_guard=False,
        composite=None,
    ),
    ChannelSpec(
        name="disk_percent",
        transform="identity",
        min_sigma=0.5,
        physical_frozen_guard=False,
        composite=None,
        non_stationary=True,
    ),
    ChannelSpec(
        name="eeg_delta_power",
        transform="identity",
        min_sigma=0.03,
        physical_frozen_guard=False,
        composite="stability",
    ),
    ChannelSpec(
        name="eeg_theta_power",
        transform="identity",
        min_sigma=0.03,
        physical_frozen_guard=False,
        composite="stability",
    ),
)

CHANNEL_SPEC_MAP: Dict[str, ChannelSpec] = {
    spec.name: spec for spec in CHANNEL_SPECS
}


def _finite_float(value: Any) -> Optional[float]:
    """Convert value to a finite float, rejecting None, bool, and NaN/Inf."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _empty_status_counts() -> Dict[str, int]:
    return {"warm": 0, "warming": 0, "missing": 0, "frozen": 0}


def _empty_histogram() -> Dict[str, int]:
    return {bucket: 0 for bucket in HISTOGRAM_BINS}


def _histogram_bucket(abs_z: float) -> str:
    if abs_z < 1.0:
        return "lt_1"
    if abs_z < 2.0:
        return "1_to_2"
    if abs_z < 3.0:
        return "2_to_3"
    return "ge_3"


@dataclass
class OnlineCorrelation:
    """Numerically stable Welford online co-moment accumulator for Pearson r."""

    n: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    c_xy: float = 0.0

    def update(self, x: Any, y: Any) -> None:
        fx = _finite_float(x)
        fy = _finite_float(y)
        if fx is None or fy is None:
            return
        self.n += 1
        dx = fx - self.mean_x
        dy = fy - self.mean_y
        self.mean_x += dx / self.n
        self.mean_y += dy / self.n
        dx2 = fx - self.mean_x
        dy2 = fy - self.mean_y
        self.m2_x += dx * dx2
        self.m2_y += dy * dy2
        self.c_xy += dx * dy2

    @property
    def pearson_r(self) -> Optional[float]:
        if self.n < 2 or self.m2_x <= 1e-12 or self.m2_y <= 1e-12:
            return None
        denom = math.sqrt(self.m2_x * self.m2_y)
        if denom <= 1e-12 or not math.isfinite(denom):
            return None
        r = self.c_xy / denom
        return max(-1.0, min(1.0, r))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "m2_x": self.m2_x,
            "m2_y": self.m2_y,
            "c_xy": self.c_xy,
        }

    def summary(self) -> Dict[str, Any]:
        r = self.pearson_r
        return {
            "n": self.n,
            "pearson_r": round(r, 4) if r is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "OnlineCorrelation":
        if not isinstance(data, Mapping):
            return cls()
        try:
            n = int(data["n"])
            mean_x = float(data["mean_x"])
            mean_y = float(data["mean_y"])
            m2_x = float(data["m2_x"])
            m2_y = float(data["m2_y"])
            c_xy = float(data["c_xy"])
        except (KeyError, TypeError, ValueError):
            return cls()
        if (
            n < 0
            or m2_x < 0.0
            or m2_y < 0.0
            or not all(map(math.isfinite, (mean_x, mean_y, m2_x, m2_y, c_xy)))
        ):
            return cls()
        return cls(n=n, mean_x=mean_x, mean_y=mean_y, m2_x=m2_x, m2_y=m2_y, c_xy=c_xy)


@dataclass
class ResidualBand:
    """Per-channel Welford-then-EW baseline with deferred O(1) frozen-run hold."""

    spec: ChannelSpec
    min_samples: int = DEFAULT_MIN_SAMPLES
    window_samples: int = DEFAULT_WINDOW_SAMPLES
    frozen_run_threshold: int = DEFAULT_FROZEN_RUN_THRESHOLD

    n: int = 0
    mean: float = 0.0
    var: float = 0.0

    observed_ticks: int = 0
    valid_ticks: int = 0
    last_valid_at: Optional[float] = None
    floored_ticks: int = 0
    status_counts: Dict[str, int] = field(default_factory=_empty_status_counts)
    z_abs_histogram: Dict[str, int] = field(default_factory=_empty_histogram)

    # Deferred identical-value run hold for physical I2C/ADC channels.
    # Repeated identical readings are held in O(1) until the run breaks (< threshold)
    # or hits frozen_run_threshold, so the first 59 stuck ticks never shrink sigma.
    run_value: Optional[float] = None
    run_count: int = 0
    held_repeats: int = 0
    run_frozen: bool = False
    frozen_by_parent: bool = False

    def __post_init__(self) -> None:
        self.min_samples = max(1, int(self.min_samples))
        self.window_samples = max(self.min_samples, int(self.window_samples))
        self.frozen_run_threshold = max(2, int(self.frozen_run_threshold))

    @property
    def raw_sigma(self) -> float:
        return math.sqrt(max(0.0, self.var))

    @property
    def effective_sigma(self) -> float:
        return max(self.spec.min_sigma, self.raw_sigma)

    @property
    def sigma_floored(self) -> bool:
        return self.raw_sigma < self.spec.min_sigma

    @property
    def valid_fraction(self) -> float:
        if self.observed_ticks <= 0:
            return 0.0
        return self.valid_ticks / self.observed_ticks

    def _fold_one(self, x: float) -> None:
        self.n += 1
        n_eff = min(self.n, self.window_samples)
        delta = x - self.mean
        self.mean += delta / n_eff
        # Exact population Welford variance while n <= window_samples, seamlessly
        # becoming exponential forgetting with alpha = 1 / window_samples afterward.
        self.var = max(0.0, self.var + (delta * (x - self.mean) - self.var) / n_eff)

    def _flush_held(self) -> None:
        if (
            not self.run_frozen
            and self.held_repeats > 0
            and self.run_value is not None
            and math.isfinite(self.run_value)
        ):
            for _ in range(self.held_repeats):
                self._fold_one(self.run_value)
        self.held_repeats = 0

    def _score_pre_update(self, x: float) -> Tuple[str, Optional[float], bool]:
        floored = self.sigma_floored
        if self.n < self.min_samples:
            return "warming", None, floored
        eff = self.effective_sigma
        z = (x - self.mean) / eff
        if not math.isfinite(z):
            return "warming", None, floored
        return "warm", z, floored

    def _record_scored_tick(
        self,
        status: str,
        z: Optional[float],
        floored: bool,
        now: float,
    ) -> Dict[str, Any]:
        self.valid_ticks += 1
        self.last_valid_at = now
        self.status_counts[status] += 1
        if status == "warm" and z is not None:
            if floored:
                self.floored_ticks += 1
            self.z_abs_histogram[_histogram_bucket(abs(z))] += 1
        return {
            "status": status,
            "z": z,
            "sigma_floored": floored,
            "samples": self.n,
            "min_samples": self.min_samples,
            "valid_fraction": self.valid_fraction,
        }

    def _record_unusable_tick(self, status: str) -> Dict[str, Any]:
        self.status_counts[status] += 1
        return {
            "status": status,
            "z": None,
            "sigma_floored": self.sigma_floored,
            "samples": self.n,
            "min_samples": self.min_samples,
            "valid_fraction": self.valid_fraction,
        }

    def _reset_run(
        self,
        *,
        value: Optional[float] = None,
        count: int = 0,
        held: int = 0,
        frozen: bool = False,
        frozen_by_parent: bool = False,
    ) -> None:
        self.run_value = value
        self.run_count = count
        self.held_repeats = held
        self.run_frozen = frozen
        self.frozen_by_parent = frozen_by_parent

    def observe(
        self,
        value: Any,
        *,
        now: float,
        force_missing: bool = False,
        force_frozen: bool = False,
    ) -> Dict[str, Any]:
        """Score pre-update residual z (or return non-warm status) and update baseline."""
        self.observed_ticks += 1
        x = None if force_missing else _finite_float(value)

        if not self.spec.physical_frozen_guard:
            if force_frozen:
                self._reset_run(frozen=True, frozen_by_parent=True)
                return self._record_unusable_tick("frozen")
            if x is None:
                return self._record_unusable_tick("missing")
            status, z, floored = self._score_pre_update(x)
            self._fold_one(x)
            return self._record_scored_tick(status, z, floored, now)

        # Physical channel with frozen-run guard
        if force_frozen:
            self._reset_run(frozen=True, frozen_by_parent=True)
            return self._record_unusable_tick("frozen")

        if self.frozen_by_parent:
            # Parent recovered; resume clean run without polluting baseline
            self._reset_run()

        if x is None:
            self._flush_held()
            self._reset_run()
            return self._record_unusable_tick("missing")

        if self.run_value is not None and x == self.run_value:
            self.run_count += 1
            if self.run_frozen or self.run_count >= self.frozen_run_threshold:
                self.run_frozen = True
                self.held_repeats = 0
                return self._record_unusable_tick("frozen")
            status, z, floored = self._score_pre_update(x)
            self.held_repeats += 1
            return self._record_scored_tick(status, z, floored, now)

        # New value: flush any genuine short identical run before scoring x
        self._flush_held()
        status, z, floored = self._score_pre_update(x)
        self._fold_one(x)
        self._reset_run(value=x, count=1, held=0, frozen=False)
        return self._record_scored_tick(status, z, floored, now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "var": self.var,
            "observed_ticks": self.observed_ticks,
            "valid_ticks": self.valid_ticks,
            "last_valid_at": self.last_valid_at,
            "floored_ticks": self.floored_ticks,
            "status_counts": dict(self.status_counts),
            "z_abs_histogram": dict(self.z_abs_histogram),
            "run_value": self.run_value,
            "run_count": self.run_count,
            "held_repeats": self.held_repeats,
            "run_frozen": self.run_frozen,
            "frozen_by_parent": self.frozen_by_parent,
        }

    def load_comparison_fields(self, data: Mapping[str, Any]) -> None:
        """Restore occupancy/histogram stats safely, keeping zeros on bad fields."""
        try:
            obs = int(data.get("observed_ticks", 0))
            val = int(data.get("valid_ticks", 0))
            floored = int(data.get("floored_ticks", 0))
            if obs >= 0 and val >= 0 and floored >= 0:
                self.observed_ticks = obs
                self.valid_ticks = val
                self.floored_ticks = floored
        except (TypeError, ValueError):
            pass

        last_valid = _finite_float(data.get("last_valid_at"))
        if last_valid is not None and last_valid >= 0.0:
            self.last_valid_at = last_valid

        sc = data.get("status_counts")
        if isinstance(sc, Mapping):
            restored_sc = _empty_status_counts()
            for key in restored_sc:
                try:
                    c = int(sc.get(key, 0))
                except (TypeError, ValueError):
                    c = 0
                restored_sc[key] = max(0, c)
            self.status_counts = restored_sc

        hist = data.get("z_abs_histogram")
        if isinstance(hist, Mapping):
            restored_hist = _empty_histogram()
            for key in HISTOGRAM_BINS:
                try:
                    c = int(hist.get(key, 0))
                except (TypeError, ValueError):
                    c = 0
                restored_hist[key] = max(0, c)
            self.z_abs_histogram = restored_hist

    @classmethod
    def from_dict(
        cls,
        spec: ChannelSpec,
        data: Any,
        *,
        min_samples: int,
        window_samples: int,
        frozen_run_threshold: int,
        restore_comparison: bool = True,
    ) -> "ResidualBand":
        band = cls(
            spec=spec,
            min_samples=min_samples,
            window_samples=window_samples,
            frozen_run_threshold=frozen_run_threshold,
        )
        if not isinstance(data, Mapping):
            return band
        try:
            n = int(data["n"])
            mean = float(data["mean"])
            var = float(data["var"])
        except (KeyError, TypeError, ValueError):
            return band
        if n < 0 or var < 0.0 or not all(map(math.isfinite, (mean, var))):
            return band

        band.n = n
        band.mean = mean
        band.var = var

        rv = _finite_float(data.get("run_value"))
        try:
            rc = max(0, int(data.get("run_count", 0)))
            hr = max(0, int(data.get("held_repeats", 0)))
        except (TypeError, ValueError):
            rv, rc, hr = None, 0, 0
        band.run_value = rv
        band.run_count = rc
        band.held_repeats = min(hr, band.frozen_run_threshold)
        band.run_frozen = bool(data.get("run_frozen", False))
        band.frozen_by_parent = bool(data.get("frozen_by_parent", False))

        if restore_comparison:
            band.load_comparison_fields(data)
        return band

    def detail_summary(self, now: Optional[float] = None) -> Dict[str, Any]:
        ts = time.time() if now is None else now
        last_age = (
            round(max(0.0, ts - self.last_valid_at), 1)
            if self.last_valid_at is not None
            else None
        )
        raw_sig = self.raw_sigma
        eff_sig = self.effective_sigma
        return {
            "samples": self.n,
            "min_samples": self.min_samples,
            "window_samples": self.window_samples,
            "mean": round(self.mean, 6),
            "raw_sigma": round(raw_sig, 6),
            "effective_sigma": round(eff_sig, 6),
            "min_sigma": self.spec.min_sigma,
            "sigma_floored": self.sigma_floored,
            "transform": self.spec.transform,
            "physical_frozen_guard": self.spec.physical_frozen_guard,
            "non_stationary": self.spec.non_stationary,
            "conditional": self.spec.conditional,
            "observed_ticks": self.observed_ticks,
            "valid_ticks": self.valid_ticks,
            "valid_fraction": round(self.valid_fraction, 4),
            "last_valid_age_s": last_age,
            "floored_ticks": self.floored_ticks,
            "status_counts": dict(self.status_counts),
            "z_abs_histogram": dict(self.z_abs_histogram),
            "frozen": self.run_frozen,
        }


def _reading_attr(readings: Any, name: str) -> Any:
    if isinstance(readings, Mapping):
        return readings.get(name)
    return getattr(readings, name, None)


def _quarantine_corrupt_file(path: Path) -> None:
    """Move a corrupt or schema-incompatible persistence file aside safely."""
    try:
        corrupt_path = path.with_name(f"{path.name}.corrupt")
        path.replace(corrupt_path)
    except OSError:
        pass


class SensorResidualTracker:
    """Broker-owned shadow tracker for per-sensor self-relative baselines."""

    def __init__(
        self,
        persistence_path: Optional[Path | str] = None,
        *,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        frozen_run_threshold: int = DEFAULT_FROZEN_RUN_THRESHOLD,
        save_interval_s: float = DEFAULT_SAVE_INTERVAL_S,
        auto_load: bool = True,
    ) -> None:
        self.persistence_path = Path(
            DEFAULT_PERSISTENCE_PATH if persistence_path is None else persistence_path
        )
        self.min_samples = max(1, int(min_samples))
        self.window_samples = max(self.min_samples, int(window_samples))
        self.frozen_run_threshold = max(2, int(frozen_run_threshold))
        self.save_interval_s = max(0.0, float(save_interval_s))

        self.bands: Dict[str, ResidualBand] = {
            spec.name: ResidualBand(
                spec=spec,
                min_samples=self.min_samples,
                window_samples=self.window_samples,
                frozen_run_threshold=self.frozen_run_threshold,
            )
            for spec in CHANNEL_SPECS
        }
        self.correlations: Dict[str, OnlineCorrelation] = {
            key: OnlineCorrelation() for key in CORRELATION_KEYS
        }
        self.observed_ticks: int = 0
        self._last_results: Dict[str, Dict[str, Any]] = {}
        self._last_composites: Dict[str, Any] = self._build_composites({}, now=time.time())
        self._last_save_at: float = time.time()

        if auto_load:
            self._load()

    def _reset_fresh(self) -> None:
        for spec in CHANNEL_SPECS:
            self.bands[spec.name] = ResidualBand(
                spec=spec,
                min_samples=self.min_samples,
                window_samples=self.window_samples,
                frozen_run_threshold=self.frozen_run_threshold,
            )
        self.correlations = {key: OnlineCorrelation() for key in CORRELATION_KEYS}
        self.observed_ticks = 0

    def _load(self) -> None:
        if not self.persistence_path.exists():
            return
        try:
            raw = self.persistence_path.read_text()
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(
                f"[SensorResiduals] Corrupt persistence file ({e}); cold-starting",
                file=sys.stderr,
                flush=True,
            )
            _quarantine_corrupt_file(self.persistence_path)
            self._reset_fresh()
            return

        if not isinstance(data, Mapping) or data.get("schema_version") != SCHEMA_VERSION:
            print(
                "[SensorResiduals] Schema mismatch or invalid root; cold-starting",
                file=sys.stderr,
                flush=True,
            )
            _quarantine_corrupt_file(self.persistence_path)
            self._reset_fresh()
            return

        restore_comparison = data.get("comparison_version") == COMPARISON_VERSION
        channels_data = data.get("channels")
        if isinstance(channels_data, Mapping):
            for spec in CHANNEL_SPECS:
                self.bands[spec.name] = ResidualBand.from_dict(
                    spec,
                    channels_data.get(spec.name),
                    min_samples=self.min_samples,
                    window_samples=self.window_samples,
                    frozen_run_threshold=self.frozen_run_threshold,
                    restore_comparison=restore_comparison,
                )

        if restore_comparison:
            try:
                ticks = int(data.get("observed_ticks", 0))
                self.observed_ticks = max(0, ticks)
            except (TypeError, ValueError):
                self.observed_ticks = 0
            corr_data = data.get("correlations")
            if isinstance(corr_data, Mapping):
                for key in CORRELATION_KEYS:
                    self.correlations[key] = OnlineCorrelation.from_dict(
                        corr_data.get(key)
                    )
        else:
            self.observed_ticks = max(
                (b.observed_ticks for b in self.bands.values()), default=0
            )
            self.correlations = {key: OnlineCorrelation() for key in CORRELATION_KEYS}

    def observe(
        self,
        readings: Any,
        *,
        external_light_lux: Any = None,
        current_warmth: Any = None,
        current_stability: Any = None,
        current_clarity: Any = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Observe one broker tick in shadow-only mode and return compact SHM snapshot."""
        ts = time.time() if now is None else float(now)
        self.observed_ticks += 1
        results: Dict[str, Dict[str, Any]] = {}

        # 1. Parent thermal channels + derived CPU excess above ambient (#79)
        amb_raw = _reading_attr(readings, "ambient_temp_c")
        cpu_temp_raw = _reading_attr(readings, "cpu_temp_c")
        amb_res = self.bands["ambient_temp_c"].observe(amb_raw, now=ts)
        cpu_temp_res = self.bands["cpu_temp_c"].observe(cpu_temp_raw, now=ts)
        results["ambient_temp_c"] = amb_res
        results["cpu_temp_c"] = cpu_temp_res

        parent_frozen = (
            amb_res["status"] == "frozen" or cpu_temp_res["status"] == "frozen"
        )
        parent_missing = (
            amb_res["status"] == "missing" or cpu_temp_res["status"] == "missing"
        )
        if parent_frozen or parent_missing:
            diff_val = None
        else:
            diff_val = float(cpu_temp_raw) - float(amb_raw)
        results["cpu_temp_above_ambient_c"] = self.bands[
            "cpu_temp_above_ambient_c"
        ].observe(
            diff_val,
            now=ts,
            force_missing=parent_missing and not parent_frozen,
            force_frozen=parent_frozen,
        )

        # 2. Gated external light ONLY (never raw readings.light_lux)
        gated_lux = _finite_float(external_light_lux)
        log_lux = math.log1p(max(0.0, gated_lux)) if gated_lux is not None else None
        results["external_light_lux"] = self.bands["external_light_lux"].observe(
            log_lux, now=ts
        )

        # 3. Remaining direct physical + system + computational channels
        for ch_name in (
            "humidity_pct",
            "pressure_hpa",
            "memory_percent",
            "cpu_percent",
            "disk_percent",
            "eeg_delta_power",
            "eeg_theta_power",
        ):
            results[ch_name] = self.bands[ch_name].observe(
                _reading_attr(readings, ch_name), now=ts
            )

        composites = self._build_composites(results, now=ts)

        # 4. Update comparison Pearson correlations while composites are warm
        warmth_comp = composites["warmth"]
        if warmth_comp["status"] == "warm":
            self.correlations["warmth_signed_z_vs_current_warmth"].update(
                warmth_comp["signed_z"], current_warmth
            )

        stab_comp = composites["stability"]
        if stab_comp["status"] == "warm":
            self.correlations["stability_mean_abs_z_vs_current_stability"].update(
                stab_comp["mean_abs_z"], current_stability
            )

        light_comp = composites["clarity_light"]
        if light_comp["status"] == "warm":
            self.correlations["clarity_light_z_vs_current_clarity"].update(
                light_comp["z"], current_clarity
            )

        self._last_results = results
        self._last_composites = composites
        return self.shm_snapshot(now=ts)

    def _build_composites(
        self,
        results: Mapping[str, Mapping[str, Any]],
        *,
        now: float,
    ) -> Dict[str, Any]:
        # Warmth composite: de-aliases CPU heat from ambient warmth (#79)
        amb_res = results.get("ambient_temp_c")
        diff_res = results.get("cpu_temp_above_ambient_c")
        amb_band = self.bands["ambient_temp_c"]
        diff_band = self.bands["cpu_temp_above_ambient_c"]

        if (
            amb_res is not None
            and diff_res is not None
            and amb_res.get("status") == "warm"
            and diff_res.get("status") == "warm"
            and amb_res.get("z") is not None
            and diff_res.get("z") is not None
        ):
            z_amb = float(amb_res["z"])
            z_diff = float(diff_res["z"])
            warmth_comp = {
                "status": "warm",
                "signed_z": round((z_amb + z_diff) / 2.0, 4),
                "ambient_temp_z": round(z_amb, 4),
                "cpu_above_ambient_z": round(z_diff, 4),
                "components": ["ambient_temp_c", "cpu_temp_above_ambient_c"],
                "note": "de-aliases cpu_temp_c from ambient_temp_c (#79)",
            }
        else:
            warmth_status = _composite_non_warm_status(
                bands=(amb_band, diff_band),
                statuses=(
                    amb_res.get("status") if amb_res else "warming",
                    diff_res.get("status") if diff_res else "warming",
                ),
            )
            warmth_comp = {
                "status": warmth_status,
                "signed_z": None,
                "ambient_temp_z": None,
                "cpu_above_ambient_z": None,
                "components": ["ambient_temp_c", "cpu_temp_above_ambient_c"],
                "note": "de-aliases cpu_temp_c from ambient_temp_c (#79)",
            }

        # Stability composite: multi-channel deviation magnitude, disk excluded
        contributors = []
        abs_zs = []
        for ch_name in STABILITY_CHANNELS:
            ch_res = results.get(ch_name)
            if (
                ch_res is not None
                and ch_res.get("status") == "warm"
                and ch_res.get("z") is not None
            ):
                contributors.append(ch_name)
                abs_zs.append(abs(float(ch_res["z"])))

        if len(contributors) >= STABILITY_MIN_CONTRIBUTORS:
            stability_comp = {
                "status": "warm",
                "mean_abs_z": round(sum(abs_zs) / len(abs_zs), 4),
                "max_abs_z": round(max(abs_zs), 4),
                "contributors": contributors,
                "contributor_count": len(contributors),
                "expected_contributors": len(STABILITY_CHANNELS),
                "min_contributors": STABILITY_MIN_CONTRIBUTORS,
                "excluded_channels": ["disk_percent"],
                "overlap_note": STABILITY_OVERLAP_NOTE,
            }
        else:
            stab_bands = tuple(self.bands[c] for c in STABILITY_CHANNELS)
            any_warming = any(b.n < b.min_samples for b in stab_bands)
            stability_comp = {
                "status": "warming" if any_warming else "insufficient",
                "mean_abs_z": None,
                "max_abs_z": None,
                "contributors": contributors,
                "contributor_count": len(contributors),
                "expected_contributors": len(STABILITY_CHANNELS),
                "min_contributors": STABILITY_MIN_CONTRIBUTORS,
                "excluded_channels": ["disk_percent"],
                "overlap_note": STABILITY_OVERLAP_NOTE,
            }

        # Clarity light component (daytime/attributable only)
        light_res = results.get("external_light_lux")
        light_band = self.bands["external_light_lux"]
        light_z = (
            round(float(light_res["z"]), 4)
            if light_res is not None and light_res.get("status") == "warm" and light_res.get("z") is not None
            else None
        )
        light_age = (
            round(max(0.0, now - light_band.last_valid_at), 1)
            if light_band.last_valid_at is not None
            else None
        )
        clarity_light_comp = {
            "status": light_res.get("status") if light_res else (
                "warming" if light_band.n < light_band.min_samples else "missing"
            ),
            "z": light_z,
            "valid_fraction": round(light_band.valid_fraction, 4),
            "last_valid_age_s": light_age,
            "conditional": "daytime_attributable_only",
        }

        return {
            "warmth": warmth_comp,
            "stability": stability_comp,
            "clarity_light": clarity_light_comp,
        }

    def shm_snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Return compact JSON-serializable snapshot for SHM publication."""
        ts = time.time() if now is None else float(now)
        channels: Dict[str, Any] = {}
        for spec in CHANNEL_SPECS:
            name = spec.name
            band = self.bands[name]
            res = self._last_results.get(name)
            if res is not None:
                z_val = round(float(res["z"]), 4) if res.get("z") is not None else None
                status = res["status"]
            else:
                z_val = None
                status = "warming" if band.n < band.min_samples else "missing"
            entry: Dict[str, Any] = {
                "status": status,
                "z": z_val,
                "sigma_floored": band.sigma_floored,
                "samples": band.n,
                "min_samples": band.min_samples,
                "valid_fraction": round(band.valid_fraction, 4),
            }
            if spec.non_stationary:
                entry["non_stationary"] = True
            if spec.conditional is not None:
                entry["conditional"] = spec.conditional
            channels[name] = entry

        composites = (
            self._last_composites
            if self._last_results
            else self._build_composites({}, now=ts)
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "authority": AUTHORITY,
            "moves_no_gate": MOVES_NO_GATE,
            "observed_ticks": self.observed_ticks,
            "channels": channels,
            "composites": composites,
        }

    def maybe_save(self, now: Optional[float] = None) -> bool:
        ts = time.time() if now is None else float(now)
        if ts < self._last_save_at:
            # Wall clock stepped backwards (e.g. NTP sync): re-anchor save timer.
            self._last_save_at = ts
            return False
        if ts - self._last_save_at >= self.save_interval_s:
            saved = self.save(now=ts)
            if saved:
                self._last_save_at = ts
            return saved
        return False

    def save(self, now: Optional[float] = None) -> bool:
        """Persist baselines and comparison accumulators atomically."""
        ts = time.time() if now is None else float(now)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "comparison_version": COMPARISON_VERSION,
            "authority": AUTHORITY,
            "moves_no_gate": MOVES_NO_GATE,
            "saved_at": ts,
            "observed_ticks": self.observed_ticks,
            "channels": {
                spec.name: self.bands[spec.name].to_dict()
                for spec in CHANNEL_SPECS
            },
            "correlations": {
                key: self.correlations[key].to_dict()
                for key in CORRELATION_KEYS
            },
            "composites": (
                self._last_composites
                if self._last_results
                else self._build_composites({}, now=ts)
            ),
        }
        try:
            atomic_json_write(self.persistence_path, payload)
            return True
        except Exception as e:
            print(
                f"[SensorResiduals] Save error: {e}",
                file=sys.stderr,
                flush=True,
            )
            return False


def _composite_non_warm_status(
    *,
    bands: Iterable[ResidualBand],
    statuses: Iterable[Optional[str]],
) -> str:
    if any(b.n < b.min_samples for b in bands):
        return "warming"
    status_list = list(statuses)
    if "frozen" in status_list:
        return "frozen"
    if "missing" in status_list:
        return "missing"
    return "warming"


def load_persisted_sensor_residuals(
    path: Optional[Path | str] = None,
    *,
    now: Optional[float] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    frozen_run_threshold: int = DEFAULT_FROZEN_RUN_THRESHOLD,
) -> Dict[str, Any]:
    """Pure read-only loader for full diagnostics inspection (writes nothing)."""
    target = Path(DEFAULT_PERSISTENCE_PATH if path is None else path)
    if not target.exists():
        return {
            "available": False,
            "authority": AUTHORITY,
            "moves_no_gate": MOVES_NO_GATE,
            "reason": "file_not_found",
        }
    try:
        data = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return {
            "available": False,
            "authority": AUTHORITY,
            "moves_no_gate": MOVES_NO_GATE,
            "reason": f"read_error: {type(e).__name__}",
        }

    if not isinstance(data, Mapping) or data.get("schema_version") != SCHEMA_VERSION:
        return {
            "available": False,
            "authority": AUTHORITY,
            "moves_no_gate": MOVES_NO_GATE,
            "reason": "schema_mismatch",
        }

    ts = time.time() if now is None else float(now)
    saved_at = _finite_float(data.get("saved_at"))
    snapshot_age_s = (
        round(max(0.0, ts - saved_at), 1) if saved_at is not None else None
    )
    restore_comparison = data.get("comparison_version") == COMPARISON_VERSION

    channels_raw = data.get("channels")
    channels_out: Dict[str, Any] = {}
    for spec in CHANNEL_SPECS:
        ch_raw = channels_raw.get(spec.name) if isinstance(channels_raw, Mapping) else None
        band = ResidualBand.from_dict(
            spec,
            ch_raw,
            min_samples=min_samples,
            window_samples=window_samples,
            frozen_run_threshold=frozen_run_threshold,
            restore_comparison=restore_comparison,
        )
        channels_out[spec.name] = band.detail_summary(now=ts)

    corr_raw = data.get("correlations")
    correlations_out: Dict[str, Any] = {}
    for key in CORRELATION_KEYS:
        entry = (
            corr_raw.get(key)
            if restore_comparison and isinstance(corr_raw, Mapping)
            else None
        )
        correlations_out[key] = OnlineCorrelation.from_dict(entry).summary()

    try:
        observed_ticks = max(0, int(data.get("observed_ticks", 0)))
    except (TypeError, ValueError):
        observed_ticks = 0

    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "comparison_version": data.get("comparison_version"),
        "authority": AUTHORITY,
        "moves_no_gate": MOVES_NO_GATE,
        "saved_at": saved_at,
        "snapshot_age_s": snapshot_age_s,
        "observed_ticks": observed_ticks,
        "channels": channels_out,
        "composites": data.get("composites") if isinstance(data.get("composites"), Mapping) else {},
        "correlations": correlations_out,
    }
