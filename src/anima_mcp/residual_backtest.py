"""Offline historical SensorResidualTracker replay and Stage-2 readiness evaluator (#79, #175).

Reads Lumen's durable ``state_history`` SQLite table strictly read-only
(``?mode=ro`` by default so WAL-backed DBs include ``-wal`` frames cleanly;
opt-in ``immutable=True`` / ``--immutable`` for frozen backup copies) or an
existing persisted ``sensor_residuals.json`` snapshot, replays historical
observations through the real
:class:`anima_mcp.sensor_residuals.SensorResidualTracker`, and emits a
structured Stage-2 readiness report (JSON + Markdown).

Design invariants:
  * Shadow/offline analysis only: zero coupling into ``anima.py``,
    ``eisv_mapper.py``, or ``anima_broker/lib/anima_broker/governance/eisv_mapper.ex``.
  * Strictly read-only: never modifies SQLite databases or persisted snapshots,
    and never invokes ``SensorResidualTracker.save()``.
  * Fail toward unknown (Invariant 2): NULL, missing, boolean, string, or
    non-finite numeric fields reconstruct as ``None`` rather than fabricated
    defaults, and unevaluated distributions/gates report ``None`` / ``unknown``
    rather than passing.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

from .anima import Anima
from .sensor_residuals import (
    AUTHORITY,
    CHANNEL_SPECS,
    COMPARISON_VERSION,
    CORRELATION_KEYS,
    DEFAULT_FROZEN_RUN_THRESHOLD,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_WINDOW_SAMPLES,
    HISTOGRAM_BINS,
    MOVES_NO_GATE,
    SCHEMA_VERSION as RESIDUALS_SCHEMA_VERSION,
    STABILITY_CHANNELS,
    STABILITY_MIN_CONTRIBUTORS,
    OnlineCorrelation,
    ResidualBand,
    SensorResidualTracker,
    _finite_float,
)
from .sensors.base import SensorReadings

BACKTEST_SCHEMA_VERSION = 1
BROKER_TICK_SECONDS = 2.0
SEVEN_DAYS_SECONDS = 7 * 86400
DEFAULT_DB_PATH = Path.home() / ".anima" / "anima.db"

SIGMA_FLOOR_WATCH_CHANNELS: Tuple[str, ...] = (
    "memory_percent",
    "eeg_delta_power",
    "eeg_theta_power",
)
WARMTH_CHANNELS: Tuple[str, ...] = (
    "ambient_temp_c",
    "cpu_temp_above_ambient_c",
)
COMPOSITE_ORDER: Tuple[str, ...] = ("warmth", "stability", "clarity_light")
COMPOSITE_METRIC_NAMES: Dict[str, str] = {
    "warmth": "signed_z",
    "stability": "mean_abs_z",
    "clarity_light": "z",
}
COMPARISON_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "warmth",
        "warmth_signed_z_vs_current_warmth",
        "shadow.warmth.signed_z",
        "legacy.anima.warmth",
    ),
    (
        "stability",
        "stability_mean_abs_z_vs_current_stability",
        "shadow.stability.mean_abs_z",
        "legacy.anima.stability",
    ),
    (
        "clarity_light",
        "clarity_light_z_vs_current_clarity",
        "shadow.clarity_light.z",
        "legacy.anima.clarity",
    ),
)

MIN_NON_DEGENERATE_VARIANCE = 1e-6
SIGMA_FLOOR_WARN_FRACTION = 0.50

_SENSOR_FLOAT_FIELDS: Tuple[str, ...] = (
    "cpu_temp_c",
    "ambient_temp_c",
    "humidity_pct",
    "light_lux",
    "light_observed_precision_seconds",
    "sound_level",
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "power_watts",
    "led_brightness",
    "pressure_hpa",
    "pressure_temp_c",
    "eeg_tp9",
    "eeg_af7",
    "eeg_af8",
    "eeg_tp10",
    "eeg_aux1",
    "eeg_aux2",
    "eeg_aux3",
    "eeg_aux4",
    "eeg_delta_power",
    "eeg_theta_power",
    "eeg_alpha_power",
    "eeg_beta_power",
    "eeg_gamma_power",
)
_SENSOR_BOOL_FIELDS: Tuple[str, ...] = (
    "undervoltage_now",
    "throttled_now",
    "freq_capped_now",
    "undervoltage_occurred",
)

PYTHON_ELIXIR_PARITY_ITEMS: Tuple[str, ...] = (
    "Confirm Stage-1 shadow soak (#290) accumulated >=7 warm days on live Lumen "
    "(~/.anima/sensor_residuals.json) at 2s broker ticks with moves_no_gate=true.",
    "Review sigma-floor calibration for memory_percent, eeg_delta_power, and "
    "eeg_theta_power against live 2s-tick raw_sigma before any cutover.",
    "Keep anima.py (_sense_warmth, _sense_stability, _sense_clarity) and "
    "eisv_mapper.py aligned with Invariant 1 (self-relative bands) and "
    "Invariant 2 (fail toward unknown).",
    "Mirror any governance-facing projection change identically in "
    "anima_broker/lib/anima_broker/governance/eisv_mapper.ex with cross-language "
    "parity tests.",
    "Rebuild and restart the on-Pi Elixir release (MIX_ENV=prod mix release "
    "--overwrite && sudo systemctl restart anima-broker-ex) alongside "
    "anima-broker and anima.",
)


@dataclass(frozen=True)
class StateHistoryRecord:
    """Reconstructed chronological row from ``state_history``.

    Note: ``anima`` is an :class:`Anima` instance whose scalar fields may be
    ``None`` when the corresponding ``state_history`` column was NULL or
    non-finite (preserving Invariant 2 rather than substituting defaults).
    """

    timestamp: datetime
    timestamp_unix: float
    readings: SensorReadings
    anima: Anima
    external_light_lux: Optional[float]
    light_attribution_status: Optional[str]
    valid_sensors_json: bool

    def as_tuple(self) -> Tuple[datetime, SensorReadings, Anima]:
        return (self.timestamp, self.readings, self.anima)


def _sqlite_readonly_uri(db_path: Path | str, *, immutable: bool = False) -> str:
    resolved = Path(db_path).expanduser().resolve()
    encoded = quote(resolved.as_posix(), safe="/:")
    if immutable:
        return f"file:{encoded}?mode=ro&immutable=1"
    return f"file:{encoded}?mode=ro"


def _connect_readonly(
    db_path: Path | str,
    *,
    immutable: bool = False,
) -> sqlite3.Connection:
    target = Path(db_path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"SQLite database not found: {target}")
    return sqlite3.connect(
        _sqlite_readonly_uri(target, immutable=immutable),
        uri=True,
        isolation_level=None,
    )


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _datetime_to_unix(dt: datetime) -> float:
    """Convert datetime to a normalized UTC-referenced Unix timestamp.

    Timezone-aware timestamps are converted to UTC. Naive timestamps (as written
    by Lumen's local ``datetime.now().isoformat()`` in ``state_history``) are
    interpreted on a fixed UTC reference frame so elapsed intervals are
    monotonic and unaffected by local DST transitions.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).timestamp()
    return dt.astimezone(timezone.utc).timestamp()


def _strict_numeric_float(value: Any) -> Optional[float]:
    """Convert a JSON/SQLite numeric value to finite float, rejecting bool/str/None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _extract_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _extract_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _extract_gated_external_light(
    sensors_dict: Mapping[str, Any],
) -> Tuple[Optional[float], Optional[str]]:
    """Return gated external_light_lux ONLY when attribution is 'ready_shadow'.

    Never falls back to raw ``light_lux`` and rejects missing/non-ready status
    as well as bool/str values (mirroring ``light_attribution.py``).
    """
    raw_status = sensors_dict.get("light_attribution_status")
    status = raw_status if isinstance(raw_status, str) else None
    if status != "ready_shadow":
        return None, status
    ext_lux = _strict_numeric_float(sensors_dict.get("external_light_lux"))
    return ext_lux, status


def reconstruct_state_history_row(
    timestamp_raw: Any,
    warmth_raw: Any,
    clarity_raw: Any,
    stability_raw: Any,
    presence_raw: Any,
    sensors_raw: Any,
) -> Optional[StateHistoryRecord]:
    """Reconstruct one ``state_history`` row into ``(timestamp, SensorReadings, Anima)``.

    Honors Invariant 2 (fail toward unknown): NULL, missing, boolean, string,
    non-finite, or malformed JSON fields become ``None`` rather than invented
    defaults. Note that ``Anima`` numeric fields are preserved as ``None`` when
    missing/NULL in ``state_history``.
    """
    ts_dt = _parse_iso_datetime(timestamp_raw)
    if ts_dt is None:
        return None
    ts_unix = _datetime_to_unix(ts_dt)

    sensors_dict: Dict[str, Any] = {}
    valid_json = False
    if isinstance(sensors_raw, Mapping):
        sensors_dict = dict(sensors_raw)
        valid_json = True
    elif isinstance(sensors_raw, (str, bytes, bytearray)):
        text = (
            sensors_raw.decode("utf-8", errors="replace")
            if isinstance(sensors_raw, (bytes, bytearray))
            else sensors_raw
        )
        if text.strip():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    sensors_dict = parsed
                    valid_json = True
            except (ValueError, TypeError):
                valid_json = False

    reading_kwargs: Dict[str, Any] = {"timestamp": ts_dt}
    for field_name in _SENSOR_FLOAT_FIELDS:
        reading_kwargs[field_name] = _strict_numeric_float(sensors_dict.get(field_name))
    reading_kwargs["light_observed_at"] = _parse_iso_datetime(
        sensors_dict.get("light_observed_at")
    )
    reading_kwargs["hearing_available"] = (
        sensors_dict.get("hearing_available")
        if isinstance(sensors_dict.get("hearing_available"), bool)
        else False
    )
    reading_kwargs["throttle_bits"] = _extract_int(sensors_dict.get("throttle_bits"))
    for field_name in _SENSOR_BOOL_FIELDS:
        reading_kwargs[field_name] = _extract_optional_bool(sensors_dict.get(field_name))

    readings = SensorReadings(**reading_kwargs)
    ext_lux, light_status = _extract_gated_external_light(sensors_dict)

    # Preserve None for missing/NULL legacy anima values (fail toward unknown).
    anima = Anima(
        warmth=_strict_numeric_float(warmth_raw),      # type: ignore[arg-type]
        clarity=_strict_numeric_float(clarity_raw),    # type: ignore[arg-type]
        stability=_strict_numeric_float(stability_raw),  # type: ignore[arg-type]
        presence=_strict_numeric_float(presence_raw),  # type: ignore[arg-type]
        readings=readings,
    )
    return StateHistoryRecord(
        timestamp=ts_dt,
        timestamp_unix=ts_unix,
        readings=readings,
        anima=anima,
        external_light_lux=ext_lux,
        light_attribution_status=light_status,
        valid_sensors_json=valid_json,
    )


def _has_state_history_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_history'"
    ).fetchone()
    return row is not None


def _validate_days(days: Optional[float]) -> Optional[float]:
    if days is None:
        return None
    if isinstance(days, bool):
        raise ValueError("days must be a positive finite number")
    days_f = float(days)
    if not math.isfinite(days_f) or days_f <= 0.0:
        raise ValueError("days must be a positive finite number")
    return days_f


_SQLITE_ROWID_CHUNK_SIZE = 500


def iter_state_history_records(
    db_path: Path | str,
    *,
    days: Optional[float] = None,
    immutable: bool = False,
    counters: Optional[Dict[str, int]] = None,
) -> Iterator[StateHistoryRecord]:
    """Stream chronological :class:`StateHistoryRecord` rows from SQLite read-only.

    Anchors ``--days`` and orders rows strictly by parsed, UTC-normalized
    timestamps (never SQL text ``MAX(timestamp)`` or lexical text sorting),
    skipping and counting unparseable timestamps cleanly.
    """
    days_f = _validate_days(days)
    conn = _connect_readonly(db_path, immutable=immutable)
    try:
        # Execute both passes inside one explicit read transaction so concurrent
        # WAL commits cannot skew --days anchoring or row fetches.
        conn.execute("BEGIN")
        if not _has_state_history_table(conn):
            raise ValueError("state_history table not found")

        # Pass 1: parse timestamps across table to anchor --days and sort
        # chronologically on normalized timestamps, ignoring bad timestamp rows.
        valid_index: List[Tuple[float, int]] = []
        max_unix: Optional[float] = None
        for rowid, ts_raw in conn.execute("SELECT rowid, timestamp FROM state_history"):
            ts_dt = _parse_iso_datetime(ts_raw)
            if ts_dt is None:
                if counters is not None:
                    counters["skipped_bad_timestamp_rows"] = (
                        counters.get("skipped_bad_timestamp_rows", 0) + 1
                    )
                continue
            ts_unix = _datetime_to_unix(ts_dt)
            valid_index.append((ts_unix, int(rowid)))
            if max_unix is None or ts_unix > max_unix:
                max_unix = ts_unix

        if not valid_index or max_unix is None:
            return

        if days_f is not None:
            cutoff_unix = max_unix - days_f * 86400.0
            valid_index = [
                (ts_unix, rid) for ts_unix, rid in valid_index if ts_unix >= cutoff_unix
            ]
            if not valid_index:
                return

        valid_index.sort(key=lambda item: (item[0], item[1]))

        # Pass 2: stream full rows in bounded rowid batches in exact sorted order.
        for start in range(0, len(valid_index), _SQLITE_ROWID_CHUNK_SIZE):
            batch = valid_index[start : start + _SQLITE_ROWID_CHUNK_SIZE]
            batch_rids = [rid for _, rid in batch]
            placeholders = ",".join("?" for _ in batch_rids)
            fetched: Dict[int, Tuple[Any, Any, Any, Any, Any, Any]] = {}
            for rid, ts_raw, w, c, s, p, sensors_raw in conn.execute(
                f"SELECT rowid, timestamp, warmth, clarity, stability, presence, sensors "
                f"FROM state_history WHERE rowid IN ({placeholders})",
                batch_rids,
            ):
                fetched[int(rid)] = (ts_raw, w, c, s, p, sensors_raw)

            for _, rid in batch:
                raw_row = fetched.get(rid)
                if raw_row is None:
                    continue
                record = reconstruct_state_history_row(*raw_row)
                if record is None:
                    continue
                if counters is not None:
                    if not record.valid_sensors_json:
                        counters["malformed_sensor_json_rows"] = (
                            counters.get("malformed_sensor_json_rows", 0) + 1
                        )
                    if any(
                        v is None
                        for v in (
                            record.anima.warmth,
                            record.anima.clarity,
                            record.anima.stability,
                            record.anima.presence,
                        )
                    ):
                        counters["null_anima_rows"] = (
                            counters.get("null_anima_rows", 0) + 1
                        )
                yield record
    finally:
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        conn.close()


def load_state_history_tuples(
    db_path: Path | str,
    *,
    days: Optional[float] = None,
    immutable: bool = False,
) -> List[Tuple[datetime, SensorReadings, Anima]]:
    """Read chronological ``(timestamp, SensorReadings, Anima)`` tuples read-only."""
    return [
        rec.as_tuple()
        for rec in iter_state_history_records(db_path, days=days, immutable=immutable)
    ]


def _percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, q)) * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _series_distribution(values: Sequence[float]) -> Dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "variance": None,
            "variance_raw": None,
            "sd_raw": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    ordered = sorted(values)
    mean = math.fsum(ordered) / n
    if n >= 2:
        var = math.fsum((v - mean) ** 2 for v in ordered) / n
        var = max(0.0, var)
        sd = math.sqrt(var)
    else:
        var = None
        sd = None
    return {
        "n": n,
        "mean": round(mean, 6),
        "sd": round(sd, 6) if sd is not None else None,
        "variance": round(var, 9) if var is not None else None,
        "variance_raw": var,
        "sd_raw": sd,
        "p05": round(_percentile(ordered, 0.05), 6),  # type: ignore[arg-type]
        "p50": round(_percentile(ordered, 0.50), 6),  # type: ignore[arg-type]
        "p95": round(_percentile(ordered, 0.95), 6),  # type: ignore[arg-type]
    }


def _round_opt(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _ratio_opt(numerator: int, denominator: int, digits: int = 4) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, digits)


def _histogram_fractions(hist: Mapping[str, int]) -> Dict[str, Optional[float]]:
    total = sum(int(hist.get(b, 0)) for b in HISTOGRAM_BINS)
    if total <= 0:
        return {b: None for b in HISTOGRAM_BINS}
    return {b: round(int(hist.get(b, 0)) / total, 4) for b in HISTOGRAM_BINS}


def _summarize_intervals(intervals_s: Sequence[float]) -> Dict[str, Any]:
    positives = sorted(dt for dt in intervals_s if math.isfinite(dt) and dt > 0.0)
    non_positive = sum(1 for dt in intervals_s if not math.isfinite(dt) or dt <= 0.0)
    if not positives:
        return {
            "interval_count": 0,
            "non_positive_intervals": non_positive,
            "median_seconds": None,
            "p10_seconds": None,
            "p90_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "broker_tick_seconds": BROKER_TICK_SECONDS,
            "cadence_ratio_vs_broker": None,
            "implied_24h_rows": None,
            "implied_7d_rows": None,
            "cadence_note": "Insufficient rows to measure sampling interval.",
        }
    med = _percentile(positives, 0.50)
    p10 = _percentile(positives, 0.10)
    p90 = _percentile(positives, 0.90)
    assert med is not None
    ratio = med / BROKER_TICK_SECONDS
    implied_24h = max(1, int(round(86400.0 / med)))
    implied_7d = max(1, int(round(SEVEN_DAYS_SECONDS / med)))
    note = (
        f"Median state_history interval is {med:.2f}s ({ratio:.1f}x live {BROKER_TICK_SECONDS:.0f}s "
        f"broker ticks; ~{implied_24h} rows/24h, ~{implied_7d} rows/7d vs "
        f"{DEFAULT_MIN_SAMPLES} warmup / {DEFAULT_WINDOW_SAMPLES} 7d live ticks)."
    )
    return {
        "interval_count": len(positives),
        "non_positive_intervals": non_positive,
        "median_seconds": round(med, 3),
        "p10_seconds": _round_opt(p10, 3),
        "p90_seconds": _round_opt(p90, 3),
        "min_seconds": round(positives[0], 3),
        "max_seconds": round(positives[-1], 3),
        "broker_tick_seconds": BROKER_TICK_SECONDS,
        "cadence_ratio_vs_broker": round(ratio, 2),
        "implied_24h_rows": implied_24h,
        "implied_7d_rows": implied_7d,
        "cadence_note": note,
    }


def _has_valid_nonneg_int_key(mapping: Any, key: str) -> bool:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return False
    val = mapping.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        return False
    return val >= 0


def _build_channel_report(
    band: ResidualBand,
    *,
    warm_z_values: Optional[Sequence[float]],
    distribution_source: str,
    floored_ticks_known: bool = True,
) -> Dict[str, Any]:
    spec = band.spec
    warm_ticks = int(band.status_counts.get("warm", 0))
    warming_ticks = int(band.status_counts.get("warming", 0))
    missing_ticks = int(band.status_counts.get("missing", 0))
    frozen_ticks = int(band.status_counts.get("frozen", 0))

    if warm_z_values is not None:
        dist = _series_distribution(warm_z_values)
    else:
        dist = {
            "n": warm_ticks,
            "mean": None,
            "sd": None,
            "variance": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }

    floored_ticks: Optional[int] = band.floored_ticks if floored_ticks_known else None
    sigma_floored_fraction: Optional[float] = (
        _ratio_opt(band.floored_ticks, warm_ticks) if floored_ticks_known else None
    )

    return {
        "name": spec.name,
        "composite": spec.composite,
        "transform": spec.transform,
        "min_sigma": spec.min_sigma,
        "physical_frozen_guard": spec.physical_frozen_guard,
        "non_stationary": spec.non_stationary,
        "conditional": spec.conditional,
        "sigma_floor_watch": spec.name in SIGMA_FLOOR_WATCH_CHANNELS,
        "n": band.n,
        "observed_ticks": band.observed_ticks,
        "valid_ticks": band.valid_ticks,
        "warm_ticks": warm_ticks,
        "warming_ticks": warming_ticks,
        "missing_ticks": missing_ticks,
        "frozen_ticks": frozen_ticks,
        "warm_fraction": _ratio_opt(warm_ticks, band.observed_ticks),
        "valid_fraction": _ratio_opt(band.valid_ticks, band.observed_ticks),
        "floored_ticks": floored_ticks,
        "floored_ticks_known": floored_ticks_known,
        "sigma_floored_fraction": sigma_floored_fraction,
        "current_sigma_floored": band.sigma_floored if band.n > 0 else None,
        "baseline_mean": round(band.mean, 6) if band.n > 0 else None,
        "raw_sigma": round(band.raw_sigma, 6) if band.n > 0 else None,
        "effective_sigma": round(band.effective_sigma, 6),
        "p05": dist["p05"],
        "p50": dist["p50"],
        "p95": dist["p95"],
        "sd": dist["sd"],
        "z_mean": dist["mean"],
        "z_variance": dist["variance"],
        "z_abs_histogram": dict(band.z_abs_histogram),
        "z_abs_histogram_fractions": _histogram_fractions(band.z_abs_histogram),
        "distribution_source": distribution_source,
    }


def _build_comparison_entry(
    pair_name: str,
    correlation_key: str,
    shadow_label: str,
    legacy_label: str,
    corr: OnlineCorrelation,
    *,
    legacy_all_values: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if corr.n >= 2:
        shadow_var = max(0.0, corr.m2_x / corr.n)
        legacy_var = max(0.0, corr.m2_y / corr.n)
        shadow_sd = math.sqrt(shadow_var)
        legacy_sd = math.sqrt(legacy_var)
    else:
        shadow_var = None
        legacy_var = None
        shadow_sd = None
        legacy_sd = None

    entry: Dict[str, Any] = {
        "pair": pair_name,
        "correlation_key": correlation_key,
        "shadow_metric": shadow_label,
        "legacy_metric": legacy_label,
        "n": corr.n,
        "pearson_r": _round_opt(corr.pearson_r, 4),
        "shadow_mean": round(corr.mean_x, 6) if corr.n > 0 else None,
        "shadow_variance": round(shadow_var, 9) if shadow_var is not None else None,
        "shadow_sd": round(shadow_sd, 6) if shadow_sd is not None else None,
        "legacy_mean": round(corr.mean_y, 6) if corr.n > 0 else None,
        "legacy_variance": round(legacy_var, 9) if legacy_var is not None else None,
        "legacy_sd": round(legacy_sd, 6) if legacy_sd is not None else None,
    }
    if pair_name == "stability":
        entry["expected_direction_note"] = (
            "shadow stability.mean_abs_z is a deviation magnitude (higher = less stable), "
            "so alignment with legacy anima.stability corresponds to negative Pearson r."
        )
    if legacy_all_values is not None:
        all_dist = _series_distribution(legacy_all_values)
        entry["legacy_all_rows"] = {
            "n": all_dist["n"],
            "mean": all_dist["mean"],
            "variance": all_dist["variance"],
            "sd": all_dist["sd"],
            "p05": all_dist["p05"],
            "p50": all_dist["p50"],
            "p95": all_dist["p95"],
        }
    return entry


def _max_simultaneous_k_of_n_ticks(counts: Sequence[int], k: int) -> int:
    """Exact tight upper bound on ticks with at least ``k`` active channels."""
    if k <= 0 or len(counts) < k:
        return 0
    sorted_desc = sorted((max(0, int(c)) for c in counts), reverse=True)
    return min(sum(sorted_desc[j:]) // (k - j) for j in range(k))


def _build_stage2_readiness(
    *,
    source_kind: str,
    using_production_defaults: bool,
    min_samples: int,
    window_samples: int,
    observed_ticks: int,
    wall_clock_span_days: Optional[float],
    cadence_ratio_vs_broker: Optional[float],
    channels: Mapping[str, Mapping[str, Any]],
    composites: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    composite_shadow_raw: Mapping[str, Tuple[int, Optional[float], Optional[float]]],
    correlation_r_raw: Mapping[str, Optional[float]],
) -> Dict[str, Any]:
    """Evaluate explicit Stage-2 cutover gates without fabricating missing evidence."""
    gates: List[Dict[str, Any]] = []

    # Gate 1: 7-day live warm coverage at 2s broker cadence
    required_live_warm_ticks = DEFAULT_WINDOW_SAMPLES  # 302,400 (~7d at 2s)
    required_live_total_ticks = DEFAULT_MIN_SAMPLES + DEFAULT_WINDOW_SAMPLES
    warmth_warm_ticks = {
        ch: int(channels[ch]["warm_ticks"]) for ch in WARMTH_CHANNELS
    }
    stability_warm_ticks = {
        ch: int(channels[ch]["warm_ticks"]) for ch in STABILITY_CHANNELS
    }
    warmth_composite_warm_ticks = int(composites["warmth"]["n"])
    stability_composite_warm_ticks = int(composites["stability"]["n"])
    clarity_light_composite_warm_ticks = int(composites["clarity_light"]["n"])

    warmth_channel_bounds_ok = all(
        t >= required_live_warm_ticks for t in warmth_warm_ticks.values()
    )
    stability_channel_warm_count = sum(
        1 for t in stability_warm_ticks.values() if t >= required_live_warm_ticks
    )
    # Exact tight upper bound on simultaneous 3-of-5 stability ticks:
    # rotating 3-of-5 contributors can reach >= required_live_warm_ticks even
    # when fewer than 3 individual channels each exceed required_live_warm_ticks,
    # while still catching skewed totals like [306800, 306800, 300000, 0, 0].
    stability_simultaneous_upper_bound = _max_simultaneous_k_of_n_ticks(
        list(stability_warm_ticks.values()),
        STABILITY_MIN_CONTRIBUTORS,
    )

    if source_kind == "persisted_snapshot":
        simultaneous_proven = (
            warmth_composite_warm_ticks >= required_live_warm_ticks
            and stability_composite_warm_ticks >= required_live_warm_ticks
        )
        if not using_production_defaults:
            live_gate_status = "fail"
            live_reason = (
                f"Custom tracker parameters (min_samples={min_samples}, window_samples={window_samples}) "
                f"cannot satisfy the production 7-day live soak gate."
            )
        elif observed_ticks < required_live_total_ticks or not warmth_channel_bounds_ok:
            live_gate_status = "fail"
            min_warmth = min(warmth_warm_ticks.values())
            live_reason = (
                f"Live snapshot has {observed_ticks:,}/{required_live_total_ticks:,} total ticks "
                f"(warmup={DEFAULT_MIN_SAMPLES:,}, 7d warm target={required_live_warm_ticks:,}); "
                f"min warmth channel warm_ticks={min_warmth:,}."
            )
        elif simultaneous_proven:
            live_gate_status = "pass"
            live_reason = (
                f"Live snapshot cleared >=7d simultaneous composite warm coverage "
                f"({observed_ticks:,} ticks >= {required_live_total_ticks:,}; "
                f"paired simultaneous warm ticks: warmth={warmth_composite_warm_ticks:,}, "
                f"stability={stability_composite_warm_ticks:,} >= {required_live_warm_ticks:,})."
            )
        elif stability_simultaneous_upper_bound < required_live_warm_ticks:
            live_gate_status = "fail"
            live_reason = (
                f"Live snapshot stability channels cannot reach >=7d simultaneous 3-of-5 coverage "
                f"(pigeonhole upper bound={stability_simultaneous_upper_bound:,} < {required_live_warm_ticks:,}; "
                f"paired stability warm ticks={stability_composite_warm_ticks:,})."
            )
        else:
            live_gate_status = "unknown"
            live_reason = (
                f"Channel totals do not rule out >= {required_live_warm_ticks:,} simultaneous warm ticks "
                f"(stability 3-of-5 upper bound={stability_simultaneous_upper_bound:,}), but snapshot "
                f"cannot prove >=7d simultaneous composite contributor overlap "
                f"(paired composite warm ticks: warmth={warmth_composite_warm_ticks:,}, "
                f"stability={stability_composite_warm_ticks:,} < {required_live_warm_ticks:,})."
            )
    else:
        # Offline SQLite state_history replay cannot substitute for live 2s broker soak.
        live_gate_status = "fail"
        span_str = (
            f"{wall_clock_span_days:.2f}d"
            if wall_clock_span_days is not None
            else "unknown span"
        )
        cadence_str = (
            f"{cadence_ratio_vs_broker:.1f}x sparser than 2s broker ticks"
            if cadence_ratio_vs_broker is not None
            else "unknown cadence"
        )
        live_reason = (
            f"Offline state_history replay ({observed_ticks:,} rows over {span_str}, {cadence_str}; "
            f"simultaneous composite warm ticks: warmth={warmth_composite_warm_ticks:,}, "
            f"stability={stability_composite_warm_ticks:,}; min_samples={min_samples:,}) "
            f"does not replace live 7-day 2s-tick broker soak "
            f"(target >= {required_live_warm_ticks:,} warm ticks in live ~/.anima/sensor_residuals.json)."
        )

    offline_7d_span_and_warm = (
        wall_clock_span_days is not None
        and wall_clock_span_days >= 7.0
        and warmth_composite_warm_ticks > 0
        and stability_composite_warm_ticks > 0
    )
    gates.append(
        {
            "id": "seven_day_live_warm_coverage",
            "name": "7-day live warm coverage (2s broker ticks)",
            "status": live_gate_status,
            "blocking": True,
            "evidence": {
                "source_kind": source_kind,
                "using_production_defaults": using_production_defaults,
                "observed_ticks": observed_ticks,
                "required_live_total_ticks": required_live_total_ticks,
                "required_live_warm_ticks": required_live_warm_ticks,
                "wall_clock_span_days": _round_opt(wall_clock_span_days, 3),
                "cadence_ratio_vs_broker": cadence_ratio_vs_broker,
                "warmth_composite_warm_ticks": warmth_composite_warm_ticks,
                "stability_composite_warm_ticks": stability_composite_warm_ticks,
                "clarity_light_composite_warm_ticks": clarity_light_composite_warm_ticks,
                "warmth_channel_warm_ticks": warmth_warm_ticks,
                "stability_channel_warm_ticks": stability_warm_ticks,
                "stability_channels_meeting_7d_live": stability_channel_warm_count,
                "stability_simultaneous_upper_bound": stability_simultaneous_upper_bound,
                "offline_7d_span_and_warm": offline_7d_span_and_warm,
            },
            "reason": live_reason,
        }
    )

    # Gate 2: Non-degenerate shadow stability variance
    variance_sample_scope = (
        "paired_warm_ticks" if source_kind == "persisted_snapshot" else "all_warm_ticks"
    )
    variance_sample_label = (
        "paired warm samples (snapshot OnlineCorrelation)"
        if source_kind == "persisted_snapshot"
        else "all warm ticks"
    )
    stab_n, stab_var_raw, stab_sd_raw = composite_shadow_raw["stability"]
    stab_comp = comparisons["stability"]
    if stab_n < 2 or stab_var_raw is None or stab_sd_raw is None:
        stab_status = "unknown"
        stab_reason = (
            f"Insufficient {variance_sample_label} for stability (n={stab_n}; need >=2) "
            "to evaluate shadow stability variance."
        )
    elif stab_var_raw > MIN_NON_DEGENERATE_VARIANCE:
        stab_status = "pass"
        stab_reason = (
            f"Shadow stability.mean_abs_z is non-degenerate across {variance_sample_label} "
            f"(n={stab_n:,}, sd={stab_sd_raw:.4f}, var={stab_var_raw:.9g} > {MIN_NON_DEGENERATE_VARIANCE:g})."
        )
    else:
        stab_status = "fail"
        stab_reason = (
            f"Shadow stability.mean_abs_z variance collapsed across {variance_sample_label} "
            f"(n={stab_n:,}, var={stab_var_raw:.9g} <= {MIN_NON_DEGENERATE_VARIANCE:g})."
        )
    gates.append(
        {
            "id": "non_degenerate_stability_variance",
            "name": "Non-degenerate shadow stability variance",
            "status": stab_status,
            "blocking": True,
            "evidence": {
                "sample_scope": variance_sample_scope,
                "n": stab_n,
                "shadow_variance": round(stab_var_raw, 9) if stab_var_raw is not None else None,
                "shadow_sd": round(stab_sd_raw, 6) if stab_sd_raw is not None else None,
                "paired_n": int(stab_comp["n"]),
                "legacy_variance": stab_comp["legacy_variance"],
                "legacy_sd": stab_comp["legacy_sd"],
                "min_variance_threshold": MIN_NON_DEGENERATE_VARIANCE,
            },
            "reason": stab_reason,
        }
    )

    # Gate 3: Non-degenerate shadow warmth variance
    warmth_n, warmth_var_raw, warmth_sd_raw = composite_shadow_raw["warmth"]
    warmth_comp = comparisons["warmth"]
    if warmth_n < 2 or warmth_var_raw is None or warmth_sd_raw is None:
        warmth_status = "unknown"
        warmth_reason = (
            f"Insufficient {variance_sample_label} for warmth (n={warmth_n}; need >=2) "
            "to evaluate shadow warmth variance."
        )
    elif warmth_var_raw > MIN_NON_DEGENERATE_VARIANCE:
        warmth_status = "pass"
        warmth_reason = (
            f"Shadow warmth.signed_z is non-degenerate across {variance_sample_label} "
            f"(n={warmth_n:,}, sd={warmth_sd_raw:.4f}, var={warmth_var_raw:.9g} > {MIN_NON_DEGENERATE_VARIANCE:g})."
        )
    else:
        warmth_status = "fail"
        warmth_reason = (
            f"Shadow warmth.signed_z variance collapsed across {variance_sample_label} "
            f"(n={warmth_n:,}, var={warmth_var_raw:.9g} <= {MIN_NON_DEGENERATE_VARIANCE:g})."
        )
    gates.append(
        {
            "id": "non_degenerate_warmth_variance",
            "name": "Non-degenerate shadow warmth variance",
            "status": warmth_status,
            "blocking": True,
            "evidence": {
                "sample_scope": variance_sample_scope,
                "n": warmth_n,
                "shadow_variance": round(warmth_var_raw, 9) if warmth_var_raw is not None else None,
                "shadow_sd": round(warmth_sd_raw, 6) if warmth_sd_raw is not None else None,
                "paired_n": int(warmth_comp["n"]),
                "legacy_variance": warmth_comp["legacy_variance"],
                "legacy_sd": warmth_comp["legacy_sd"],
                "min_variance_threshold": MIN_NON_DEGENERATE_VARIANCE,
            },
            "reason": warmth_reason,
        }
    )

    # Gate 4: Shadow vs legacy correlation direction alignment (warmth r > 0, stability r < 0)
    warmth_r_raw = correlation_r_raw.get("warmth")
    stab_r_raw = correlation_r_raw.get("stability")
    wrong_sign: List[str] = []
    missing_r: List[str] = []
    if warmth_r_raw is None:
        missing_r.append("warmth")
    elif warmth_r_raw <= 0.0:
        wrong_sign.append(f"warmth (r={warmth_r_raw:.4f}, expected > 0)")
    if stab_r_raw is None:
        missing_r.append("stability")
    elif stab_r_raw >= 0.0:
        wrong_sign.append(f"stability (r={stab_r_raw:.4f}, expected < 0)")

    if wrong_sign:
        corr_status = "fail"
        extra_missing = f"; missing pearson_r on {', '.join(missing_r)}" if missing_r else ""
        corr_reason = (
            f"Wrong-sign shadow vs legacy correlation on {', '.join(wrong_sign)}{extra_missing}."
        )
    elif missing_r:
        corr_status = "unknown"
        corr_reason = (
            f"Pearson r unavailable on {', '.join(missing_r)} "
            f"(warmth paired n={warmth_comp['n']:,}, stability paired n={stab_comp['n']:,}); "
            "cannot confirm shadow-vs-legacy correlation alignment."
        )
    else:
        corr_status = "pass"
        corr_reason = (
            f"Shadow vs legacy correlations have expected signs "
            f"(warmth r={warmth_r_raw:.4f} > 0, n={warmth_comp['n']:,}; "
            f"stability r={stab_r_raw:.4f} < 0, n={stab_comp['n']:,})."
        )
    gates.append(
        {
            "id": "shadow_legacy_correlation_alignment",
            "name": "Shadow vs legacy correlation alignment (warmth > 0, stability < 0)",
            "status": corr_status,
            "blocking": True,
            "evidence": {
                "warmth_paired_n": int(warmth_comp["n"]),
                "warmth_pearson_r": warmth_comp["pearson_r"],
                "stability_paired_n": int(stab_comp["n"]),
                "stability_pearson_r": stab_comp["pearson_r"],
                "wrong_sign": wrong_sign,
                "missing_correlation": missing_r,
            },
            "reason": corr_reason,
        }
    )

    # Gate 5: Sigma-floor watch channels (memory_percent, eeg_delta_power, eeg_theta_power)
    watch_evidence: Dict[str, Any] = {}
    high_floored: List[str] = []
    unwarm_watch: List[str] = []
    missing_floored_watch: List[str] = []
    currently_floored_watch: List[str] = []
    for ch_name in SIGMA_FLOOR_WATCH_CHANNELS:
        ch = channels[ch_name]
        sff = ch["sigma_floored_fraction"]
        floored_known = bool(ch.get("floored_ticks_known", True))
        if ch["warm_ticks"] <= 0:
            unwarm_watch.append(ch_name)
        elif not floored_known or sff is None:
            missing_floored_watch.append(ch_name)
        if sff is not None and sff >= SIGMA_FLOOR_WARN_FRACTION:
            high_floored.append(f"{ch_name} ({sff * 100:.1f}%)")
        if ch["current_sigma_floored"] is True:
            currently_floored_watch.append(ch_name)
        watch_evidence[ch_name] = {
            "n": ch["n"],
            "warm_ticks": ch["warm_ticks"],
            "raw_sigma": ch["raw_sigma"],
            "min_sigma": ch["min_sigma"],
            "current_sigma_floored": ch["current_sigma_floored"],
            "floored_ticks_known": floored_known,
            "sigma_floored_fraction": sff,
        }

    incomplete_parts: List[str] = []
    if unwarm_watch:
        incomplete_parts.append(f"no warm ticks on {', '.join(unwarm_watch)}")
    if missing_floored_watch:
        incomplete_parts.append(
            f"missing floored_ticks in snapshot on {', '.join(missing_floored_watch)}"
        )

    if high_floored:
        watch_status = "warn"
        incomplete_note = (
            f" ({'; '.join(incomplete_parts)})" if incomplete_parts else ""
        )
        watch_reason = (
            f"High sigma-floored fraction (>={SIGMA_FLOOR_WARN_FRACTION * 100:.0f}%) on "
            f"{', '.join(high_floored)}{incomplete_note}; review min_sigma floors before Stage-2 cutover."
        )
    elif incomplete_parts:
        watch_status = "unknown"
        curr_note = (
            f"; current raw_sigma < min_sigma on: {', '.join(currently_floored_watch)}"
            if currently_floored_watch
            else ""
        )
        watch_reason = (
            f"Incomplete sigma-floor watch evidence ({'; '.join(incomplete_parts)}){curr_note}."
        )
    else:
        watch_status = "pass"
        watch_reason = (
            "All three sigma-floor watch channels (memory_percent, eeg_delta_power, "
            f"eeg_theta_power) had warm ticks and stayed below {SIGMA_FLOOR_WARN_FRACTION * 100:.0f}% floored."
        )
    gates.append(
        {
            "id": "sigma_floor_watch_channels",
            "name": "Sigma-floor watch (memory_percent, eeg_delta_power, eeg_theta_power)",
            "status": watch_status,
            "blocking": False,
            "evidence": watch_evidence,
            "reason": watch_reason,
        }
    )

    # Gate 6: Python + Elixir parity checklist (manual release/code-parity gate)
    gates.append(
        {
            "id": "python_elixir_parity_checklist",
            "name": "Python + Elixir parity & Pi release checklist",
            "status": "manual_required",
            "blocking": True,
            "evidence": {
                "files_requiring_parity": [
                    "src/anima_mcp/anima.py",
                    "src/anima_mcp/eisv_mapper.py",
                    "anima_broker/lib/anima_broker/governance/eisv_mapper.ex",
                ],
                "checklist": [
                    {"item": item, "verified": False}
                    for item in PYTHON_ELIXIR_PARITY_ITEMS
                ],
            },
            "reason": (
                "Stage-2 cutover requires manual verification of Python+Elixir parity "
                "and an on-Pi Elixir release rebuild (anima-broker-ex)."
            ),
        }
    )

    blocking_gates = [
        g["id"] for g in gates if g["blocking"] and g["status"] != "pass"
    ]
    return {
        "stage2_ready": len(blocking_gates) == 0,
        "blocking_gates": blocking_gates,
        "gates": gates,
    }


def replay_state_history(
    db_path: Path | str,
    *,
    days: Optional[float] = None,
    min_samples: Optional[int] = None,
    window_samples: Optional[int] = None,
    frozen_run_threshold: Optional[int] = None,
    immutable: bool = False,
) -> Dict[str, Any]:
    """Replay historical ``state_history`` rows through ``SensorResidualTracker``."""
    _validate_days(days)
    eff_min = DEFAULT_MIN_SAMPLES if min_samples is None else max(1, int(min_samples))
    eff_window = (
        DEFAULT_WINDOW_SAMPLES
        if window_samples is None
        else max(1, int(window_samples))
    )
    eff_window = max(eff_min, eff_window)
    eff_frozen = (
        DEFAULT_FROZEN_RUN_THRESHOLD
        if frozen_run_threshold is None
        else max(2, int(frozen_run_threshold))
    )
    using_prod_defaults = (
        min_samples is None
        and window_samples is None
        and frozen_run_threshold is None
    )

    # Isolated in-memory tracker: auto_load=False, never calls save().
    tracker = SensorResidualTracker(
        persistence_path=Path("/nonexistent/anima_residual_backtest_memory_only.json"),
        min_samples=eff_min,
        window_samples=eff_window,
        frozen_run_threshold=eff_frozen,
        auto_load=False,
    )

    channel_warm_zs: Dict[str, array] = {
        spec.name: array("d") for spec in CHANNEL_SPECS
    }
    composite_warm_vals: Dict[str, array] = {
        name: array("d") for name in COMPOSITE_ORDER
    }
    composite_status_counts: Dict[str, Dict[str, int]] = {
        name: {} for name in COMPOSITE_ORDER
    }
    composite_valid_ticks: Dict[str, int] = {name: 0 for name in COMPOSITE_ORDER}
    composite_floored_ticks: Dict[str, int] = {name: 0 for name in COMPOSITE_ORDER}
    stability_contributor_counts: array = array("d")
    stability_floored_contributor_fracs: array = array("d")
    legacy_all_vals: Dict[str, array] = {
        "warmth": array("d"),
        "stability": array("d"),
        "clarity_light": array("d"),
    }

    intervals_s: array = array("d")
    counters: Dict[str, int] = {
        "skipped_bad_timestamp_rows": 0,
        "malformed_sensor_json_rows": 0,
        "null_anima_rows": 0,
    }
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    first_unix: Optional[float] = None
    last_unix: Optional[float] = None
    prev_unix: Optional[float] = None
    rows_replayed = 0

    for rec in iter_state_history_records(
        db_path, days=days, immutable=immutable, counters=counters
    ):
        rows_replayed += 1
        if first_ts is None:
            first_ts = rec.timestamp
            first_unix = rec.timestamp_unix
        last_ts = rec.timestamp
        last_unix = rec.timestamp_unix
        if prev_unix is not None:
            intervals_s.append(rec.timestamp_unix - prev_unix)
        prev_unix = rec.timestamp_unix

        if rec.anima.warmth is not None:
            legacy_all_vals["warmth"].append(rec.anima.warmth)
        if rec.anima.stability is not None:
            legacy_all_vals["stability"].append(rec.anima.stability)
        if rec.anima.clarity is not None:
            legacy_all_vals["clarity_light"].append(rec.anima.clarity)

        snap = tracker.observe(
            rec.readings,
            external_light_lux=rec.external_light_lux,
            current_warmth=rec.anima.warmth,
            current_stability=rec.anima.stability,
            current_clarity=rec.anima.clarity,
            now=rec.timestamp_unix,
        )

        ch_snaps = snap["channels"]
        for spec in CHANNEL_SPECS:
            ch_entry = ch_snaps[spec.name]
            if ch_entry["status"] == "warm" and ch_entry["z"] is not None:
                channel_warm_zs[spec.name].append(float(ch_entry["z"]))

        comp_snaps = snap["composites"]

        # Warmth composite tracking
        w_comp = comp_snaps["warmth"]
        w_status = str(w_comp["status"])
        composite_status_counts["warmth"][w_status] = (
            composite_status_counts["warmth"].get(w_status, 0) + 1
        )
        if all(
            ch_snaps[ch]["status"] in ("warm", "warming") for ch in WARMTH_CHANNELS
        ):
            composite_valid_ticks["warmth"] += 1
        if w_status == "warm" and w_comp["signed_z"] is not None:
            composite_warm_vals["warmth"].append(float(w_comp["signed_z"]))
            if any(bool(ch_snaps[ch]["sigma_floored"]) for ch in WARMTH_CHANNELS):
                composite_floored_ticks["warmth"] += 1

        # Stability composite tracking
        s_comp = comp_snaps["stability"]
        s_status = str(s_comp["status"])
        composite_status_counts["stability"][s_status] = (
            composite_status_counts["stability"].get(s_status, 0) + 1
        )
        valid_stab_channels = sum(
            1
            for ch in STABILITY_CHANNELS
            if ch_snaps[ch]["status"] in ("warm", "warming")
        )
        if valid_stab_channels >= STABILITY_MIN_CONTRIBUTORS:
            composite_valid_ticks["stability"] += 1
        if s_status == "warm" and s_comp["mean_abs_z"] is not None:
            composite_warm_vals["stability"].append(float(s_comp["mean_abs_z"]))
            active_contribs = s_comp.get("contributors") or []
            stability_contributor_counts.append(float(len(active_contribs)))
            if active_contribs:
                floored_active = sum(
                    1 for ch in active_contribs if bool(ch_snaps[ch]["sigma_floored"])
                )
                stability_floored_contributor_fracs.append(
                    floored_active / len(active_contribs)
                )
                if floored_active > 0:
                    composite_floored_ticks["stability"] += 1

        # Clarity light composite tracking
        l_comp = comp_snaps["clarity_light"]
        l_status = str(l_comp["status"])
        composite_status_counts["clarity_light"][l_status] = (
            composite_status_counts["clarity_light"].get(l_status, 0) + 1
        )
        if ch_snaps["external_light_lux"]["status"] in ("warm", "warming"):
            composite_valid_ticks["clarity_light"] += 1
        if l_status == "warm" and l_comp["z"] is not None:
            composite_warm_vals["clarity_light"].append(float(l_comp["z"]))
            if bool(ch_snaps["external_light_lux"]["sigma_floored"]):
                composite_floored_ticks["clarity_light"] += 1

    wall_span_s: Optional[float] = None
    wall_span_days: Optional[float] = None
    if first_unix is not None and last_unix is not None:
        wall_span_s = max(0.0, last_unix - first_unix)
        wall_span_days = wall_span_s / 86400.0

    sampling = _summarize_intervals(intervals_s)

    channels_report: Dict[str, Dict[str, Any]] = {}
    for spec in CHANNEL_SPECS:
        channels_report[spec.name] = _build_channel_report(
            tracker.bands[spec.name],
            warm_z_values=channel_warm_zs[spec.name],
            distribution_source="replay_exact_z_series",
        )

    composites_report: Dict[str, Dict[str, Any]] = {}
    composite_shadow_raw: Dict[str, Tuple[int, Optional[float], Optional[float]]] = {}
    for comp_name in COMPOSITE_ORDER:
        warm_vals = composite_warm_vals[comp_name]
        dist = _series_distribution(warm_vals)
        warm_ticks = len(warm_vals)
        composite_shadow_raw[comp_name] = (
            warm_ticks,
            dist["variance_raw"],
            dist["sd_raw"],
        )
        entry: Dict[str, Any] = {
            "name": comp_name,
            "metric": COMPOSITE_METRIC_NAMES[comp_name],
            "n": warm_ticks,
            "observed_ticks": rows_replayed,
            "valid_ticks": composite_valid_ticks[comp_name],
            "warm_fraction": _ratio_opt(warm_ticks, rows_replayed),
            "valid_fraction": _ratio_opt(composite_valid_ticks[comp_name], rows_replayed),
            "floored_ticks": composite_floored_ticks[comp_name],
            "sigma_floored_fraction": _ratio_opt(
                composite_floored_ticks[comp_name], warm_ticks
            ),
            "p05": dist["p05"],
            "p50": dist["p50"],
            "p95": dist["p95"],
            "sd": dist["sd"],
            "mean": dist["mean"],
            "variance": dist["variance"],
            "status_counts": dict(composite_status_counts[comp_name]),
            "distribution_source": "replay_exact_composite_series",
        }
        if comp_name == "stability":
            entry["mean_contributor_count"] = (
                round(
                    math.fsum(stability_contributor_counts)
                    / len(stability_contributor_counts),
                    3,
                )
                if stability_contributor_counts
                else None
            )
            entry["mean_floored_contributor_fraction"] = (
                round(
                    math.fsum(stability_floored_contributor_fracs)
                    / len(stability_floored_contributor_fracs),
                    4,
                )
                if stability_floored_contributor_fracs
                else None
            )
        composites_report[comp_name] = entry

    comparisons_report: Dict[str, Dict[str, Any]] = {}
    correlation_r_raw: Dict[str, Optional[float]] = {}
    for pair_name, corr_key, shadow_label, legacy_label in COMPARISON_SPECS:
        corr = tracker.correlations[corr_key]
        correlation_r_raw[pair_name] = corr.pearson_r
        comparisons_report[pair_name] = _build_comparison_entry(
            pair_name,
            corr_key,
            shadow_label,
            legacy_label,
            corr,
            legacy_all_values=legacy_all_vals[pair_name],
        )

    readiness = _build_stage2_readiness(
        source_kind="sqlite_replay",
        using_production_defaults=using_prod_defaults,
        min_samples=eff_min,
        window_samples=eff_window,
        observed_ticks=rows_replayed,
        wall_clock_span_days=wall_span_days,
        cadence_ratio_vs_broker=sampling["cadence_ratio_vs_broker"],
        channels=channels_report,
        composites=composites_report,
        comparisons=comparisons_report,
        composite_shadow_raw=composite_shadow_raw,
        correlation_r_raw=correlation_r_raw,
    )

    return {
        "available": True,
        "backtest_schema_version": BACKTEST_SCHEMA_VERSION,
        "residuals_schema_version": RESIDUALS_SCHEMA_VERSION,
        "authority": AUTHORITY,
        "moves_no_gate": MOVES_NO_GATE,
        "source": {
            "kind": "sqlite_replay",
            "path": str(Path(db_path).expanduser()),
            "sqlite_immutable": bool(immutable),
            "window_days_requested": days,
            "window_anchor": "max_valid_db_timestamp" if days is not None else "full_history",
            "wall_clock_start": first_ts.isoformat() if first_ts is not None else None,
            "wall_clock_end": last_ts.isoformat() if last_ts is not None else None,
            "wall_clock_span_seconds": _round_opt(wall_span_s, 1),
            "wall_clock_span_days": _round_opt(wall_span_days, 3),
            "rows_replayed": rows_replayed,
            "data_hygiene": counters,
        },
        "config": {
            "mode": "production_defaults" if using_prod_defaults else "custom_override",
            "using_production_defaults": using_prod_defaults,
            "min_samples": eff_min,
            "window_samples": eff_window,
            "frozen_run_threshold": eff_frozen,
            "production_defaults": {
                "min_samples": DEFAULT_MIN_SAMPLES,
                "window_samples": DEFAULT_WINDOW_SAMPLES,
                "frozen_run_threshold": DEFAULT_FROZEN_RUN_THRESHOLD,
            },
        },
        "sampling": sampling,
        "sigma_floor_watch_channels": list(SIGMA_FLOOR_WATCH_CHANNELS),
        "channels": channels_report,
        "composites": composites_report,
        "comparisons": comparisons_report,
        "stage2_readiness": readiness,
    }


def evaluate_snapshot(
    snapshot_path: Path | str,
    *,
    min_samples: Optional[int] = None,
    window_samples: Optional[int] = None,
    frozen_run_threshold: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a persisted ``sensor_residuals.json`` snapshot strictly read-only."""
    target = Path(snapshot_path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Snapshot file not found: {target}")
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != RESIDUALS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported or invalid sensor_residuals snapshot schema in {target}"
        )

    eff_min = DEFAULT_MIN_SAMPLES if min_samples is None else max(1, int(min_samples))
    eff_window = (
        DEFAULT_WINDOW_SAMPLES
        if window_samples is None
        else max(1, int(window_samples))
    )
    eff_window = max(eff_min, eff_window)
    eff_frozen = (
        DEFAULT_FROZEN_RUN_THRESHOLD
        if frozen_run_threshold is None
        else max(2, int(frozen_run_threshold))
    )
    using_prod_defaults = (
        min_samples is None
        and window_samples is None
        and frozen_run_threshold is None
    )

    restore_comparison = raw.get("comparison_version") == COMPARISON_VERSION
    try:
        observed_ticks = max(0, int(raw.get("observed_ticks", 0)))
    except (TypeError, ValueError):
        observed_ticks = 0

    saved_at = _finite_float(raw.get("saved_at"))
    saved_at_iso = (
        datetime.fromtimestamp(saved_at, tz=timezone.utc).isoformat()
        if saved_at is not None
        else None
    )

    channels_raw = raw.get("channels")
    bands: Dict[str, ResidualBand] = {}
    channels_report: Dict[str, Dict[str, Any]] = {}
    for spec in CHANNEL_SPECS:
        ch_data = (
            channels_raw.get(spec.name) if isinstance(channels_raw, Mapping) else None
        )
        floored_ticks_known = restore_comparison and _has_valid_nonneg_int_key(
            ch_data, "floored_ticks"
        )
        # Strip invalid floored_ticks (e.g. JSON Infinity/1e309) on a shallow copy
        # so ResidualBand.from_dict cannot raise OverflowError on int(inf).
        if isinstance(ch_data, Mapping) and not _has_valid_nonneg_int_key(
            ch_data, "floored_ticks"
        ):
            band_input: Any = {
                k: v for k, v in ch_data.items() if k != "floored_ticks"
            }
        else:
            band_input = ch_data
        band = ResidualBand.from_dict(
            spec,
            band_input,
            min_samples=eff_min,
            window_samples=eff_window,
            frozen_run_threshold=eff_frozen,
            restore_comparison=restore_comparison,
        )
        bands[spec.name] = band
        channels_report[spec.name] = _build_channel_report(
            band,
            warm_z_values=None,
            distribution_source="snapshot_histogram_only",
            floored_ticks_known=floored_ticks_known,
        )

    corr_raw = raw.get("correlations")
    correlations: Dict[str, OnlineCorrelation] = {}
    for key in CORRELATION_KEYS:
        entry = (
            corr_raw.get(key)
            if restore_comparison and isinstance(corr_raw, Mapping)
            else None
        )
        correlations[key] = OnlineCorrelation.from_dict(entry)

    comparisons_report: Dict[str, Dict[str, Any]] = {}
    composite_shadow_raw: Dict[str, Tuple[int, Optional[float], Optional[float]]] = {}
    correlation_r_raw: Dict[str, Optional[float]] = {}
    for pair_name, corr_key, shadow_label, legacy_label in COMPARISON_SPECS:
        corr = correlations[corr_key]
        correlation_r_raw[pair_name] = corr.pearson_r
        if corr.n >= 2:
            raw_var = max(0.0, corr.m2_x / corr.n)
            raw_sd = math.sqrt(raw_var)
        else:
            raw_var = None
            raw_sd = None
        composite_shadow_raw[pair_name] = (corr.n, raw_var, raw_sd)
        comparisons_report[pair_name] = _build_comparison_entry(
            pair_name,
            corr_key,
            shadow_label,
            legacy_label,
            corr,
            legacy_all_values=None,
        )

    # Composite stats from snapshot: exact paired count/mean/sd/var from OnlineCorrelation;
    # exact quantiles are unavailable without raw tick series (fail toward unknown).
    persisted_composites = (
        raw.get("composites") if isinstance(raw.get("composites"), Mapping) else {}
    )
    composites_report: Dict[str, Dict[str, Any]] = {}
    for pair_name, corr_key, _, _ in COMPARISON_SPECS:
        corr = correlations[corr_key]
        comp_stats = comparisons_report[pair_name]
        if pair_name == "clarity_light":
            valid_frac = channels_report["external_light_lux"]["valid_fraction"]
            floored_frac = channels_report["external_light_lux"]["sigma_floored_fraction"]
        else:
            valid_frac = None
            floored_frac = None
        latest_comp = persisted_composites.get(pair_name)
        latest_status = (
            latest_comp.get("status") if isinstance(latest_comp, Mapping) else None
        )
        composites_report[pair_name] = {
            "name": pair_name,
            "metric": COMPOSITE_METRIC_NAMES[pair_name],
            "n": corr.n,
            "observed_ticks": observed_ticks,
            "valid_ticks": None,
            "warm_fraction": _ratio_opt(corr.n, observed_ticks),
            "valid_fraction": valid_frac,
            "floored_ticks": None,
            "sigma_floored_fraction": floored_frac,
            "p05": None,
            "p50": None,
            "p95": None,
            "sd": comp_stats["shadow_sd"],
            "mean": comp_stats["shadow_mean"],
            "variance": comp_stats["shadow_variance"],
            "latest_status": latest_status,
            "distribution_source": "snapshot_online_moments_only",
        }

    approx_span_days = (
        (observed_ticks * BROKER_TICK_SECONDS) / 86400.0
        if observed_ticks > 0
        else 0.0
    )

    readiness = _build_stage2_readiness(
        source_kind="persisted_snapshot",
        using_production_defaults=using_prod_defaults,
        min_samples=eff_min,
        window_samples=eff_window,
        observed_ticks=observed_ticks,
        wall_clock_span_days=approx_span_days,
        cadence_ratio_vs_broker=1.0,
        channels=channels_report,
        composites=composites_report,
        comparisons=comparisons_report,
        composite_shadow_raw=composite_shadow_raw,
        correlation_r_raw=correlation_r_raw,
    )

    return {
        "available": True,
        "backtest_schema_version": BACKTEST_SCHEMA_VERSION,
        "residuals_schema_version": RESIDUALS_SCHEMA_VERSION,
        "authority": AUTHORITY,
        "moves_no_gate": MOVES_NO_GATE,
        "source": {
            "kind": "persisted_snapshot",
            "path": str(target),
            "saved_at": saved_at,
            "saved_at_iso": saved_at_iso,
            "comparison_version": raw.get("comparison_version"),
            "observed_ticks": observed_ticks,
            "implied_live_span_days_at_2s": round(approx_span_days, 3),
        },
        "config": {
            "mode": "production_defaults" if using_prod_defaults else "custom_override",
            "using_production_defaults": using_prod_defaults,
            "min_samples": eff_min,
            "window_samples": eff_window,
            "frozen_run_threshold": eff_frozen,
            "production_defaults": {
                "min_samples": DEFAULT_MIN_SAMPLES,
                "window_samples": DEFAULT_WINDOW_SAMPLES,
                "frozen_run_threshold": DEFAULT_FROZEN_RUN_THRESHOLD,
            },
        },
        "sampling": {
            "broker_tick_seconds": BROKER_TICK_SECONDS,
            "cadence_ratio_vs_broker": 1.0,
            "cadence_note": (
                f"Persisted live broker snapshot: {observed_ticks:,} observed ticks "
                f"(~{approx_span_days:.2f}d at {BROKER_TICK_SECONDS:.0f}s ticks)."
            ),
        },
        "sigma_floor_watch_channels": list(SIGMA_FLOOR_WATCH_CHANNELS),
        "channels": channels_report,
        "composites": composites_report,
        "comparisons": comparisons_report,
        "stage2_readiness": readiness,
    }


def _fmt_num(val: Optional[float], digits: int = 3) -> str:
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    return f"{val * 100:.1f}%"


def format_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a Stage-2 readiness report as human-readable GitHub Markdown."""
    src = report["source"]
    cfg = report["config"]
    sampling = report["sampling"]
    channels = report["channels"]
    composites = report["composites"]
    comparisons = report["comparisons"]
    readiness = report["stage2_readiness"]

    lines: List[str] = []
    lines.append("# Sensor Residual Backtest & Stage-2 Readiness Report")
    lines.append("")
    lines.append(
        f"- **Source**: `{src['kind']}` (`{src['path']}`)"
    )
    mode_label = (
        "PRODUCTION DEFAULTS"
        if cfg["using_production_defaults"]
        else "CUSTOM OVERRIDE"
    )
    lines.append(
        f"- **Tracker Config**: **{mode_label}** "
        f"(`min_samples={cfg['min_samples']:,}`, "
        f"`window_samples={cfg['window_samples']:,}`, "
        f"`frozen_run_threshold={cfg['frozen_run_threshold']}`)"
    )
    if src["kind"] == "sqlite_replay":
        win_label = (
            f"last {src['window_days_requested']} days (anchored to max DB timestamp)"
            if src["window_days_requested"] is not None
            else "full history"
        )
        lines.append(
            f"- **Window**: {win_label} | **Rows replayed**: `{src['rows_replayed']:,}` | "
            f"**Wall-clock span**: `{_fmt_num(src['wall_clock_span_days'], 2)} days` "
            f"(`{src['wall_clock_start']}` -> `{src['wall_clock_end']}`)"
        )
        lines.append(f"- **Sampling Cadence**: {sampling['cadence_note']}")
        if sampling.get("p10_seconds") is not None:
            lines.append(
                f"  - Interval distribution: `p10={sampling['p10_seconds']}s`, "
                f"`median={sampling['median_seconds']}s`, `p90={sampling['p90_seconds']}s`, "
                f"`min={sampling['min_seconds']}s`, `max={sampling['max_seconds']}s`"
            )
        hyg = src["data_hygiene"]
        lines.append(
            f"- **Data Hygiene**: `malformed_sensor_json_rows={hyg['malformed_sensor_json_rows']}`, "
            f"`skipped_bad_timestamp_rows={hyg['skipped_bad_timestamp_rows']}`, "
            f"`null_anima_rows={hyg['null_anima_rows']}`"
        )
    else:
        lines.append(
            f"- **Snapshot Ticks**: `{src['observed_ticks']:,}` "
            f"(~`{src['implied_live_span_days_at_2s']:.2f} days` at 2s broker ticks; "
            f"`saved_at={src['saved_at_iso']}`)"
        )
    lines.append("")

    # Per-channel table
    lines.append("## Channel Summary")
    lines.append("")
    lines.append(
        "| Channel | Composite | n | Valid% | Warm% | Floored% | raw_sigma / min_sigma | p05 | p50 | p95 | sd | Flags |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for spec in CHANNEL_SPECS:
        ch = channels[spec.name]
        flags: List[str] = []
        if ch["sigma_floor_watch"]:
            flags.append("FLOOR_WATCH")
        if ch["current_sigma_floored"] is True:
            flags.append("curr_floored")
        if ch["non_stationary"]:
            flags.append("non_stationary")
        if ch["conditional"]:
            flags.append(ch["conditional"])
        raw_sig_s = _fmt_num(ch["raw_sigma"], 4)
        min_sig_s = _fmt_num(ch["min_sigma"], 4)
        lines.append(
            f"| `{ch['name']}` | {ch['composite'] or '-'} | {ch['n']:,} | "
            f"{_fmt_pct(ch['valid_fraction'])} | {_fmt_pct(ch['warm_fraction'])} | "
            f"{_fmt_pct(ch['sigma_floored_fraction'])} | {raw_sig_s} / {min_sig_s} | "
            f"{_fmt_num(ch['p05'], 3)} | {_fmt_num(ch['p50'], 3)} | "
            f"{_fmt_num(ch['p95'], 3)} | {_fmt_num(ch['sd'], 3)} | "
            f"{', '.join(flags) if flags else '-'} |"
        )
    lines.append("")

    # Highlighted sigma-floor watch channels
    lines.append("### Sigma-Floor Watch Channels (`memory_percent`, `eeg_delta_power`, `eeg_theta_power`)")
    lines.append("")
    for ch_name in SIGMA_FLOOR_WATCH_CHANNELS:
        ch = channels[ch_name]
        lines.append(
            f"- **`{ch_name}`**: `n={ch['n']:,}`, `warm_ticks={ch['warm_ticks']:,}`, "
            f"`raw_sigma={_fmt_num(ch['raw_sigma'], 4)}` vs `min_sigma={ch['min_sigma']}`, "
            f"`current_sigma_floored={ch['current_sigma_floored']}`, "
            f"`sigma_floored_fraction={_fmt_pct(ch['sigma_floored_fraction'])}`"
        )
    lines.append("")

    # Composite table
    lines.append("## Composite Summary")
    lines.append("")
    lines.append(
        "| Composite | Metric | n (warm) | Valid% | Warm% | Floored% | p05 | p50 | p95 | sd | var |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for comp_name in COMPOSITE_ORDER:
        comp = composites[comp_name]
        lines.append(
            f"| `{comp_name}` | `{comp['metric']}` | {comp['n']:,} | "
            f"{_fmt_pct(comp['valid_fraction'])} | {_fmt_pct(comp['warm_fraction'])} | "
            f"{_fmt_pct(comp['sigma_floored_fraction'])} | "
            f"{_fmt_num(comp['p05'], 3)} | {_fmt_num(comp['p50'], 3)} | "
            f"{_fmt_num(comp['p95'], 3)} | {_fmt_num(comp['sd'], 4)} | "
            f"{_fmt_num(comp['variance'], 6)} |"
        )
    lines.append("")

    # Shadow vs Legacy comparison table
    lines.append("## Shadow vs Legacy Correlations & Variance")
    lines.append("")
    lines.append(
        "| Pair | n (paired) | Pearson r | Shadow sd (var) | Legacy sd (var) | Legacy all-rows sd (var) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for pair_name, _, _, _ in COMPARISON_SPECS:
        comp = comparisons[pair_name]
        shadow_sv = (
            f"{_fmt_num(comp['shadow_sd'], 4)} ({_fmt_num(comp['shadow_variance'], 6)})"
            if comp["shadow_sd"] is not None
            else "n/a"
        )
        legacy_sv = (
            f"{_fmt_num(comp['legacy_sd'], 4)} ({_fmt_num(comp['legacy_variance'], 6)})"
            if comp["legacy_sd"] is not None
            else "n/a"
        )
        all_rows = comp.get("legacy_all_rows")
        if isinstance(all_rows, Mapping) and all_rows.get("sd") is not None:
            all_sv = f"{_fmt_num(all_rows['sd'], 4)} ({_fmt_num(all_rows['variance'], 6)})"
        else:
            all_sv = "n/a"
        lines.append(
            f"| `{pair_name}` | {comp['n']:,} | {_fmt_num(comp['pearson_r'], 4)} | "
            f"{shadow_sv} | {legacy_sv} | {all_sv} |"
        )
    lines.append("")
    lines.append(
        "> Note: `stability` shadow `mean_abs_z` measures deviation magnitude (higher = less stable), "
        "whereas legacy `anima.stability` is higher when stable; aligned behavior yields negative Pearson `r`."
    )
    lines.append("")

    # Stage-2 Gate Readiness Checklist
    lines.append("## Stage-2 Gate Readiness Checklist")
    lines.append("")
    verdict = "READY" if readiness["stage2_ready"] else "NOT READY"
    lines.append(f"- **Overall Stage-2 Cutover Status**: **{verdict}**")
    if readiness["blocking_gates"]:
        lines.append(
            f"- **Blocking Gates**: {', '.join(f'`{g}`' for g in readiness['blocking_gates'])}"
        )
    lines.append("")
    for gate in readiness["gates"]:
        tag = gate["status"].upper()
        lines.append(f"- **[{tag}] {gate['name']}** (`{gate['id']}`): {gate['reason']}")
        if gate["id"] == "python_elixir_parity_checklist":
            for item in gate["evidence"]["checklist"]:
                box = "x" if item["verified"] else " "
                lines.append(f"  - [{box}] {item['item']}")
    lines.append("")
    return "\n".join(lines)
