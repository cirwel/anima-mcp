"""
Display Screens - Info screen mixin.

Renders sensors, identity, diagnostics, and health screens.
"""

import math
import sys
from typing import Optional, Dict, Any

from .design import COLORS
from ..anima import Anima
from ..sensors.base import SensorReadings
from ..identity.store import CreatureIdentity


_EISV_KEYS = ("E", "I", "S", "V")


def _validated_eisv_vector(candidate: Any) -> Optional[Dict[str, float]]:
    """Accept only complete finite vectors so absent values never render as zero."""
    if not isinstance(candidate, dict):
        return None
    vector: Dict[str, float] = {}
    for key in _EISV_KEYS:
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        lower, upper = (-1.0, 1.0) if key == "V" else (0.0, 1.0)
        if not lower <= numeric <= upper:
            return None
        vector[key] = numeric
    return vector


def _select_eisv_for_display(
    governance: Optional[Dict[str, Any]],
) -> tuple[Optional[Dict[str, float]], str]:
    """Prefer UNITARES state, then explicitly labeled body input telemetry."""
    if not governance:
        return None, ""
    governance_eisv = _validated_eisv_vector(
        governance.get("governance_eisv")
    )
    if governance_eisv is not None:
        return governance_eisv, "gov state"
    body_projection = _validated_eisv_vector(
        governance.get("body_eisv_projection")
    )
    if body_projection is None:
        body_projection = _validated_eisv_vector(governance.get("eisv"))
    if body_projection is not None:
        return body_projection, "body input"
    return None, ""


def _eisv_bar_fraction(key: str, value: float) -> float:
    """Map deployed EISV ranges to a display bar without legacy /2 scaling."""
    magnitude = abs(value) if key == "V" else value
    return min(1.0, max(0.0, magnitude))


def _sparkline_segments(values, x0, width, y_top, height):
    """Points for a sparkline, split into segments at missing (None) values.

    Scaling uses only the values that are present, so a gap neither draws a
    line to 0 nor stretches the range. Runs shorter than two points are
    dropped because a single point cannot be drawn as a line.
    """
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return []
    v_min, v_max = min(present), max(present)
    v_range = v_max - v_min if v_max > v_min else 1.0
    n = len(values)
    segments, current = [], []
    for i, v in enumerate(values):
        if v is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        px = x0 + int(i * width / max(1, n - 1))
        py = y_top + height - int(((v - v_min) / v_range) * height)
        current.append((px, py))
    if len(current) >= 2:
        segments.append(current)
    return segments


class InfoMixin:
    """Mixin for info-group screens (sensors, identity, diagnostics, health)."""

    def _render_sensors(self, readings: Optional[SensorReadings]):
        """Render sensor readings with gauge bars and visual grouping."""
        if not readings:
            self._display.render_text("feeling\nblind", (10, 10), color=COLORS.TEXT_DIM)
            return

        # Cache: sensor values rounded to display precision (avoids redraw for noise)
        throttle = getattr(readings, 'throttle_bits', None) or 0
        # A missing reading keys as "-", not 0, so a sensor dying in a dark
        # room (0 lux) still redraws the screen.
        def _k(value, spec):
            return "-" if value is None else format(value, spec)

        cache_key = (
            f"{_k(readings.ambient_temp_c, '.1f')}|{_k(readings.humidity_pct, '.0f')}|"
            f"{_k(readings.light_lux, '.0f')}|{(readings.cpu_temp_c or 0):.1f}|"
            f"{(readings.cpu_percent or 0):.0f}|{(readings.disk_percent or 0):.0f}|"
            f"{(readings.memory_percent or 0):.0f}|{throttle}"
        )
        if self._check_screen_cache("sensors", cache_key):
            return

        if not hasattr(self._display, '_create_canvas'):
            self._render_sensors_text_fallback(readings)
            return

        try:
            from .design import lighten_color
            image, draw = self._display._create_canvas(COLORS.BG_DARK)
            fonts = self._get_fonts()
            f_title = fonts['title']
            f_small = fonts['small']
            f_tiny  = fonts['tiny']
            f_micro = fonts['micro']

            # ── Palette ────────────────────────────────────────────────────────
            TITLE  = (100, 200, 240)
            MUTED  = ( 88, 105, 125)
            DIV    = ( 30,  42,  62)
            C_OK   = ( 75, 200,  85)
            C_WARM = (215, 160,  45)
            C_HOT  = (200,  80,  80)
            C_COOL = ( 65, 160, 220)

            def _dim(c, f): return tuple(int(x * f) for x in c)

            # ── Layout ─────────────────────────────────────────────────────────
            LINE      = 13
            PAD       = 3
            BAR_H     = 6
            LIST_X    = 13    # text left (after 3px bar)
            BAR_W     = 155   # gauge bar width (13 to 168), sparklines right of 170
            INLINE_W  = 78
            INLINE_H  = 4

            # ── Title ──────────────────────────────────────────────────────────
            draw.text((10, 5), "sensors", fill=TITLE, font=f_title)

            # ── Environment section ─────────────────────────────────────────
            y = 23
            draw.text((LIST_X, y), "environment", fill=MUTED, font=f_tiny)
            draw.line([(84, y + 7), (230, y + 7)], fill=DIV, width=1)
            y += LINE + 2

            # Compute env gauge data
            env_gauges = []

            temp = readings.ambient_temp_c
            if temp is not None:
                if temp > 35:
                    feel, color = "hot", C_HOT
                elif temp > 25:
                    feel, color = "warm", C_WARM
                elif temp < 10:
                    feel, color = "cold", C_COOL
                elif temp < 18:
                    feel, color = "cool", C_COOL
                else:
                    feel, color = "mild", C_OK
                frac = max(0.0, min(1.0, temp / 50.0))
                env_gauges.append(("temp", f"{temp:.1f}\u00b0C", frac, color, feel))
            else:
                env_gauges.append(("temp", "--", 0.0, MUTED, ""))

            hum = readings.humidity_pct
            if hum is not None:
                if hum < 30:
                    feel, color = "dry", C_WARM
                elif hum > 70:
                    feel, color = "damp", C_COOL
                else:
                    feel, color = "ok", C_OK
                frac = max(0.0, min(1.0, hum / 100.0))
                env_gauges.append(("humidity", f"{hum:.0f}%", frac, color, feel))
            else:
                env_gauges.append(("humidity", "--", 0.0, MUTED, ""))

            light = readings.light_lux
            if light is not None:
                # This physical reading mixes room light with the adjacent
                # DotStars. Show it as raw telemetry; without attribution it
                # cannot honestly be called dark, dim, or bright room light.
                feel, color = "room+LED", (150, 180, 200)
                frac = max(0.0, min(1.0, light / 2000.0))
                env_gauges.append(("raw lux", f"{light:.0f}", frac, color, feel))
            else:
                env_gauges.append(("raw lux", "--", 0.0, MUTED, ""))

            # Track bar y positions for sparklines
            spark_bar_ys = []

            for label, val_str, frac, color, feel in env_gauges:
                # Left-edge 3px colored bar spanning this gauge row
                row_h = LINE + BAR_H + PAD
                draw.rectangle([6, y, 9, y + row_h], fill=_dim(color, 0.55) if frac > 0 else _dim(MUTED, 0.3))

                # Label (dim) + value (colored) + feel (right, dim)
                draw.text((LIST_X, y), label,   fill=MUTED,             font=f_small)
                draw.text((   65,  y), val_str, fill=color,             font=f_small)
                if feel:
                    draw.text((200, y), feel, fill=_dim(color, 0.7), font=f_small)
                y += LINE

                # Gauge bar
                draw.rectangle([LIST_X, y, LIST_X + BAR_W, y + BAR_H], fill=(16, 20, 30))
                fw = int(frac * BAR_W)
                if fw > 0:
                    draw.rectangle([LIST_X, y, LIST_X + fw, y + BAR_H], fill=color)
                    if fw > 3:
                        draw.rectangle([LIST_X + fw - 2, y, LIST_X + fw, y + BAR_H],
                                       fill=lighten_color(color, 60))
                spark_bar_ys.append(y)  # record for sparklines
                y += BAR_H + PAD

            if not any([readings.ambient_temp_c, readings.humidity_pct, readings.light_lux]):
                draw.text((LIST_X, y), "I2C off? sudo raspi-config", fill=MUTED, font=f_micro)

            # ── System section ──────────────────────────────────────────────
            y = 132
            draw.text((LIST_X, y), "system", fill=MUTED, font=f_tiny)
            draw.line([(54, y + 7), (230, y + 7)], fill=DIV, width=1)
            y += LINE + 2

            cpu_temp = readings.cpu_temp_c or 0
            cpu_color = C_HOT if cpu_temp > 60 else C_WARM if cpu_temp > 50 else C_OK
            cpu_bar_y = None

            sys_rows = [
                ("cpu",  f"{cpu_temp:.0f}\u00b0C", max(0.0, min(1.0, (cpu_temp - 30) / 55.0)), cpu_color),
                ("load", f"{(readings.cpu_percent or 0):.0f}%",
                         max(0.0, min(1.0, (readings.cpu_percent or 0) / 100.0)), MUTED),
                ("mem",  f"{(readings.memory_percent or 0):.0f}%",
                         max(0.0, min(1.0, (readings.memory_percent or 0) / 100.0)), MUTED),
            ]
            disk = readings.disk_percent or 0
            disk_color = C_HOT if disk > 80 else C_WARM if disk > 60 else C_OK
            sys_rows.append(("disk", f"{disk:.0f}%", max(0.0, min(1.0, disk / 100.0)), disk_color))

            for i, (label, val_str, frac, color) in enumerate(sys_rows):
                draw.text((LIST_X, y), label,   fill=MUTED,  font=f_small)
                draw.text((    48, y), val_str, fill=color,  font=f_small)
                bar_y = y + 5
                draw.rectangle([100, bar_y, 100 + INLINE_W, bar_y + INLINE_H], fill=(16, 20, 30))
                fw = int(frac * INLINE_W)
                if fw > 0:
                    draw.rectangle([100, bar_y, 100 + fw, bar_y + INLINE_H], fill=color)
                if i == 0:  # cpu row — record for sparkline
                    cpu_bar_y = bar_y
                y += LINE + 2

            # ── Footer ─────────────────────────────────────────────────────
            y = 216
            if hasattr(readings, 'undervoltage_now') and readings.undervoltage_now is not None:
                if readings.undervoltage_now:
                    draw.text((10, y), "\u26a1UNDERVOLT!", fill=C_HOT,  font=f_micro)
                elif readings.undervoltage_occurred:
                    draw.text((10, y), "pwr:warn",       fill=C_WARM, font=f_micro)
                elif readings.throttled_now:
                    draw.text((10, y), "pwr:throttle",   fill=C_WARM, font=f_micro)
                else:
                    draw.text((10, y), "pwr:ok",         fill=C_OK,   font=f_micro)

            if readings.pressure_hpa:
                draw.text((70, y), f"{readings.pressure_hpa:.0f}hPa", fill=MUTED, font=f_micro)

            wifi_status = self._get_wifi_status()
            if wifi_status["connected"]:
                ssid    = wifi_status.get("ssid", "")[:8]
                signal  = wifi_status.get("signal", 0)
                ip      = wifi_status.get("ip", "")
                wc = C_OK if signal > 70 else C_WARM if signal > 40 else (200, 130, 60)
                draw.text((130, y), f"{ssid} {signal}%", fill=wc,   font=f_micro)
                if ip:
                    draw.text((10, y + 11), f"ip: {ip}", fill=MUTED, font=f_micro)
            else:
                draw.text((130, y), "no wifi", fill=C_HOT, font=f_micro)

            # ── Sparklines (right of env bars) ─────────────────────────────
            # Missing readings stay None and leave a gap in the sparkline,
            # rather than drawing a plunge to 0 that looks like a real value.
            self._sensor_history.append((
                readings.ambient_temp_c,
                readings.humidity_pct,
                readings.cpu_temp_c,
            ))
            if len(self._sensor_history) >= 5 and len(spark_bar_ys) >= 2:
                history = list(self._sensor_history)
                spark_x = 172
                spark_w = 56
                spark_h = 8
                sparklines = [
                    (0, spark_bar_ys[0], cpu_color if (readings.ambient_temp_c is not None and readings.ambient_temp_c > 25) else C_OK),
                    (1, spark_bar_ys[1], C_OK),
                ]
                if cpu_bar_y is not None:
                    sparklines.append((2, cpu_bar_y - 1, cpu_color))
                for data_idx, y_top, color in sparklines:
                    values = [h[data_idx] for h in history]
                    for seg in _sparkline_segments(values, spark_x, spark_w, y_top, spark_h):
                        draw.line(seg, fill=_dim(color, 0.7), width=1)

            self._draw_status_bar(draw)
            self._store_screen_cache("sensors", cache_key, image)
            if hasattr(self._display, '_image'):
                self._display._image = image
            if hasattr(self._display, '_show'):
                self._display._show()
            return

        except Exception as e:
            import traceback
            print(f"[Sensors Screen] Canvas error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

        self._render_sensors_text_fallback(readings)

    def _render_sensors_text_fallback(self, readings: SensorReadings):
        """Text-only fallback for sensors screen."""
        lines_with_colors = []
        if not any([readings.ambient_temp_c, readings.humidity_pct, readings.light_lux]):
            lines_with_colors.append(("I2C off? sudo raspi-config", COLORS.TEXT_DIM))
        if readings.ambient_temp_c:
            temp = readings.ambient_temp_c
            tc = COLORS.SOFT_ORANGE if temp > 25 else COLORS.SOFT_CYAN if temp < 18 else COLORS.SOFT_GREEN
            lines_with_colors.append((f"air: {temp:.1f}\u00b0C", tc))
        if readings.humidity_pct:
            hum = readings.humidity_pct
            hc = COLORS.SOFT_YELLOW if hum < 30 else COLORS.SOFT_BLUE if hum > 70 else COLORS.SOFT_GREEN
            lines_with_colors.append((f"humidity: {hum:.0f}%", hc))
        if readings.light_lux:
            lines_with_colors.append((f"raw lux: {readings.light_lux:.0f} (room+LED)", COLORS.TEXT_PRIMARY))
        cpu_t = readings.cpu_temp_c or 0
        lines_with_colors.append((f"cpu: {cpu_t:.1f}\u00b0C", COLORS.SOFT_GREEN))
        if getattr(readings, 'undervoltage_now', None):
            lines_with_colors.append(("UNDERVOLT!", COLORS.SOFT_CORAL))
        elif getattr(readings, 'undervoltage_occurred', None):
            lines_with_colors.append(("pwr: warn", COLORS.SOFT_ORANGE))
        if lines_with_colors:
            if hasattr(self._display, 'render_colored_text'):
                self._display.render_colored_text(lines_with_colors, (10, 10))
            else:
                self._display.render_text("\n".join(label for label, _ in lines_with_colors), (10, 10))
        else:
            self._display.render_text("sensors\n\nno data", (10, 10), color=COLORS.TEXT_DIM)

    def _draw_sensor_sparklines(self, draw, readings):
        """Legacy sparkline helper — kept for compatibility, draw inline in _render_sensors now."""
        pass

    def _render_identity(self, identity: Optional[CreatureIdentity]):
        """Render identity screen."""
        if not identity:
            self._display.render_text("who am i?\n(unknown)", (10, 10), color=COLORS.TEXT_DIM)
            return

        age_days    = identity.age_seconds() / 86400
        alive_hours = identity.total_alive_seconds / 3600
        alive_pct   = identity.alive_ratio() * 100
        name        = identity.name or "unnamed"

        id_key = (
            f"{name}|{age_days:.1f}|{alive_hours:.1f}|{alive_pct:.0f}|"
            f"{identity.total_awakenings}|{self._unitares_agent_id or ''}"
        )
        if self._check_screen_cache("identity", id_key):
            return

        if not hasattr(self._display, '_create_canvas'):
            self._render_identity_text_fallback(identity, age_days, alive_hours)
            return

        try:
            image, draw = self._display._create_canvas(COLORS.BG_DARK)
            fonts   = self._get_fonts()
            f_title = fonts['title']
            f_med   = fonts['medium']
            f_small = fonts['small']
            f_tiny  = fonts['tiny']

            # ── Palette ────────────────────────────────────────────────────
            TITLE    = (100, 200, 240)
            MUTED    = ( 88, 105, 125)
            DIV      = ( 30,  42,  62)
            C_CYAN   = COLORS.SOFT_CYAN
            C_BLUE   = COLORS.SOFT_BLUE
            C_YELLOW = COLORS.SOFT_YELLOW
            C_ORANGE = COLORS.SOFT_ORANGE
            C_PURPLE = COLORS.SOFT_PURPLE
            C_GREEN  = COLORS.SOFT_GREEN

            LINE = 15  # row height

            # ── Title ──────────────────────────────────────────────────────
            draw.text((10, 5), f"i am {name}", fill=TITLE, font=f_title)
            draw.line([(10, 22), (230, 22)], fill=DIV, width=1)

            # ── Stats (left column, x=13–130) ──────────────────────────────
            # Ring on right: cx=190, r=44 → x=146–234, y=68–156. No overlap.
            y = 28

            # Age
            if age_days < 1:
                age_str, age_color = f"{age_days * 24:.1f} hours old", C_PURPLE
            elif age_days < 7:
                age_str, age_color = f"{age_days:.1f} days old", C_BLUE
            else:
                age_str, age_color = f"{age_days:.0f} days old", C_CYAN
            draw.rectangle([6, y, 9, y + LINE], fill=age_color)
            draw.text((13, y), age_str, fill=age_color, font=f_med)
            y += LINE + 2

            # Awakenings
            if identity.total_awakenings == 1:
                wake_str, wake_color = "first awakening", C_ORANGE
            else:
                wake_str, wake_color = f"awakened {identity.total_awakenings}\u00d7", C_PURPLE
            draw.rectangle([6, y, 9, y + LINE], fill=wake_color)
            draw.text((13, y), wake_str, fill=wake_color, font=f_med)
            y += LINE + 2

            # Awake time
            awake_str = f"awake {alive_hours:.1f}h" if alive_hours < 24 else f"awake {alive_hours/24:.1f}d"
            draw.rectangle([6, y, 9, y + LINE], fill=C_YELLOW)
            draw.text((13, y), awake_str, fill=C_YELLOW, font=f_med)
            y += LINE + 2

            # Presence descriptor (no bar, lighter register)
            if alive_pct > 80:
                pres_str, pres_color = "mostly here",    C_GREEN
            elif alive_pct > 50:
                pres_str, pres_color = "sometimes here", C_YELLOW
            else:
                pres_str, pres_color = "often away",     COLORS.TEXT_SECONDARY
            draw.text((16, y), f"({pres_str})", fill=pres_color, font=f_small)

            # ── Alive ratio ring (right side, hero element) ─────────────────
            # Large ring — alive ratio is the defining stat of this screen.
            # Position: cx=190, cy=112, r=44 → entirely right of x=146.
            try:
                from .design import wellness_to_color
                alive_ratio = identity.alive_ratio()
                ring_cx, ring_cy, ring_r = 190, 112, 44
                ring_bbox = [ring_cx - ring_r, ring_cy - ring_r,
                             ring_cx + ring_r, ring_cy + ring_r]

                # Track (background)
                draw.arc(ring_bbox, 0, 360, fill=(28, 34, 48), width=10)
                # Fill arc (clockwise from top)
                ring_color = wellness_to_color(alive_ratio) if alive_ratio > 0.01 else MUTED
                if alive_ratio > 0.01:
                    draw.arc(ring_bbox, -90, -90 + int(alive_ratio * 360),
                             fill=ring_color, width=10)

                # Percentage inside ring (f_title for prominence)
                pct_text = f"{alive_pct:.0f}%"
                try:
                    bbox = f_title.getbbox(pct_text)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = len(pct_text) * 7, 13
                draw.text((ring_cx - tw // 2, ring_cy - th // 2 - 1),
                          pct_text, fill=COLORS.TEXT_PRIMARY, font=f_title)

                # "present" label below the % (f_tiny, muted)
                try:
                    sub_bbox = f_tiny.getbbox("present")
                    sw = sub_bbox[2] - sub_bbox[0]
                except Exception:
                    sw = 34
                draw.text((ring_cx - sw // 2, ring_cy + th // 2 + 3),
                          "present", fill=MUTED, font=f_tiny)

            except Exception:
                pass

            # ── IDs (below ring) ───────────────────────────────────────────
            y_ids = 164
            draw.line([(10, y_ids - 4), (230, y_ids - 4)], fill=DIV, width=1)
            short_id = identity.creature_id[:8] if identity.creature_id else "unknown"
            draw.text((13, y_ids), f"id: {short_id}", fill=MUTED, font=f_small)
            if self._unitares_agent_id:
                draw.text((130, y_ids), f"gov: {self._unitares_agent_id[:8]}", fill=C_GREEN, font=f_small)

            self._draw_status_bar(draw)
            self._store_screen_cache("identity", id_key, image)
            if hasattr(self._display, '_image'):
                self._display._image = image
            if hasattr(self._display, '_show'):
                self._display._show()
            return
        except Exception as e:
            print(f"[Identity Screen] Canvas error: {e}", file=sys.stderr, flush=True)

        self._render_identity_text_fallback(identity, age_days, alive_hours)

    def _render_identity_text_fallback(self, identity, age_days, alive_hours):
        lines_with_colors = [(f"i am {identity.name or 'unnamed'}", COLORS.SOFT_CYAN), ("", COLORS.TEXT_PRIMARY)]
        if age_days < 1:
            lines_with_colors.append((f"{age_days * 24:.1f} hours old", COLORS.SOFT_PURPLE))
        elif age_days < 7:
            lines_with_colors.append((f"{age_days:.1f} days old", COLORS.SOFT_BLUE))
        else:
            lines_with_colors.append((f"{age_days:.0f} days old", COLORS.SOFT_CYAN))
        if alive_hours < 24:
            lines_with_colors.append((f"awake {alive_hours:.1f}h", COLORS.SOFT_YELLOW))
        else:
            lines_with_colors.append((f"awake {alive_hours/24:.1f}d", COLORS.SOFT_YELLOW))
        if identity.total_awakenings == 1:
            lines_with_colors.append(("first awakening", COLORS.SOFT_ORANGE))
        else:
            lines_with_colors.append((f"awakened {identity.total_awakenings}x", COLORS.SOFT_PURPLE))
        if hasattr(self._display, 'render_colored_text'):
            self._display.render_colored_text(lines_with_colors, (10, 10))
        else:
            self._display.render_text("\n".join(label for label, _ in lines_with_colors), (10, 10))

    def _render_diagnostics(self, anima: Optional[Anima], readings: Optional[SensorReadings], governance: Optional[Dict[str, Any]]):
        """Render diagnostics screen — anima state, mood, governance, trajectory."""
        if not anima:
            self._display.render_text("diagnostics\n\nno data", (10, 10))
            return

        gov_state = (governance.get("unitares_agent_id") or "")[:8] if governance else ""
        displayed_eisv, eisv_label = _select_eisv_for_display(governance)
        vector_display_key = (
            tuple(f"{displayed_eisv[key]:.3f}" for key in _EISV_KEYS)
            if displayed_eisv
            else ()
        )
        gov_display_key = "|".join(
            [
                str(governance.get("action", "")) if governance else "",
                str(governance.get("margin", "")) if governance else "",
                str(governance.get("source", "")) if governance else "",
                eisv_label,
                *vector_display_key,
            ]
        )
        try:
            from ..eisv import get_trajectory_awareness
            _traj_shape = get_trajectory_awareness().current_shape or ""
        except Exception:
            _traj_shape = ""
        diag_key = (
            f"{anima.warmth:.2f}|{anima.clarity:.2f}|{anima.stability:.2f}|"
            f"{anima.presence:.2f}|{gov_state}|{gov_display_key}|{_traj_shape}"
        )
        if self._check_screen_cache("diagnostics", diag_key):
            return

        try:
            if not hasattr(self._display, '_create_canvas'):
                self._render_diagnostics_text_fallback(anima, governance)
                return

            image, draw = self._display._create_canvas(COLORS.BG_DARK)
            fonts     = self._get_fonts()
            f_title   = fonts['title']
            f_med     = fonts['medium']
            f_small   = fonts['small']
            f_tiny    = fonts['tiny']

            # ── Palette ────────────────────────────────────────────────────
            TITLE  = (100, 200, 240)
            MUTED  = ( 88, 105, 125)
            DIV    = ( 30,  42,  62)
            BG     = ( 16,  20,  30)
            # Anima dimension colors (consistent with msgs palette)
            C_WARM = (215, 160,  45)   # warmth — amber
            C_CLAR = ( 75, 200,  85)   # clarity — lime
            C_STAB = ( 65, 160, 220)   # stability — cool blue
            C_PRES = (140,  90, 210)   # presence — violet
            C_OK   = ( 75, 200,  85)
            C_WARN = (215, 160,  45)
            C_HOT  = (200,  80,  80)

            def _dim(c, f): return tuple(int(x * f) for x in c)

            LINE   = 13
            BAR_H  = 7
            PAD    = 3
            LIST_X = 13
            BAR_W  = 162   # wider — fills the display properly

            # ── Title ──────────────────────────────────────────────────────
            draw.text((10, 5), "diagnostics", fill=TITLE, font=f_title)
            y = 23
            draw.line([(10, y), (230, y)], fill=DIV, width=1)
            y += 4

            # ── Anima dimensions ───────────────────────────────────────────
            dims = [
                ("warmth",    anima.warmth,    C_WARM,
                 "warm" if anima.warmth > 0.6 else "cold" if anima.warmth < 0.3 else "cool" if anima.warmth < 0.5 else "ok"),
                ("clarity",   anima.clarity,   C_CLAR,
                 "clear" if anima.clarity > 0.7 else "foggy" if anima.clarity < 0.5 else "mixed"),
                ("stability", anima.stability, C_STAB,
                 "steady" if anima.stability > 0.7 else "shaky" if anima.stability < 0.5 else "ok"),
                ("presence",  anima.presence,  C_PRES,
                 "here" if anima.presence > 0.7 else "distant" if anima.presence < 0.5 else "near"),
            ]

            from .design import lighten_color
            for label, value, color, desc in dims:
                row_h = LINE + BAR_H + PAD
                draw.rectangle([6, y, 9, y + row_h], fill=_dim(color, 0.55))

                # Label + descriptor (bar + right-side value convey the number)
                draw.text((LIST_X, y), label, fill=MUTED, font=f_small)
                draw.text((73, y), desc, fill=_dim(color, 0.7), font=f_small)
                y += LINE

                # Bar with value label right of end
                draw.rectangle([LIST_X, y, LIST_X + BAR_W, y + BAR_H], fill=BG)
                fw = int(value * BAR_W)
                if fw > 0:
                    draw.rectangle([LIST_X, y, LIST_X + fw, y + BAR_H], fill=color)
                    if fw > 4:
                        draw.rectangle([LIST_X + fw - 2, y, LIST_X + fw, y + BAR_H],
                                       fill=lighten_color(color, 50))
                draw.text((LIST_X + BAR_W + 4, y - 1), f"{value:.0%}", fill=color, font=f_tiny)
                y += BAR_H + PAD

            # ── Mood ───────────────────────────────────────────────────────
            y += 2
            feeling = anima.feeling()
            mood = feeling.get('mood', 'unknown')
            mood_low = mood.lower()
            if "content" in mood_low or "happy" in mood_low or "calm" in mood_low:
                mood_color = C_OK
            elif "stressed" in mood_low or "anxious" in mood_low:
                mood_color = C_HOT
            elif "neutral" in mood_low:
                mood_color = MUTED
            elif "alert" in mood_low or "focused" in mood_low:
                mood_color = C_WARM
            else:
                mood_color = COLORS.TEXT_SECONDARY
            draw.text((LIST_X, y), f"mood  {mood}", fill=mood_color, font=f_med)
            y += 20

            draw.line([(10, y), (230, y)], fill=DIV, width=1)
            y += 4

            # ── Governance ─────────────────────────────────────────────────
            if governance:
                action = governance.get("action", "unknown")
                margin = governance.get("margin", "")
                source = governance.get("source", "")
                action_colors = {
                    "proceed": C_OK,
                    "guide":   C_WARN,
                    "pause":   (200, 130, 60),
                    "halt":    C_HOT,
                    "reject":  C_HOT,
                }
                a_color     = action_colors.get(action, MUTED)
                is_unitares = "unitares" in (source or "").lower()
                src_color   = C_OK if is_unitares else C_WARN
                src_label   = "UNITARES" if is_unitares else "local"

                # Left-edge bar spanning action row
                draw.rectangle([6, y, 9, y + LINE], fill=_dim(a_color, 0.6))

                # Verdict word (large)
                draw.text((LIST_X, y), action, fill=a_color, font=f_med)

                # Margin (center, small)
                if margin:
                    margin_colors = {
                        "comfortable": C_OK,
                        "tight":       C_WARN,
                        "warning":     (200, 130, 60),
                        "critical":    C_HOT,
                    }
                    m_color = margin_colors.get(margin.lower(), MUTED)
                    draw.text((83, y + 2), margin.lower(), fill=m_color, font=f_tiny)

                # Source connection — right side, fits within 230px
                draw.text((175, y + 2), src_label, fill=src_color, font=f_tiny)
                y += LINE + 2

                # Prefer UNITARES's inferred vector. If the minimal response
                # omitted it, label the submitted body projection explicitly.
                if displayed_eisv and y < 210:
                    draw.text((LIST_X, y), eisv_label, fill=MUTED, font=f_tiny)
                    y += 9
                    v_raw   = displayed_eisv.get("V", 0.0)
                    v_color = C_WARM if v_raw > 0.05 else C_STAB if v_raw < -0.05 else MUTED
                    sign_ch = "+" if v_raw > 0.05 else "\u2212" if v_raw < -0.05 else "~"

                    # [label, bar_frac (0-1), color, val_str]
                    eisv_rows = [
                        [("E", _eisv_bar_fraction("E", displayed_eisv["E"]), C_OK,
                          f"{displayed_eisv.get('E', 0.0):.0%}"),
                         ("I", _eisv_bar_fraction("I", displayed_eisv["I"]), C_STAB,
                          f"{displayed_eisv.get('I', 0.0):.0%}")],
                        [("S", _eisv_bar_fraction("S", displayed_eisv["S"]), C_WARN,
                          f"{displayed_eisv.get('S', 0.0):.2f}"),
                         ("V", _eisv_bar_fraction("V", v_raw), v_color,
                          f"{sign_ch}{abs(v_raw):.2f}")],
                    ]
                    mb_w, mb_h = 52, 6
                    COL_W = 112
                    for row_items in eisv_rows:
                        for ci, (lbl, frac, color, val_str) in enumerate(row_items):
                            x0 = LIST_X + ci * COL_W
                            draw.text((x0, y), lbl, fill=color, font=f_small)
                            bx = x0 + 13
                            draw.rectangle([bx, y + 3, bx + mb_w, y + 3 + mb_h], fill=BG)
                            fw = int(frac * mb_w)
                            if fw > 0:
                                draw.rectangle([bx, y + 3, bx + fw, y + 3 + mb_h],
                                               fill=_dim(color, 0.8))
                            draw.text((bx + mb_w + 3, y + 2), val_str,
                                      fill=_dim(color, 0.85), font=f_tiny)
                        y += 11
                    y += 2
            else:
                draw.text((LIST_X, y), "gov: waiting…", fill=MUTED, font=f_small)
                y += LINE + 2

            # ── Trajectory ─────────────────────────────────────────────────
            if y < 228:
                try:
                    from ..eisv import get_trajectory_awareness
                    _traj = get_trajectory_awareness()
                    _shape = _traj.current_shape or "…"
                    _buf   = f"{_traj.buffer_size}/{_traj._buffer.maxlen}"
                    draw.text((LIST_X, y), f"traj: {_shape}  ({_buf})", fill=MUTED, font=f_tiny)
                except Exception:
                    pass

            self._draw_status_bar(draw)
            self._store_screen_cache("diagnostics", diag_key, image)
            if hasattr(self._display, '_image'):
                self._display._image = image
            if hasattr(self._display, '_show'):
                self._display._show()

        except Exception as e:
            import traceback
            print(f"[Diagnostics Screen] Error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._render_diagnostics_text_fallback(anima, governance)

    def _render_diagnostics_text_fallback(self, anima: Anima, governance: Optional[Dict[str, Any]]):
        """Text-only fallback for diagnostics screen."""
        lines = [
            f"warmth: {anima.warmth:.0%}",
            f"clarity: {anima.clarity:.0%}",
            f"stability: {anima.stability:.0%}",
            f"presence: {anima.presence:.0%}",
            f"mood: {anima.feeling().get('mood', 'unknown')}",
        ]
        if governance:
            action = governance.get('action', '?')
            margin = governance.get('margin', '')
            source = governance.get('source', '')
            lines.append(f"gov: {action}")
            if margin:
                lines.append(f"margin: {margin}")
            if source:
                lines.append(f"source: {source}")
            eisv, display_label = _select_eisv_for_display(governance)
            if display_label == "gov state":
                label = "gov EISV"
            else:
                label = "body projection"
            if eisv:
                lines.append(
                    f"{label}: E={eisv.get('E', 0):.0%} "
                    f"I={eisv.get('I', 0):.0%} S={eisv.get('S', 0):.0%} "
                    f"V={eisv.get('V', 0):+.0%}"
                )
        try:
            from ..eisv import get_trajectory_awareness
            _traj = get_trajectory_awareness()
            if _traj.current_shape:
                lines.append(f"traj: {_traj.current_shape}")
        except Exception:
            pass
        self._display.render_text("\n".join(lines), (10, 10))

    def _render_health(self):
        """Render subsystem health monitoring screen.

        One row per subsystem with left-edge colored status bar.
        Status: ok=green, stale=yellow, degraded=orange, missing=red.
        """
        try:
            from ..health import get_health_registry
            registry = get_health_registry()
        except Exception:
            self._display.render_text("health\n\nno registry", (10, 10))
            return

        status_data = registry.status()
        overall     = registry.overall()

        health_key = "|".join(f"{n}:{d.get('status','?')}" for n, d in sorted(status_data.items()))
        if self._check_screen_cache("health", health_key):
            return

        try:
            if not hasattr(self._display, '_create_canvas'):
                lines = ["health", f"overall: {overall}", ""]
                for name, info in sorted(status_data.items()):
                    s = info.get("status", "?")
                    hb = info.get("last_heartbeat_ago_s")
                    hb_str = f" {hb:.0f}s" if hb is not None else ""
                    lines.append(f"  {s[0].upper()} {name}{hb_str}")
                self._display.render_text("\n".join(lines), (10, 10))
                return

            image, draw = self._display._create_canvas(COLORS.BG_DARK)
            fonts     = self._get_fonts()
            f_title   = fonts['title']
            f_small   = fonts['small']
            f_tiny    = fonts['tiny']

            # ── Palette ────────────────────────────────────────────────────
            TITLE  = (100, 200, 240)
            MUTED  = ( 88, 105, 125)
            DIV    = ( 30,  42,  62)

            STATUS_COLORS = {
                "ok":       ( 75, 200,  85),
                "stale":    (215, 160,  45),
                "degraded": (200, 130,  60),
                "missing":  (200,  80,  80),
                "absent":   MUTED,
                "unknown":  MUTED,
            }
            OVERALL_COLORS = {
                "ok":       ( 75, 200,  85),
                "degraded": (215, 160,  45),
                "unhealthy":(200,  80,  80),
                "unknown":  MUTED,
            }

            LINE = 13
            PAD  = 3

            # ── Header ─────────────────────────────────────────────────────
            draw.text((10, 5), "health", fill=TITLE, font=f_title)
            oc = OVERALL_COLORS.get(overall, MUTED)
            draw.text((80, 7), overall, fill=oc, font=f_small)
            y = 23
            draw.line([(10, y), (230, y)], fill=DIV, width=1)
            y += 4

            # ── Subsystem rows ─────────────────────────────────────────────
            subsystems = sorted(status_data.items())
            n_rows = max(1, len(subsystems))
            row_h  = min(20, max(LINE + PAD, (224 - y) // n_rows))

            for name, info in subsystems:
                status = info.get("status", "unknown")
                color  = STATUS_COLORS.get(status, MUTED)
                hb_ago = info.get("last_heartbeat_ago_s")
                probe  = info.get("probe")

                # Left-edge 3px status bar (same as msgs rows)
                draw.rectangle([6, y, 9, y + row_h - 1], fill=color)

                # Subsystem name
                draw.text((13, y + 2), name, fill=COLORS.TEXT_PRIMARY, font=f_small)

                # Heartbeat age (right-aligned area)
                if hb_ago is not None:
                    age_str = f"{hb_ago:.0f}s" if hb_ago < 60 else f"{hb_ago/60:.0f}m"
                    draw.text((155, y + 2), age_str, fill=MUTED, font=f_tiny)

                # Probe status
                if probe is not None:
                    p_color = ( 75, 200, 85) if probe == "ok" else (200, 80, 80)
                    draw.text((195, y + 2), probe[:12], fill=p_color, font=f_tiny)

                y += row_h

            self._draw_status_bar(draw)
            self._display.render_image(image)
            self._store_screen_cache("health", health_key, image)

        except Exception as e:
            print(f"[Screen] Health render error: {e}", file=sys.stderr, flush=True)
            self._display.render_text(f"health\n\n{overall}\n\nerror:\n{str(e)[:40]}", (10, 10))
