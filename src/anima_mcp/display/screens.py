"""
Display Screens - Different views for Lumen's display.

Screens can be toggled via joystick:
- Face (default): Lumen's expressive face
- Sensors: Current sensor readings
- Identity: Name, age, awakenings, alive time
- Diagnostics: System health, governance status
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import sys
import math

from .face import FaceState
from .design import Timing, ease_smooth
from ..anima import Anima
from ..sensors.base import SensorReadings
from ..identity.store import CreatureIdentity
from ..learning_visualization import LearningVisualizer
from .drawing_engine import (
    DrawingEngine,
    DrawingEISV as DrawingEISV,
    DrawingIntent as DrawingIntent,
    _EISV_PARAMS as _EISV_PARAMS,
)
from .screen_home import HomeMixin
from .screen_info import InfoMixin
from .screen_mind import MindMixin
from .screen_messages import MessagesMixin
from .screen_art import ArtMixin


class ScreenMode(Enum):
    """Available display screens."""
    FACE = "face"                    # Default: Lumen's expressive face
    SENSORS = "sensors"              # Sensor readings (temp, humidity, etc.)
    IDENTITY = "identity"            # Name, age, awakenings, alive time
    DIAGNOSTICS = "diagnostics"      # System health, governance status
    NEURAL = "neural"                # Neural activity - computational EEG bands
    INNER_LIFE = "inner_life"        # Inner life - cognitive and emotional state
    LEARNING = "learning"            # Learning visualization - why Lumen feels what it feels
    SELF_GRAPH = "self_graph"        # Self-schema G_t visualization
    GOALS_BELIEFS = "goals_beliefs"  # Goals and self-beliefs
    AGENCY = "agency"                # Action selection and TD-learning
    NOTEPAD = "notepad"              # Drawing canvas - Lumen's creative space
    MESSAGES = "messages"            # Message board - Lumen's observations only
    QUESTIONS = "questions"          # Q&A - Lumen's questions and answers
    VISITORS = "visitors"            # Messages from agents and humans
    ART_ERAS = "art_eras"            # Art era history and current era
    HEALTH = "health"                # Subsystem health monitoring


@dataclass
class ScreenState:
    """Current screen state."""
    mode: ScreenMode = ScreenMode.FACE
    last_switch_time: float = 0.0
    last_user_action_time: float = 0.0  # Track when user last interacted

    # Message board interaction state
    message_scroll_index: int = 0  # Which message is currently selected/visible
    message_expanded_id: Optional[str] = None  # Message ID that is expanded (showing full text)
    message_text_scroll: int = 0  # Line offset when scrolling within expanded message text

    # Q&A screen interaction state
    qa_scroll_index: int = 0  # Which Q&A pair is selected
    qa_expanded: bool = False  # Whether current Q&A is expanded
    qa_focus: str = "question"  # "question" or "answer" - which part is focused when expanded
    qa_text_scroll: int = 0  # Line offset when scrolling within focused text
    qa_full_view: bool = False  # Full-screen view for answer (maximum readability)

    # Screen transition state (fade effect)
    transition_progress: float = 1.0  # 0.0 = start, 1.0 = complete
    transition_start_time: float = 0.0
    transition_duration: float = Timing.SCREEN_TRANSITION_MS / 1000.0
    previous_image: Optional[Any] = None  # PIL Image of previous screen

    # Loading state (spinner during LLM calls)
    is_loading: bool = False
    loading_message: str = ""
    loading_start_time: float = 0.0

    # Status bar state
    wifi_connected: bool = True
    governance_connected: bool = False

    # Input feedback state (visual acknowledgment of joystick/button)
    input_feedback_until: float = 0.0  # Show feedback until this time
    input_feedback_direction: str = ""  # "left", "right", "up", "down", "press"

    # Art eras screen interaction state
    era_cursor: int = 0  # Which era is highlighted (index into list_all_era_info)
    era_marquee_offset: int = 0  # Pixel offset for marquee scrolling description
    era_marquee_time: float = 0.0  # Last marquee tick time

    # Brightness overlay state
    brightness_changed_at: float = 0.0  # When brightness last changed
    brightness_overlay_name: str = ""  # Preset name to display
    brightness_overlay_level: float = 1.0  # Display brightness level for bar
    controls_overlay_until: float = 0.0  # Show controls overlay until this timestamp

    # Governance verdict enforcement: pause drawing when governance says pause/halt
    governance_paused: bool = False  # True when action in ("pause", "halt")


class ScreenRenderer(HomeMixin, InfoMixin, MindMixin, MessagesMixin, ArtMixin):
    """Renders different screens to display."""

    # Pre-compiled keyword sets for mood coloring (avoid recreating on each render)
    _FEELING_WORDS = frozenset(['feel', 'warm', 'comfort', 'content', 'happy', 'joy'])
    _CURIOSITY_WORDS = frozenset(['wonder', 'curious', 'think', 'notice', 'observe'])
    _GROWTH_WORDS = frozenset(['learn', 'grow', 'new', 'discover', 'understand'])
    _CALM_WORDS = frozenset(['quiet', 'rest', 'peace', 'calm', 'still'])

    # Thread lock to prevent concurrent renders causing display corruption
    import threading
    _render_lock = threading.Lock()

    def __init__(self, display_renderer, db_path: Optional[str] = None, identity_store=None):
        """Initialize with display renderer."""
        self._display = display_renderer
        self._state = ScreenState()
        self._db_path = db_path or "anima.db"
        self._identity_store = identity_store

        # Drawing engine owns canvas, intent, era, mood tracker
        self.drawing_engine = DrawingEngine(db_path=self._db_path, identity_store=identity_store)

        # Initialize user action time
        self._state.last_user_action_time = time.time()
        # Cache for learning screen (DB queries are slow - 20+ seconds)
        self._learning_visualizer: Optional[LearningVisualizer] = None
        self._learning_cache: Optional[Dict[str, Any]] = None
        self._learning_cache_time: float = 0.0
        self._learning_cache_ttl: float = 60.0  # Refresh every 60 seconds (data changes slowly)
        self._learning_cache_refreshing: bool = False  # Prevent concurrent refreshes
        # Font cache (font loading from disk is slow - adds ~500ms per render)
        self._fonts: Optional[Dict[str, Any]] = None
        # Text measurement cache (avoid creating PIL Image on every wrap call)
        self._measure_draw: Optional[Any] = None
        # Message screen image cache (text rendering is slow - ~500ms)
        self._messages_cache_image: Optional[Any] = None
        self._messages_cache_hash: str = ""  # Hash of messages + scroll state
        # Generic screen image cache: {screen_name: (hash_str, PIL.Image)}, max 12 entries
        self._screen_cache: Dict[str, tuple] = {}
        self._screen_cache_max_size = 12
        self._screen_cache_order: List[str] = []  # LRU order
        # UNITARES agent_id (for display on identity screen)
        self._unitares_agent_id: Optional[str] = None
        # Shared memory data (set by server.py before render())
        self._shm_data: Optional[Dict[str, Any]] = None
        # Sensor sparkline history
        from collections import deque
        self._sensor_history: deque = deque(maxlen=40)

    # --- Backward-compat properties for drawing engine internals ---
    @property
    def _canvas(self):
        return self.drawing_engine.canvas

    @property
    def _intent(self):
        return self.drawing_engine.intent

    @property
    def _active_era(self):
        return self.drawing_engine.active_era

    @_active_era.setter
    def _active_era(self, value):
        self.drawing_engine.active_era = value

    @property
    def _drawing_goal(self):
        return self.drawing_engine.drawing_goal

    @_drawing_goal.setter
    def _drawing_goal(self, value):
        self.drawing_engine.drawing_goal = value

    @property
    def _last_anima(self):
        return self.drawing_engine.last_anima

    @_last_anima.setter
    def _last_anima(self, value):
        self.drawing_engine.last_anima = value

    @property
    def _mood_tracker(self):
        return self.drawing_engine._mood_tracker

    # --- Pass-through methods for server.py callers ---
    def canvas_save(self, **kw):
        return self.drawing_engine.canvas_save(**kw)

    def canvas_clear(self, **kw):
        return self.drawing_engine.canvas_clear(**kw)

    def get_drawing_eisv(self):
        return self.drawing_engine.get_drawing_eisv()

    def get_current_era(self):
        return self.drawing_engine.get_current_era()

    def set_era(self, era_name, force_immediate=False):
        return self.drawing_engine.set_era(era_name, force_immediate)

    def era_cursor_up(self):
        return self.drawing_engine.era_cursor_up(self._state)

    def era_cursor_down(self):
        return self.drawing_engine.era_cursor_down(self._state)

    def era_select_current(self):
        return self.drawing_engine.era_select_current(self._state)

    def canvas_check_autonomy(self, anima=None):
        return self.drawing_engine.canvas_check_autonomy(anima)

    def _lumen_draw(self, anima, draw=None):
        return self.drawing_engine.draw(anima, draw)

    def _get_messages_cache_hash(self, messages: list, scroll_idx: int, expanded_id: Optional[str]) -> str:
        """Compute hash of message screen state for cache invalidation."""
        # Include message IDs/timestamps, scroll position, and expanded state
        msg_ids = "|".join(f"{m.message_id}:{m.timestamp}" for m in messages[:10]) if messages else ""
        return f"{msg_ids}|{scroll_idx}|{expanded_id or ''}"

    def _check_screen_cache(self, screen_name: str, cache_key: str) -> bool:
        """Check if cached image matches current state. If hit, apply it.

        Returns True if cache hit (caller should return immediately).
        Uses copy() so post-processing (overlays, transitions) never mutates the cache.
        """
        entry = self._screen_cache.get(screen_name)
        if entry and entry[0] == cache_key:
            if hasattr(self._display, '_image'):
                self._display._image = entry[1].copy()
            if hasattr(self._display, '_show'):
                self._display._show()
            # Bump to end of LRU order (recently used)
            if screen_name in self._screen_cache_order:
                self._screen_cache_order.remove(screen_name)
            self._screen_cache_order.append(screen_name)
            return True
        return False

    def _store_screen_cache(self, screen_name: str, cache_key: str, image):
        """Store rendered image in screen cache. Evict oldest when over max size."""
        if screen_name in self._screen_cache:
            self._screen_cache_order.remove(screen_name)
        self._screen_cache[screen_name] = (cache_key, image.copy())
        self._screen_cache_order.append(screen_name)
        while len(self._screen_cache) > self._screen_cache_max_size and self._screen_cache_order:
            evict = self._screen_cache_order.pop(0)
            if evict in self._screen_cache:
                del self._screen_cache[evict]

    def _get_measure_draw(self):
        """Get cached draw context for text measurement."""
        if self._measure_draw is None:
            from PIL import ImageDraw, Image
            temp_img = Image.new('RGB', (1, 1))
            self._measure_draw = ImageDraw.Draw(temp_img)
        return self._measure_draw

    def _get_fonts(self) -> Dict[str, Any]:
        """Get cached fonts (loads from disk only once)."""
        if self._fonts is None:
            try:
                from PIL import ImageFont
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                self._fonts = {
                    'micro': ImageFont.truetype(font_path, 9),
                    'tiny': ImageFont.truetype(font_path, 10),
                    'small': ImageFont.truetype(font_path, 11),
                    'small_med': ImageFont.truetype(font_path, 12),
                    'medium': ImageFont.truetype(font_path, 13),
                    'default': ImageFont.truetype(font_path, 14),
                    'large': ImageFont.truetype(font_path, 15),
                    'title': ImageFont.truetype(font_path, 16),
                    'huge': ImageFont.truetype(font_path, 18),
                    'giant': ImageFont.truetype(font_path, 20),
                }
            except (OSError, IOError):
                from PIL import ImageFont
                fallback = ImageFont.load_default()
                self._fonts = {
                    'micro': fallback,
                    'tiny': fallback,
                    'small': fallback,
                    'small_med': fallback,
                    'medium': fallback,
                    'default': fallback,
                    'large': fallback,
                    'title': fallback,
                    'huge': fallback,
                    'giant': fallback,
                }
        return self._fonts

    def _get_ip_address(self) -> str:
        """Get local IP address."""
        try:
            import socket
            # Connect to external address to find local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return ""

    def _get_wifi_status(self) -> Dict[str, Any]:
        """Get WiFi connection status."""
        import subprocess

        # Try nmcli first (works on modern Pi OS)
        try:
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'ACTIVE,SSID,SIGNAL', 'dev', 'wifi'],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.strip().split('\n'):
                if line.startswith('yes:'):
                    parts = line.split(':')
                    if len(parts) >= 3:
                        ssid = parts[1]
                        signal = int(parts[2]) if parts[2].isdigit() else 50
                        ip = self._get_ip_address()
                        return {"connected": True, "ssid": ssid, "signal": signal, "ip": ip}
        except Exception:
            pass

        # Fallback to iwconfig
        try:
            result = subprocess.run(
                ['iwconfig', 'wlan0'],
                capture_output=True, text=True, timeout=2
            )
            output = result.stdout

            if 'ESSID:' in output and 'ESSID:off/any' not in output:
                ssid = ""
                if 'ESSID:"' in output:
                    start = output.index('ESSID:"') + 7
                    end = output.index('"', start)
                    ssid = output[start:end]

                signal = 0
                if 'Link Quality=' in output:
                    try:
                        qual_str = output.split('Link Quality=')[1].split()[0]
                        num, denom = qual_str.split('/')
                        signal = int(100 * int(num) / int(denom))
                    except (IndexError, ValueError):
                        signal = 50

                ip = self._get_ip_address()
                return {"connected": True, "ssid": ssid, "signal": signal, "ip": ip}
        except Exception:
            pass

        # Final fallback: check network connectivity
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=1)
            ip = self._get_ip_address()
            return {"connected": True, "ssid": "connected", "signal": 50, "ip": ip}
        except Exception:
            return {"connected": False}

    def _get_battery_status(self) -> Dict[str, Any]:
        """Get battery status (if UPS HAT or battery available)."""
        try:
            # Check for PiJuice (common UPS HAT)
            battery_path = Path("/sys/class/power_supply/battery/capacity")
            if battery_path.exists():
                level = int(battery_path.read_text().strip())
                charging_path = Path("/sys/class/power_supply/battery/status")
                charging = False
                if charging_path.exists():
                    status = charging_path.read_text().strip().lower()
                    charging = status in ("charging", "full")
                return {"available": True, "level": level, "charging": charging}

            # Check for other common battery paths
            for path in ["/sys/class/power_supply/BAT0/capacity",
                        "/sys/class/power_supply/BAT1/capacity"]:
                p = Path(path)
                if p.exists():
                    level = int(p.read_text().strip())
                    return {"available": True, "level": level, "charging": False}

            return {"available": False}
        except Exception:
            return {"available": False}

    def warm_learning_cache(self):
        """Pre-warm the learning screen cache in background thread.

        Called after initialization to avoid 9+ second delay on first learning screen visit.
        """
        import threading

        def _warm():
            try:
                if self._learning_cache_refreshing:
                    return  # Already refreshing
                self._learning_cache_refreshing = True
                try:
                    if self._learning_visualizer is None:
                        self._learning_visualizer = LearningVisualizer(db_path=self._db_path)
                    # Get summary with None readings/anima - just warms the DB query cache
                    # The actual render will re-query with real data, but DB is now warmed
                    self._learning_cache = self._learning_visualizer.get_learning_summary(
                        readings=None, anima=None
                    )
                    self._learning_cache_time = time.time()
                    print("[Learning] Cache pre-warmed successfully", file=sys.stderr, flush=True)
                finally:
                    self._learning_cache_refreshing = False
            except Exception as e:
                print(f"[Learning] Cache pre-warm failed: {e}", file=sys.stderr, flush=True)
                self._learning_cache_refreshing = False

        thread = threading.Thread(target=_warm, daemon=True, name="learning-cache-warm")
        thread.start()
        print("[Learning] Starting cache pre-warm in background", file=sys.stderr, flush=True)

    def get_mode(self) -> ScreenMode:
        """Get current screen mode."""
        return self._state.mode
    
    def set_mode(self, mode: ScreenMode):
        """Set screen mode with fade transition."""
        import time
        now = time.time()
        if mode == self._state.mode:
            return  # Already on this mode
        # Very minimal debounce - allow rapid switching
        if now - self._state.last_switch_time < 0.02:  # 20ms debounce (almost none)
            return

        # Capture current screen for transition effect
        if hasattr(self._display, '_image') and self._display._image is not None:
            self._state.previous_image = self._display._image.copy()
            self._state.transition_progress = 0.0
            self._state.transition_start_time = now

        # Log mode changes, especially for notepad
        old_mode = self._state.mode
        self._state.mode = mode
        self._state.last_switch_time = now
        self._state.last_user_action_time = now

        if mode == ScreenMode.NOTEPAD:
            print(f"[ScreenRenderer] Switched to NOTEPAD from {old_mode.value}, pixels={len(self._canvas.pixels)}", file=sys.stderr, flush=True)
    
    def next_mode(self):
        """Cycle to next screen mode (including notepad)."""
        # Cycle through all screens including notepad, questions, and visitors
        regular_modes = [ScreenMode.FACE, ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS, ScreenMode.HEALTH, ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.MESSAGES, ScreenMode.QUESTIONS, ScreenMode.VISITORS, ScreenMode.NOTEPAD, ScreenMode.ART_ERAS]
        if self._state.mode not in regular_modes:
            # If somehow on unknown mode, go to face
            self.set_mode(ScreenMode.FACE)
            return
        current_idx = regular_modes.index(self._state.mode)
        next_idx = (current_idx + 1) % len(regular_modes)
        self.set_mode(regular_modes[next_idx])

    def previous_mode(self):
        """Cycle to previous screen mode (including notepad)."""
        # Cycle through all screens including notepad, questions, and visitors
        regular_modes = [ScreenMode.FACE, ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS, ScreenMode.HEALTH, ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.MESSAGES, ScreenMode.QUESTIONS, ScreenMode.VISITORS, ScreenMode.NOTEPAD, ScreenMode.ART_ERAS]
        if self._state.mode not in regular_modes:
            # If somehow on unknown mode, go to face
            self.set_mode(ScreenMode.FACE)
            return
        current_idx = regular_modes.index(self._state.mode)
        prev_idx = (current_idx - 1) % len(regular_modes)
        self.set_mode(regular_modes[prev_idx])
    
    def toggle_notepad(self):
        """Toggle notepad mode - enter if not on notepad, exit to face if on notepad."""
        if self._state.mode == ScreenMode.NOTEPAD:
            self.set_mode(ScreenMode.FACE)
        else:
            self.set_mode(ScreenMode.NOTEPAD)

    def next_group(self):
        """Switch to next top-level screen group."""
        group_info = self._SCREEN_GROUPS.get(self._state.mode)
        if not group_info:
            self.set_mode(ScreenMode.FACE)
            return
        group_name = group_info[0]
        group_order = ["home", "info", "mind", "msgs", "art"]
        group_default = {
            "home": ScreenMode.FACE,
            "info": ScreenMode.IDENTITY,
            "mind": ScreenMode.NEURAL,
            "msgs": ScreenMode.MESSAGES,
            "art": ScreenMode.NOTEPAD,
        }
        if group_name not in group_order:
            self.set_mode(ScreenMode.FACE)
            return
        idx = group_order.index(group_name)
        next_group = group_order[(idx + 1) % len(group_order)]
        self.set_mode(group_default[next_group])

    def previous_group(self):
        """Switch to previous top-level screen group."""
        group_info = self._SCREEN_GROUPS.get(self._state.mode)
        if not group_info:
            self.set_mode(ScreenMode.FACE)
            return
        group_name = group_info[0]
        group_order = ["home", "info", "mind", "msgs", "art"]
        group_default = {
            "home": ScreenMode.FACE,
            "info": ScreenMode.IDENTITY,
            "mind": ScreenMode.NEURAL,
            "msgs": ScreenMode.MESSAGES,
            "art": ScreenMode.NOTEPAD,
        }
        if group_name not in group_order:
            self.set_mode(ScreenMode.FACE)
            return
        idx = group_order.index(group_name)
        prev_group = group_order[(idx - 1) % len(group_order)]
        self.set_mode(group_default[prev_group])

    _CYCLE_GROUPS = {"msgs"}  # Groups where left/right cycles within before jumping

    def navigate_right(self):
        """Navigate right: next screen in group (msgs only), or next group."""
        group_info = self._SCREEN_GROUPS.get(self._state.mode)
        if not group_info:
            self.set_mode(ScreenMode.FACE)
            return
        group_name, group_screens = group_info
        if group_name not in self._CYCLE_GROUPS or len(group_screens) <= 1:
            self.next_group()
        else:
            idx = group_screens.index(self._state.mode)
            if idx == len(group_screens) - 1:
                self.next_group()
            else:
                self.set_mode(group_screens[idx + 1])

    def navigate_left(self):
        """Navigate left: previous screen in group (msgs only), or previous group."""
        group_info = self._SCREEN_GROUPS.get(self._state.mode)
        if not group_info:
            self.set_mode(ScreenMode.FACE)
            return
        group_name, group_screens = group_info
        if group_name not in self._CYCLE_GROUPS or len(group_screens) <= 1:
            self.previous_group()
        else:
            idx = group_screens.index(self._state.mode)
            if idx == 0:
                self.previous_group()
            else:
                self.set_mode(group_screens[idx - 1])

    def next_in_group(self):
        """Switch to next screen within current group."""
        group_info = self._SCREEN_GROUPS.get(self._state.mode)
        if not group_info:
            return
        _, group_screens = group_info
        if len(group_screens) <= 1:
            return
        idx = group_screens.index(self._state.mode)
        self.set_mode(group_screens[(idx + 1) % len(group_screens)])

    def previous_in_group(self):
        """Switch to previous screen within current group."""
        group_info = self._SCREEN_GROUPS.get(self._state.mode)
        if not group_info:
            return
        _, group_screens = group_info
        if len(group_screens) <= 1:
            return
        idx = group_screens.index(self._state.mode)
        self.set_mode(group_screens[(idx - 1) % len(group_screens)])

    # Screen groups for indicator display
    _SCREEN_GROUPS = {
        ScreenMode.FACE: ("home", [ScreenMode.FACE]),
        ScreenMode.IDENTITY: ("info", [ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS, ScreenMode.HEALTH]),
        ScreenMode.SENSORS: ("info", [ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS, ScreenMode.HEALTH]),
        ScreenMode.DIAGNOSTICS: ("info", [ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS, ScreenMode.HEALTH]),
        ScreenMode.HEALTH: ("info", [ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS, ScreenMode.HEALTH]),
        ScreenMode.NEURAL: ("mind", [ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.GOALS_BELIEFS, ScreenMode.AGENCY]),
        ScreenMode.INNER_LIFE: ("mind", [ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.GOALS_BELIEFS, ScreenMode.AGENCY]),
        ScreenMode.LEARNING: ("mind", [ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.GOALS_BELIEFS, ScreenMode.AGENCY]),
        ScreenMode.SELF_GRAPH: ("mind", [ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.GOALS_BELIEFS, ScreenMode.AGENCY]),
        ScreenMode.GOALS_BELIEFS: ("mind", [ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.GOALS_BELIEFS, ScreenMode.AGENCY]),
        ScreenMode.AGENCY: ("mind", [ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.GOALS_BELIEFS, ScreenMode.AGENCY]),
        ScreenMode.MESSAGES: ("msgs", [ScreenMode.MESSAGES, ScreenMode.QUESTIONS, ScreenMode.VISITORS]),
        ScreenMode.QUESTIONS: ("msgs", [ScreenMode.MESSAGES, ScreenMode.QUESTIONS, ScreenMode.VISITORS]),
        ScreenMode.VISITORS: ("msgs", [ScreenMode.MESSAGES, ScreenMode.QUESTIONS, ScreenMode.VISITORS]),
        ScreenMode.NOTEPAD: ("art", [ScreenMode.NOTEPAD, ScreenMode.ART_ERAS]),
        ScreenMode.ART_ERAS: ("art", [ScreenMode.NOTEPAD, ScreenMode.ART_ERAS]),
    }

    def _draw_screen_indicator(self, draw, current_mode: ScreenMode):
        """Draw group name + position indicator at bottom-right (e.g., 'mind 2/3')."""
        group_info = self._SCREEN_GROUPS.get(current_mode)
        if not group_info:
            return

        group_name, group_screens = group_info
        try:
            pos = group_screens.index(current_mode) + 1
        except ValueError:
            return

        total = len(group_screens)
        if total == 1:
            label = group_name
        else:
            label = f"{group_name} {pos}/{total}"

        fonts = self._get_fonts()
        font = fonts.get('tiny', fonts.get('micro'))
        DIM = (100, 100, 100)

        # Right-aligned at bottom (y=228 with micro font fits within 240px display)
        font = fonts.get('micro', fonts.get('tiny'))
        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(label) * 6
        x = 240 - text_w - 6
        y = 228
        draw.text((x, y), label, fill=DIM, font=font)

    def _draw_status_bar(self, draw):
        """Draw status indicators at top-right (WiFi, governance connection)."""
        x = 220  # Right side
        y = 4    # Top

        # WiFi indicator (simple arc symbol)
        if self._state.wifi_connected:
            # Connected - green wifi symbol
            wifi_color = (80, 200, 80)
            # Draw simple wifi bars
            draw.arc([x-8, y, x, y+8], 180, 360, fill=wifi_color, width=1)
            draw.arc([x-6, y+2, x-2, y+6], 180, 360, fill=wifi_color, width=1)
            draw.ellipse([x-5, y+5, x-3, y+7], fill=wifi_color)
        else:
            # Disconnected - red X
            wifi_color = (200, 80, 80)
            draw.line([x-8, y, x, y+8], fill=wifi_color, width=1)
            draw.line([x-8, y+8, x, y], fill=wifi_color, width=1)

        x -= 16  # Move left for governance indicator

        # Governance indicator (circle with G or dot)
        if self._state.governance_connected:
            # Connected - cyan dot
            gov_color = (80, 200, 200)
            draw.ellipse([x-6, y+1, x, y+7], fill=gov_color)
        else:
            # Disconnected - dim dot
            gov_color = (60, 60, 60)
            draw.ellipse([x-6, y+1, x, y+7], fill=gov_color)

    def _draw_loading_indicator(self, draw, image):
        """Draw loading spinner overlay when waiting for LLM response."""
        if not self._state.is_loading:
            return

        # Semi-transparent overlay
        from PIL import Image, ImageDraw
        overlay = Image.new('RGBA', (240, 240), (0, 0, 0, 128))
        overlay_draw = ImageDraw.Draw(overlay)

        # Animated spinner (rotating dots)
        elapsed = time.time() - self._state.loading_start_time
        angle_offset = int(elapsed * 360) % 360  # Full rotation per second

        cx, cy = 120, 110  # Center of screen
        radius = 20
        for i in range(8):
            angle = math.radians(i * 45 + angle_offset)
            x = cx + int(radius * math.cos(angle))
            y = cy + int(radius * math.sin(angle))
            # Dots fade as they get older in rotation
            brightness = 255 - (i * 25)
            color = (brightness, brightness, brightness, 255)
            overlay_draw.ellipse([x-3, y-3, x+3, y+3], fill=color)

        # Loading message
        if self._state.loading_message:
            fonts = self._get_fonts()
            overlay_draw.text((120, 145), self._state.loading_message,
                            fill=(200, 200, 200, 255), font=fonts['small'], anchor="mm")

        # Composite overlay onto image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return Image.alpha_composite(image, overlay).convert('RGB')

    def _apply_transition(self, new_image):
        """Apply fade transition effect between screens."""
        if self._state.transition_progress >= 1.0:
            return new_image

        if self._state.previous_image is None:
            self._state.transition_progress = 1.0
            return new_image

        # Calculate transition progress with ease-in-out
        elapsed = time.time() - self._state.transition_start_time
        linear = min(1.0, elapsed / self._state.transition_duration)
        progress = ease_smooth(linear)  # Gentle start and end
        self._state.transition_progress = progress

        if progress >= 1.0:
            self._state.previous_image = None
            return new_image

        # Blend old and new images
        from PIL import Image
        try:
            old_img = self._state.previous_image
            if old_img.size != new_image.size:
                old_img = old_img.resize(new_image.size)
            # Cross-fade with eased progress — smoother feel
            blended = Image.blend(old_img.convert('RGB'), new_image.convert('RGB'), progress)
            return blended
        except Exception:
            self._state.transition_progress = 1.0
            return new_image

    def set_loading(self, message: str = "thinking..."):
        """Set loading state (called when starting LLM request)."""
        self._state.is_loading = True
        self._state.loading_message = message
        self._state.loading_start_time = time.time()

    def clear_loading(self):
        """Clear loading state (called when LLM response received)."""
        self._state.is_loading = False
        self._state.loading_message = ""

    def update_connection_status(self, wifi: bool = None, governance: bool = None):
        """Update connection status indicators."""
        if wifi is not None:
            self._state.wifi_connected = wifi
        if governance is not None:
            self._state.governance_connected = governance

    def trigger_input_feedback(self, direction: str):
        """Trigger visual feedback for joystick/button input.

        Args:
            direction: "left", "right", "up", "down", or "press"
        """
        self._state.input_feedback_until = time.time() + 0.1  # 100ms flash
        self._state.input_feedback_direction = direction

    def _draw_input_feedback(self, draw, image):
        """Draw edge highlight for input feedback."""
        if time.time() >= self._state.input_feedback_until:
            return

        direction = self._state.input_feedback_direction
        width, height = 240, 240
        feedback_color = (60, 120, 180)  # Subtle blue
        edge_width = 4

        if direction == "left":
            # Highlight left edge
            draw.rectangle([0, 0, edge_width, height], fill=feedback_color)
        elif direction == "right":
            # Highlight right edge
            draw.rectangle([width - edge_width, 0, width, height], fill=feedback_color)
        elif direction == "up":
            # Highlight top edge
            draw.rectangle([0, 0, width, edge_width], fill=feedback_color)
        elif direction == "down":
            # Highlight bottom edge
            draw.rectangle([0, height - edge_width, width, height], fill=feedback_color)
        elif direction == "press":
            # Brief corner highlights for button press
            corner_size = 12
            draw.rectangle([0, 0, corner_size, corner_size], fill=feedback_color)
            draw.rectangle([width - corner_size, 0, width, corner_size], fill=feedback_color)
            draw.rectangle([0, height - corner_size, corner_size, height], fill=feedback_color)
            draw.rectangle([width - corner_size, height - corner_size, width, height], fill=feedback_color)

    def trigger_brightness_overlay(self, preset_name: str, display_level: float):
        """Show brightness overlay for 1.5 seconds."""
        self._state.brightness_changed_at = time.time()
        self._state.brightness_overlay_name = preset_name
        self._state.brightness_overlay_level = display_level

    def trigger_controls_overlay(self, duration_s: float = 1.8):
        """Show compact controls help overlay."""
        self._state.controls_overlay_until = time.time() + max(0.5, duration_s)

    def _get_action_hint(self, mode: ScreenMode) -> str:
        """Get one-line control hint shown at bottom-left."""
        if mode == ScreenMode.FACE:
            return "L/R groups  U/D LEDs"
        if mode in (ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS, ScreenMode.HEALTH, ScreenMode.NEURAL, ScreenMode.INNER_LIFE, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH):
            return "L/R groups  U/D pages"
        if mode in (ScreenMode.MESSAGES, ScreenMode.VISITORS):
            if self._state.message_expanded_id is not None:
                return "\u2191\u2193 scroll text   btn: back"
            return "\u2191\u2193 scroll   btn: read   \u25ce next"
        if mode == ScreenMode.QUESTIONS:
            if self._state.qa_full_view:
                return "U/D text  btn back"
            if self._state.qa_expanded:
                return "U/D text  L/R focus"
            return "U/D select  btn expand"
        if mode == ScreenMode.ART_ERAS:
            return "U/D choose  btn select"
        if mode == ScreenMode.NOTEPAD:
            return "btn save  hold btn power"
        return "L/R groups"

    def _draw_action_hint(self, draw, mode: ScreenMode):
        """Draw persistent action hint at bottom-left."""
        fonts = self._get_fonts()
        hint = self._get_action_hint(mode)
        draw.text((8, 232), hint, fill=(105, 105, 105), font=fonts.get('tiny', fonts.get('micro')))

    def _draw_controls_overlay(self, draw, mode: ScreenMode):
        """Draw a small controls card for the current screen."""
        if time.time() >= self._state.controls_overlay_until:
            return
        fonts = self._get_fonts()
        hint = self._get_action_hint(mode)
        box_x, box_y, box_w, box_h = 18, 72, 204, 98
        draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(14, 18, 24), outline=(70, 110, 150), width=2)
        draw.text((box_x + 10, box_y + 8), "controls", fill=(130, 220, 255), font=fonts['medium'])
        draw.text((box_x + 10, box_y + 30), hint, fill=(215, 215, 215), font=fonts['small'])
        draw.text((box_x + 10, box_y + 48), "L/R switch groups", fill=(150, 165, 180), font=fonts['tiny'])
        draw.text((box_x + 10, box_y + 60), "click stick: cycle pages", fill=(150, 165, 180), font=fonts['tiny'])
        draw.text((box_x + 10, box_y + 72), "hold side btn 3s: shutdown", fill=(150, 165, 180), font=fonts['tiny'])

    def _draw_brightness_overlay(self, draw, image):
        """Draw LED brightness overlay (centered box with name + bar)."""
        elapsed = time.time() - self._state.brightness_changed_at
        if elapsed >= 1.5:
            return

        name = self._state.brightness_overlay_name
        level = self._state.brightness_overlay_level
        w, h = 240, 240

        # Semi-transparent dark box (centered)
        box_w, box_h = 100, 50
        bx = (w - box_w) // 2
        by = (h - box_h) // 2

        # Fade out in last 0.3s
        alpha = 1.0 if elapsed < 1.2 else (1.5 - elapsed) / 0.3

        # Draw dark background
        bg_color = tuple(int(20 * alpha) for _ in range(3))
        draw.rectangle([bx - 2, by - 2, bx + box_w + 2, by + box_h + 2], fill=bg_color)

        fonts = self._get_fonts()

        # "LEDs" label (small, top)
        label_color = tuple(int(120 * alpha) for _ in range(3))
        draw.text((bx + 33, by + 2), "LEDs", fill=label_color, font=fonts['micro'])

        # Preset name (larger, below label)
        text_color = tuple(int(220 * alpha) for _ in range(3))
        bbox = draw.textbbox((0, 0), name, font=fonts['medium'])
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, by + 14), name, fill=text_color, font=fonts['medium'])

        # LED brightness bar - 4 segment design for clear visibility
        bar_y = by + 34
        bar_w = box_w - 16
        bar_x = bx + 8
        bar_h = 8  # Taller bar

        # Draw 4 distinct segments (one per preset level)
        segment_w = (bar_w - 6) // 4  # 4 segments with gaps
        gap = 2

        # Map level to number of segments: 1.0=4, 0.53=3, 0.27=2, 0.13=1
        if level > 0.8:
            segments_lit = 4
        elif level > 0.4:
            segments_lit = 3
        elif level > 0.2:
            segments_lit = 2
        else:
            segments_lit = 1

        for i in range(4):
            sx = bar_x + i * (segment_w + gap)
            if i < segments_lit:
                # Lit segment - bright cyan
                seg_color = tuple(int(c * alpha) for c in (100, 200, 240))
            else:
                # Unlit segment - dark
                seg_color = tuple(int(40 * alpha) for _ in range(3))
            draw.rectangle([sx, bar_y, sx + segment_w, bar_y + bar_h], fill=seg_color)

    def render(
        self,
        face_state: Optional[FaceState] = None,
        anima: Optional[Anima] = None,
        readings: Optional[SensorReadings] = None,
        identity: Optional[CreatureIdentity] = None,
        governance: Optional[Dict[str, Any]] = None
    ):
        """Render current screen based on mode."""
        import time
        render_start = time.time()
        mode = self._state.mode

        # Use lock to prevent concurrent renders (threading issue causes blank screen)
        with self._render_lock:
            t_locked = time.time()
            # Defer SPI push until after post-processing (single transfer per render)
            self._display._deferred = True

            # Store latest state for use by canvas_save growth notifications
            self._last_anima = anima
            self.drawing_engine._last_readings = readings

            # Store UNITARES agent_id for identity screen display
            if governance and governance.get("unitares_agent_id"):
                self._unitares_agent_id = governance.get("unitares_agent_id")

            # Enforce governance verdicts: pause drawing when governance says pause/halt
            if governance:
                action = governance.get("action", "proceed")
                if action in ("pause", "halt", "reject"):
                    self._state.governance_paused = True
                elif action == "proceed":
                    self._state.governance_paused = False

            # Check Lumen's canvas autonomy (can save/clear regardless of screen)
            # Throttle to every 10th frame — save/energy checks don't need 3Hz
            try:
                if not hasattr(self._state, '_frame_count'):
                    self._state._frame_count = 0
                self._state._frame_count += 1
                if self._state._frame_count % 10 == 0:
                    self.canvas_check_autonomy(anima)
                    # Heartbeat: drawing subsystem is alive
                    try:
                        from ..health import get_health_registry
                        get_health_registry().heartbeat("drawing")
                    except Exception:
                        pass
            except Exception as e:
                # Don't let autonomy errors break rendering, but always log them
                import traceback
                print(f"[Canvas] Autonomy check error: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

            # Disable auto-return - let user stay on screens as long as they want
            # Only auto-return to FACE if explicitly requested via button
            # (Auto-return disabled to prevent getting stuck)

            t_pre = time.time()
            try:
                if mode == ScreenMode.FACE:
                    self._render_face(face_state, identity)
                elif mode == ScreenMode.SENSORS:
                    self._render_sensors(readings)
                elif mode == ScreenMode.IDENTITY:
                    self._render_identity(identity)
                elif mode == ScreenMode.DIAGNOSTICS:
                    self._render_diagnostics(anima, readings, governance)
                elif mode == ScreenMode.NEURAL:
                    self._render_neural(anima, readings)
                elif mode == ScreenMode.INNER_LIFE:
                    self._render_inner_life()
                elif mode == ScreenMode.LEARNING:
                    self._render_learning(anima, readings)
                elif mode == ScreenMode.SELF_GRAPH:
                    self._render_self_graph(anima, readings, identity)
                elif mode == ScreenMode.GOALS_BELIEFS:
                    self._render_goals_beliefs(anima, identity)
                elif mode == ScreenMode.AGENCY:
                    self._render_agency()
                elif mode == ScreenMode.MESSAGES:
                    self._render_messages()
                elif mode == ScreenMode.QUESTIONS:
                    self._render_questions()
                elif mode == ScreenMode.VISITORS:
                    self._render_visitors()
                elif mode == ScreenMode.NOTEPAD:
                    try:
                        self._render_notepad(anima)
                    except Exception as e:
                        print(f"[ScreenRenderer] Error rendering notepad: {e}", file=sys.stderr, flush=True)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                        # Fallback: show text version
                        try:
                            self._display.render_text("NOTEPAD\n\nError\nrendering", (10, 10))
                        except Exception:
                            pass
                elif mode == ScreenMode.ART_ERAS:
                    self._render_art_eras(anima)
                elif mode == ScreenMode.HEALTH:
                    self._render_health()
                else:
                    # Unknown mode - show default to prevent blank screen
                    print(f"[Screen] Unknown mode: {mode}, showing default", file=sys.stderr, flush=True)
                    self._display.show_default()

                # Background drawing: Lumen draws even when notepad isn't displayed.
                # Throttled to every 5th frame (~every 10s at 2s/loop) to limit CPU when on other screens.
                # Skips when governance says pause/halt (enforces verdict).
                if (mode != ScreenMode.NOTEPAD and anima and hasattr(self._display, '_create_canvas')
                        and self._state._frame_count % 5 == 0
                        and time.time() >= self._canvas.drawing_paused_until
                        and not getattr(self._state, 'governance_paused', False)
                        and len(self._canvas.pixels) < 15000):
                    try:
                        self._lumen_draw(anima, draw=None)
                    except Exception:
                        pass  # Don't let background draw break display
            except Exception as e:
                # Any render error - show default to prevent blank screen
                print(f"[Screen] Render error for {mode.value}: {e}", file=sys.stderr, flush=True)
                try:
                    self._display.show_default()
                except Exception:
                    pass
            t_mode = time.time()

            # === Post-processing: transitions, input feedback, loading overlays ===
            try:
                if hasattr(self._display, '_image') and self._display._image is not None:
                    from PIL import ImageDraw
                    image = self._display._image

                    # Apply screen transition (fade effect)
                    if self._state.transition_progress < 1.0:
                        image = self._apply_transition(image)

                    draw = ImageDraw.Draw(image)

                    # Draw input feedback (joystick/button visual acknowledgment)
                    if time.time() < self._state.input_feedback_until:
                        self._draw_input_feedback(draw, image)

                    # Draw brightness overlay (when user changes brightness)
                    if time.time() - self._state.brightness_changed_at < 1.5:
                        self._draw_brightness_overlay(draw, image)

                    # Draw controls overlay (hold joystick button)
                    self._draw_controls_overlay(draw, mode)

                    # Apply loading indicator overlay
                    if self._state.is_loading:
                        result = self._draw_loading_indicator(None, image)
                        if result is not None:
                            image = result

                    self._display._image = image
            except Exception as e:
                print(f"[Screen] Post-processing error: {e}", file=sys.stderr, flush=True)

            # Ensure we have something to display — prevent blank screens
            if not hasattr(self._display, '_image') or self._display._image is None:
                try:
                    self._display.show_default()
                except Exception:
                    pass
            t_post = time.time()

            # Single SPI push — all drawing is done, flush to hardware
            self._display._deferred = False
            self._display.flush()
            t_flush = time.time()

        # Log slow renders with phase breakdown so outliers reveal the hotspot
        render_time = t_flush - render_start
        if render_time > 0.5:
            phases = (
                f"lock_wait={(t_locked-render_start)*1000:.0f}ms "
                f"pre={(t_pre-t_locked)*1000:.0f}ms "
                f"mode={(t_mode-t_pre)*1000:.0f}ms "
                f"post={(t_post-t_mode)*1000:.0f}ms "
                f"flush={(t_flush-t_post)*1000:.0f}ms"
            )
            print(f"[Screen] Slow render: {mode.value} took {render_time*1000:.0f}ms [{phases}]", file=sys.stderr, flush=True)
    
    
    def _wrap_text(self, text: str, font, max_width: int) -> list:
        """Wrap text to fit within max_width pixels. Returns list of lines."""
        # Use cached draw context for measurement (avoids creating PIL Image every call)
        temp_draw = self._get_measure_draw()

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            try:
                bbox = temp_draw.textbbox((0, 0), test_line, font=font)
                width = bbox[2] - bbox[0]
            except Exception:
                # Fallback: estimate ~7 pixels per character
                width = len(test_line) * 7

            if width > max_width and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line

        if current_line:
            lines.append(current_line)

        return lines
