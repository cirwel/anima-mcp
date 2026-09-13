"""
Bridge between anima-mcp and unitares-governance MCP server.

Enables creature to check in with UNITARES governance system via HTTP/SSE.
Provides fallback local governance if UNITARES server is unavailable.
"""

import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .identity.store import CreatureIdentity

from .eisv_mapper import (
    BODY_EISV_PROJECTION_SCHEMA,
    BodyEISVProjection,
    anima_to_body_eisv_projection,
    estimate_complexity,
    generate_status_text,
    compute_ethical_drift,
    compute_confidence,
)
from .anima import Anima
from .atomic_write import atomic_json_write
from .sensors.base import SensorReadings

logger = logging.getLogger(__name__)

_EISV_KEYS = ("E", "I", "S", "V")


def _validated_eisv_vector(candidate: Any) -> Optional[Dict[str, float]]:
    """Return a finite EISV vector, or None when a response is not one."""
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


def _extract_governance_eisv(
    response: Dict[str, Any],
) -> tuple[Optional[Dict[str, float]], str]:
    """Extract UNITARES's own inferred EISV without substituting body input.

    UNITARES response modes have historically placed the primary vector at
    ``primary_eisv``, at flat E/I/S/V fields inside ``metrics`` or the minimal
    response itself, or at ``eisv``. All are server outputs here, so they may
    be identified as governance state. Missing output remains explicitly
    missing.
    """
    metrics = response.get("metrics")
    candidates = [
        response.get("primary_eisv"),
        metrics.get("primary_eisv") if isinstance(metrics, dict) else None,
        metrics,
        response,
        response.get("eisv"),
    ]
    source = response.get("primary_eisv_source")
    if not source and isinstance(metrics, dict):
        source = metrics.get("primary_eisv_source")
    for candidate in candidates:
        vector = _validated_eisv_vector(candidate)
        if vector is not None:
            return vector, str(
                source or "unitares_primary_response_source_omitted"
            )
    return None, "not_returned_by_unitares"


def _state_space_fields(
    body_projection: BodyEISVProjection,
    governance_response: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build explicit state-space fields plus the legacy ``eisv`` alias."""
    body_vector = body_projection.to_dict()
    governance_vector, governance_source = _extract_governance_eisv(
        governance_response or {}
    )
    return {
        "body_eisv_projection": body_vector,
        "governance_eisv": governance_vector,
        "governance_eisv_source": governance_source,
        # Compatibility only. New readers must use body_eisv_projection.
        "eisv": body_vector,
        "eisv_source": "body_eisv_projection_legacy_alias",
        "state_space_provenance": {
            "body_eisv_projection": {
                "schema": BODY_EISV_PROJECTION_SCHEMA,
                "source": "anima_sensor_projection",
                "role": "body_measurement",
            },
            "governance_eisv": {
                "source": governance_source,
                "role": "unitares_inferred_state",
            },
            "eisv": {
                "alias_of": "body_eisv_projection",
                "deprecated": True,
            },
        },
    }


class _AnimaSnapshot:
    """Lightweight snapshot of anima state for delta computation between check-ins."""
    __slots__ = ('warmth', 'clarity', 'stability', 'presence')
    def __init__(self, w, c, s, p):
        self.warmth, self.clarity, self.stability, self.presence = w, c, s, p


class IdentityRefusedError(Exception):
    """UNITARES refused the check-in for identity reasons (typed strict refusal).

    Raised when the response says the session binding no longer resolves
    (status=identity_required / SESSION_ERROR). Distinct from transport errors
    so check_in can attempt the sanctioned re-anchor instead of just falling
    back to local governance. Found 2026-07-02 (anima-mcp #97): the typed
    refusal payload carries neither success:false nor an action, so it
    previously parsed as a silent default-"proceed" — the bridge never knew
    the server had stopped attributing its check-ins.
    """


def _is_identity_refusal(payload: Dict[str, Any]) -> bool:
    """True when a tool payload is UNITARES's typed identity refusal (#97)."""
    return (
        payload.get("status") == "identity_required"
        or payload.get("error_code") == "SESSION_ERROR"
        or payload.get("error_category") == "auth_error"
    )


def mcp_tool_refusal(response: Any) -> Optional[str]:
    """Return why a JSON-RPC ``tools/call`` response is not a success, or None.

    ``"result" in response`` is not a success test, and on ``/mcp/`` it counts
    every refusal as success. There are three refusal layers:

    - a JSON-RPC ``error`` object;
    - ``result.isError`` — the SDK's layer: an unknown tool name, or arguments
      that fail schema validation. A pre-consolidation name such as
      ``store_knowledge_graph`` lands here, because ``/mcp/`` does not resolve
      the alias table that REST and stdio do (measured 2026-09-13);
    - a normal result whose text payload says so — ``{"success": false}`` from
      a handler (binding, ownership, not-found), or the typed identity refusal,
      which is success-shaped and carries no ``success`` key at all.
    """
    if not isinstance(response, dict):
        return "no response"
    if "error" in response:
        error = response["error"]
        message = error.get("message") if isinstance(error, dict) else None
        return str(message or error)
    result = response.get("result")
    if not isinstance(result, dict):
        return "no result"
    content = result.get("content")
    first = content[0] if isinstance(content, list) and content else None
    text = first.get("text") if isinstance(first, dict) else None
    if result.get("isError"):
        return text or "isError"
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("success") is False or _is_identity_refusal(payload):
        return str(
            payload.get("error")
            or payload.get("hint")
            or payload.get("error_code")
            or payload.get("status")
            or "refused"
        )
    return None


class UnitaresBridge:
    """
    Connect anima creature to UNITARES governance.

    Supports:
    - HTTP/SSE connection to UNITARES server
    - Fallback local governance if server unavailable
    - Automatic retry and error handling
    - Connection pooling (reuses single aiohttp session)
    """

    def __init__(
        self,
        unitares_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize bridge.

        Args:
            unitares_url: URL to UNITARES governance server (e.g., "http://127.0.0.1:8767/mcp/")
                         If None, will use local governance only
            agent_id: Agent ID for UNITARES (auto-generated if None)
            timeout: Request timeout in seconds
        """
        self._url = unitares_url
        self._agent_id = agent_id
        self._timeout = timeout
        self._session_id = None
        self._available = None  # None = not checked, True/False = checked
        self._last_availability_check = None  # Timestamp of last check
        # Circuit breaker: after N consecutive failures, skip UNITARES with exponential backoff
        self._circuit_failures = 0
        self._circuit_open_until = 0.0  # Timestamp when circuit closes (half-open)
        self._circuit_threshold = 2
        self._circuit_backoff_base = 15.0
        self._circuit_backoff_max = 120.0
        self._circuit_current_backoff = 15.0
        self._http_session = None  # Reusable aiohttp session
        self._session_timeout = None  # Timeout config for session
        # Previous check-in state for computing deltas (ethical_drift, confidence)
        self._prev_anima = None        # Previous Anima snapshot (warmth, clarity, stability, presence)
        self._prev_readings = None     # Previous sensor readings
        self._prev_complexity = None   # Previous complexity value
        # Basic auth for remote tunnels (format: "user:password")
        self._basic_auth = None
        auth_str = os.environ.get("UNITARES_AUTH")
        if auth_str and ":" in auth_str:
            import aiohttp
            user, password = auth_str.split(":", 1)
            self._basic_auth = aiohttp.BasicAuth(user, password)
        # Governance identity anchor (anima-mcp #97). The bridge is echo-only —
        # it presents a deterministic client_session_id and never re-onboards —
        # so a server-side session-store wipe (2026-06-30 Redis restart) left it
        # permanently refused while every token-holding resident self-healed.
        # The anchor is the same sanctioned rescue the SDK residents use:
        # HARVEST {uuid, continuity_token} from identity() while healthy, SPEND
        # them on a resume=true rebind when a check-in is identity-refused.
        # Binding refresh, not re-onboard (Phase-2 addendum constraint intact).
        self._anchor_path = Path(
            os.environ.get("ANIMA_GOV_ANCHOR_PATH",
                           str(Path.home() / ".anima" / "gov_identity.json"))
        )
        self._anchor: Optional[Dict[str, Any]] = self._load_anchor()
        self._reanchor_last_attempt = 0.0
        self._reanchor_cooldown_s = 600.0
        self._anchor_refresh_s = 24 * 3600.0

    async def _get_session(self):
        """Get or create reusable HTTP session (event-loop aware).

        Creates a new session if the event loop has changed (e.g. broker's
        _run_async_in_background creates a fresh loop per call).
        """
        import asyncio
        import aiohttp
        current_loop = asyncio.get_running_loop()
        # Recreate session if loop changed or session is closed
        if (self._http_session is not None
                and not self._http_session.closed
                and getattr(self, '_session_loop', None) is current_loop):
            return self._http_session
        # Close stale session from a different loop
        if self._http_session is not None and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception:
                pass
        connector = aiohttp.TCPConnector(
            limit=5,
            limit_per_host=3,
            ttl_dns_cache=300,
            force_close=True,  # Disable keep-alive — uvicorn hangs on idle connections
        )
        self._session_timeout = aiohttp.ClientTimeout(total=self._timeout)
        self._http_session = aiohttp.ClientSession(
            timeout=self._session_timeout,
            connector=connector,
            auth=self._basic_auth,
        )
        self._session_loop = current_loop
        return self._http_session

    def _get_mcp_url(self) -> str:
        """Resolve the MCP endpoint URL from the configured base URL."""
        base = self._url.rstrip("/")
        if "/mcp" in base:
            return base.split("/mcp", 1)[0] + "/mcp/"
        if base.endswith("/sse"):
            return base[:-len("/sse")] + "/mcp/"
        return f"{base}/mcp/"

    def _get_health_url(self) -> str:
        """Resolve UNITARES health URL from either base, /mcp, or legacy /sse URL."""
        base = self._url.rstrip("/")
        if "/mcp" in base:
            base = base.split("/mcp", 1)[0]
        elif base.endswith("/sse"):
            base = base[:-len("/sse")]
        return f"{base}/health"

    @staticmethod
    def _parse_mcp_response(text: str, content_type: str) -> Any:
        """Parse an MCP response, handling both JSON and SSE formats.

        Returns parsed JSON dict or None if no valid data found.
        """
        if "text/event-stream" in content_type:
            for line in text.split("\n"):
                if line.startswith("data: "):
                    try:
                        return json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
            return None
        return json.loads(text)

    async def close(self):
        """Close the HTTP session. Call when done with bridge."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    async def _reset_session(self):
        """Drop the pooled HTTP session after a connection failure.

        The reusable session's TCPConnector caches DNS for ttl_dns_cache (300s),
        so a transient network/DNS blip (e.g. Tailscale MagicDNS dropping out)
        can leave the connector pinned to a stale/failed resolution. Tearing the
        session down forces _get_session() to rebuild the connector with a fresh
        DNS lookup on the next attempt, so the client self-heals once the path
        recovers instead of needing a process restart.
        """
        session = self._http_session
        self._http_session = None
        self._session_loop = None
        if session is not None and not session.closed:
            try:
                await session.close()
            except Exception:
                pass

    async def check_availability(self) -> bool:
        """
        Check if UNITARES server is available.

        Returns:
            True if server is reachable and accessible, False otherwise
        """
        if self._url is None:
            self._available = False
            return False

        import time
        current_time = time.time()

        # Circuit breaker: skip checks while open (backoff handles retry timing)
        if current_time < self._circuit_open_until:
            return False

        # If circuit was open and backoff expired, reset to allow recheck
        if self._available is False and self._circuit_open_until > 0:
            self._available = None

        # If already available, return immediately unless stale (recheck every 5 min)
        if self._available is True:
            if self._last_availability_check and (current_time - self._last_availability_check < 300.0):
                return True
            # Fall through to recheck

        try:
            # Try to connect to UNITARES server using shared session
            session = await self._get_session()

            # Try health check or list_tools endpoint
            health_url = self._get_health_url()
            try:
                import aiohttp
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=self._timeout)) as response:
                    if response.status == 200:
                        self._available = True
                        self._circuit_failures = 0
                        self._circuit_current_backoff = self._circuit_backoff_base
                        self._last_availability_check = current_time
                        return True
                    elif response.status == 401:
                        # OAuth/auth required - not accessible from this client
                        logger.warning("UNITARES requires authentication (401) - using local governance")
                        self._available = False
                        self._circuit_failures += 1
                        self._last_availability_check = current_time
                        self._maybe_open_circuit(current_time)
                        return False
            except Exception:
                # Network/timeout errors - will retry later
                pass

            # If health check fails, try MCP endpoint
            mcp_url = self._get_mcp_url()
            try:
                import aiohttp
                async with session.post(
                    mcp_url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"
                    },
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    if response.status == 200:
                        self._available = True
                        self._circuit_failures = 0
                        self._circuit_current_backoff = self._circuit_backoff_base
                        self._last_availability_check = current_time
                        return True
                    elif response.status == 401:
                        # OAuth/auth required - not accessible from this client
                        logger.warning("UNITARES requires authentication (401) - using local governance")
                        self._available = False
                        self._circuit_failures += 1
                        self._last_availability_check = current_time
                        self._maybe_open_circuit(current_time)
                        return False
            except Exception:
                # Network/timeout errors - will retry later
                pass

            # Both checks failed - mark unavailable but allow retry
            self._available = False
            self._circuit_failures += 1
            self._last_availability_check = current_time
            self._maybe_open_circuit(current_time)
            # Rebuild the session so the next probe re-resolves DNS (self-heal)
            await self._reset_session()
            return False

        except ImportError:
            # aiohttp not available
            self._available = False
            return False
        except Exception:
            self._available = False
            self._circuit_failures += 1
            self._maybe_open_circuit(time.time())
            await self._reset_session()
            return False

    def _maybe_open_circuit(self, current_time: float) -> None:
        """Open circuit breaker if failure threshold reached (exponential backoff)."""
        if self._circuit_failures >= self._circuit_threshold:
            self._circuit_open_until = current_time + self._circuit_current_backoff
            logger.info(
                "UNITARES circuit breaker open for %.0fs (%d consecutive failures)",
                self._circuit_current_backoff, self._circuit_failures
            )
            # Double backoff for next time, capped at max
            self._circuit_current_backoff = min(
                self._circuit_current_backoff * 2, self._circuit_backoff_max
            )
    
    # ------------------------------------------------------------------
    # Governance identity anchor (anima-mcp #97)
    # ------------------------------------------------------------------

    def _client_session_id(self) -> str:
        # Prefer the SERVER-ISSUED canonical key (agent-<uuid12> prefix echo —
        # the fleet-standard write path) once the anchor holds one. The legacy
        # deterministic lumen-<anima-id> key is only the bootstrap identity
        # hint: acceptance-testing #97 showed identity(resume=true) rebinds the
        # canonical key, NOT a caller-supplied bespoke key, so recovery only
        # converges if the bridge echoes what the server actually bound.
        anchored = (self._anchor or {}).get("client_session_id")
        if anchored:
            return anchored
        return f"lumen-{self._agent_id}" if self._agent_id else "lumen-anima"

    def client_session_id(self) -> str:
        """The binding key this bridge presents on writes, for callers outside it."""
        return self._client_session_id()

    def _load_anchor(self) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(self._anchor_path.read_text())
            if data.get("uuid") and data.get("continuity_token"):
                return data
        except (OSError, ValueError):
            pass
        return None

    def _save_anchor(self, uuid: str, token: str, client_session_id: Optional[str] = None) -> None:
        try:
            payload = {
                "uuid": uuid,
                "continuity_token": token,
                "client_session_id": client_session_id or self._client_session_id(),
                "saved_at": time.time(),
            }
            atomic_json_write(self._anchor_path, payload, indent=2)
            self._anchor = payload
            logger.info("Governance anchor saved (uuid=%s)", uuid[:8])
        except OSError as e:
            logger.warning("Could not persist governance anchor: %s", e)

    async def _call_identity_tool(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call the UNITARES `identity` tool; return the unwrapped payload dict."""
        mcp_request = {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "identity", "arguments": arguments},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Session-ID": self._session_id or "anima-creature",
        }
        session = await self._get_session()
        async with session.post(self._get_mcp_url(), json=mcp_request, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"identity call HTTP {response.status}")
            payload = self._parse_mcp_response(
                await response.text(), response.headers.get("Content-Type", "")
            )
            result = (payload or {}).get("result") or {}
            if result.get("isError"):
                raise Exception("identity call returned isError")
            content = (result.get("content") or [{}])[0]
            if content.get("type") == "text" and content.get("text"):
                try:
                    return json.loads(content["text"])
                except json.JSONDecodeError:
                    pass
            return result if isinstance(result, dict) else {}

    async def _harvest_anchor(self) -> None:
        """While the binding is healthy, capture {uuid, continuity_token} from
        identity() so a future store-wipe is recoverable. Cheap, best-effort,
        refreshed daily so the stored token stays recent."""
        fresh = (
            self._anchor is not None
            and (time.time() - float(self._anchor.get("saved_at", 0))) < self._anchor_refresh_s
        )
        if fresh:
            return
        try:
            resp = await self._call_identity_tool(
                {"client_session_id": self._client_session_id()}
            )
            uuid = resp.get("uuid") or (resp.get("bound_identity") or {}).get("uuid")
            token = resp.get("continuity_token")
            if uuid and token:
                self._save_anchor(uuid, token, resp.get("client_session_id"))
            else:
                logger.debug("Anchor harvest: identity() returned no uuid/token")
        except Exception as e:
            logger.debug("Anchor harvest failed (non-fatal): %s", e)

    async def _try_reanchor(self) -> bool:
        """Spend the stored anchor on a resume=true rebind after an identity
        refusal. Rate-limited; returns True when the server re-bound us."""
        # Re-read the file first: broker and server share the anchor, and a
        # sibling process may have already healed + rotated it.
        self._anchor = self._load_anchor() or self._anchor
        if not self._anchor:
            logger.error(
                "Governance identity refused and NO anchor at %s — cannot "
                "self-recover; operator recovery required (anima-mcp #97)",
                self._anchor_path,
            )
            return False
        now = time.time()
        if (now - self._reanchor_last_attempt) < self._reanchor_cooldown_s:
            return False
        self._reanchor_last_attempt = now
        try:
            resp = await self._call_identity_tool({
                "agent_uuid": self._anchor["uuid"],
                "continuity_token": self._anchor["continuity_token"],
                "client_session_id": self._client_session_id(),
                "resume": True,
            })
            got = resp.get("uuid") or (resp.get("bound_identity") or {}).get("uuid")
            if got == self._anchor["uuid"]:
                # Acceptance-testing #97 (2026-07-03) bottomed out the ontology:
                # identity(resume) RESOLVES this call but never WRITES a
                # transport binding, and no sanctioned call can (S1-c retired
                # cross-process token resume; bind_session is fail-closed on
                # unknown keys — by design, these gates are the F3 fix).
                # Binding durability is server-side: the PG session row renews
                # +24h on every check-in, so a Redis wipe self-heals via PATH2
                # with no client action. Reaching THIS code means the PG row is
                # gone too (>24h outage or DB loss) — that is operator
                # territory, and the only honest move is to say so loudly.
                bound_key = resp.get("client_session_id") or self._client_session_id()
                self._save_anchor(
                    got, resp.get("continuity_token") or self._anchor["continuity_token"],
                    bound_key,
                )
                logger.error(
                    "Governance identity VERIFIED (uuid=%s) but the session "
                    "binding is gone from both stores — a client cannot "
                    "recreate it by design. OPERATOR RECOVERY REQUIRED: run "
                    "unitares scripts/ops/rebind-resident-session.sh %s %s",
                    got[:8], got, bound_key,
                )
                return True
            logger.error("Re-anchor resolved to %s, expected %s — refusing mismatched binding",
                         (got or "nothing")[:8], self._anchor["uuid"][:8])
            return False
        except Exception as e:
            logger.error("Re-anchor attempt failed: %s", e)
            return False

    async def check_in(
        self,
        anima: Anima,
        readings: SensorReadings,
        neural_weight: float = 0.3,
        physical_weight: float = 0.7,
        identity: Optional['CreatureIdentity'] = None,
        is_first_check_in: bool = False,
        drawing_eisv: Optional[Dict[str, Any]] = None,
        experiential_summary: Optional[Dict[str, Any]] = None,
        light_attribution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check in with UNITARES governance.

        Projects anima state into body telemetry and requests a governance decision.

        Args:
            anima: Anima state
            readings: Sensor readings (physical + neural)
            neural_weight: Weight for neural signals in EISV mapping
            physical_weight: Weight for physical signals in EISV mapping
            identity: Optional CreatureIdentity for metadata sync
            is_first_check_in: If True, syncs identity metadata to UNITARES
            drawing_eisv: Optional DrawingEISV state from ScreenRenderer (None when not drawing)
            light_attribution: Provenance-bearing LED/lux attribution snapshot

        Returns:
            Governance decision dict with:
            - action: "proceed" | "pause" | "halt"
            - margin: "comfortable" | "tight" | "critical"
            - reason: Human-readable explanation
            - body_eisv_projection: Lumen's sensor-derived input projection
            - governance_eisv: UNITARES's inferred state, or None if omitted
            - eisv: deprecated alias of body_eisv_projection
            - source: "unitares" | "local" (which governance system responded)
        """
        logger.debug("check_in called: is_first_check_in=%s, identity=%s", is_first_check_in, identity is not None)

        # Project the body first (always needed). This is UNITARES input, not
        # UNITARES's inferred governance state.
        body_projection = anima_to_body_eisv_projection(
            anima, readings, neural_weight, physical_weight
        )

        # Check if UNITARES is available BEFORE trying to sync
        unitares_available = await self.check_availability()

        # Sync identity metadata on first check-in (only if UNITARES is available)
        if is_first_check_in and identity and unitares_available:
            logger.info("First check-in - syncing identity for %s", identity.name if hasattr(identity, 'name') else 'unknown')
            try:
                await self.sync_identity_metadata(identity)
            except Exception as e:
                # Non-fatal - continue with governance check-in
                logger.warning("Identity sync exception: %s", e)

        # Check if UNITARES is available
        if unitares_available:
            try:
                logger.info("Calling UNITARES (agent_id=%s)", self._agent_id[:8] if self._agent_id else 'None')
                result = await self._call_unitares(
                    anima,
                    readings,
                    body_projection,
                    identity=identity,
                    drawing_eisv=drawing_eisv,
                    experiential_summary=experiential_summary,
                    light_attribution=light_attribution,
                )
                logger.info("UNITARES responded: %s", result.get('source', 'unknown'))
                self._circuit_failures = 0  # Success resets circuit
                self._circuit_current_backoff = self._circuit_backoff_base
                # Healthy binding — keep the recovery anchor fresh (#97).
                await self._harvest_anchor()
                return result
            except IdentityRefusedError as e:
                # Server-side binding lost (e.g. session-store wipe). Spend the
                # anchor on a resume=true rebind, then retry ONCE this cycle.
                logger.warning("Identity refused — attempting re-anchor: %s", e)
                if await self._try_reanchor():
                    try:
                        result = await self._call_unitares(
                            anima,
                            readings,
                            body_projection,
                            identity=identity,
                            drawing_eisv=drawing_eisv,
                            experiential_summary=experiential_summary,
                            light_attribution=light_attribution,
                        )
                        self._circuit_failures = 0
                        self._circuit_current_backoff = self._circuit_backoff_base
                        return result
                    except Exception as retry_err:
                        logger.warning("Post-re-anchor retry failed: %s", retry_err)
                self._circuit_failures += 1
                self._maybe_open_circuit(time.time())
                return self._local_governance(
                    anima, readings, body_projection, error=str(e)
                )
            except Exception as e:
                # Fallback to local governance on error
                logger.warning("UNITARES error, falling back to local: %s", e)
                self._circuit_failures += 1
                self._maybe_open_circuit(time.time())
                # Rebuild the session so the next check-in re-resolves DNS (self-heal)
                await self._reset_session()
                return self._local_governance(
                    anima, readings, body_projection, error=str(e)
                )
        else:
            # Use local governance
            logger.debug("UNITARES not available, using local governance")
            return self._local_governance(anima, readings, body_projection)
    
    async def _call_unitares(
        self,
        anima: Anima,
        readings: SensorReadings,
        body_projection: BodyEISVProjection,
        identity: Optional['CreatureIdentity'] = None,
        drawing_eisv: Optional[Dict[str, Any]] = None,
        experiential_summary: Optional[Dict[str, Any]] = None,
        light_attribution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call UNITARES governance via HTTP/SSE."""
        try:
            # Prepare MCP request
            complexity = estimate_complexity(anima, readings)
            status_text = generate_status_text(
                anima,
                readings,
                body_projection,
                experiential_summary=experiential_summary,
            )
            
            # Build sensor_data payload — include raw sensors for dashboard visibility
            body_anima = {
                    "warmth": anima.warmth,
                    "clarity": anima.clarity,
                    "stability": anima.stability,
                    "presence": anima.presence,
            }
            body_vector = body_projection.to_dict()
            from .light_attribution import gated_external_light_lux

            raw_light_lux = getattr(readings, 'light_lux', None)
            external_light_lux = gated_external_light_lux(light_attribution)
            sensor_data = {
                "body_eisv_projection": body_vector,
                "body_anima": body_anima,
                # Legacy aliases retained for UNITARES deployments that have
                # not adopted the provenance-aware sensor schema yet.
                "eisv": body_vector,
                "anima": body_anima,
                "state_space_provenance": {
                    "body_anima": {
                        "source": "broker_published_anima",
                        "role": "physical_self_sense",
                    },
                    "anima": {
                        "alias_of": "body_anima",
                        "deprecated": True,
                    },
                    "body_eisv_projection": {
                        "schema": BODY_EISV_PROJECTION_SCHEMA,
                        "source": "anima_sensor_projection",
                        "role": "governance_input_measurement",
                    },
                    "eisv": {
                        "alias_of": "body_eisv_projection",
                        "deprecated": True,
                    },
                },
                "environment": {
                    "cpu_temp_c": getattr(readings, 'cpu_temp_c', None),
                    "ambient_temp_c": getattr(readings, 'ambient_temp_c', None),
                    "humidity_pct": getattr(readings, 'humidity_pct', None),
                    "light_lux": raw_light_lux,
                    "raw_light_lux": raw_light_lux,
                    "light_lux_composition": "room_light_plus_dotstar_glow",
                    "external_light_lux": external_light_lux,
                    "light_attribution_status": (
                        light_attribution.get("status")
                        if isinstance(light_attribution, dict)
                        else "unavailable"
                    ),
                    "cpu_percent": getattr(readings, 'cpu_percent', None),
                    "memory_percent": getattr(readings, 'memory_percent', None),
                },
            }

            # Include identity metadata if available
            if identity:
                sensor_data["identity"] = {
                    "total_awakenings": identity.total_awakenings if hasattr(identity, 'total_awakenings') else 0,
                    "total_alive_seconds": identity.total_alive_seconds if hasattr(identity, 'total_alive_seconds') else 0.0,
                    "alive_ratio": identity.alive_ratio() if hasattr(identity, 'alive_ratio') else 0.0,
                    "age_seconds": identity.age_seconds() if hasattr(identity, 'age_seconds') else 0.0,
                }

            # Include DrawingEISV if Lumen is actively drawing
            if drawing_eisv:
                sensor_data["drawing_eisv"] = drawing_eisv

            # Include experiential accumulation summary
            if experiential_summary:
                sensor_data["experiential"] = experiential_summary

            # Compute ethical drift from state changes between check-ins
            ethical_drift = compute_ethical_drift(
                anima, self._prev_anima,
                readings, self._prev_readings,
            )
            # Compute confidence from current state + transition rate
            confidence = compute_confidence(anima, readings, self._prev_anima)

            # Store current state snapshot for next check-in's delta computation
            self._prev_anima = _AnimaSnapshot(anima.warmth, anima.clarity, anima.stability, anima.presence)
            self._prev_readings = readings
            self._prev_complexity = complexity

            # Build arguments for process_agent_update
            # client_session_id is the #1 priority for identity resolution in UNITARES,
            # ensuring stable binding across service restarts regardless of HTTP fingerprint
            update_arguments = {
                "client_session_id": self._client_session_id(),
                "agent_name": "Lumen",  # cosmetic display label only — server-side name-claim recovery was removed (no-lookup-by-label invariant)
                "complexity": complexity,
                "confidence": confidence,
                "ethical_drift": ethical_drift,
                "response_text": status_text,
                "sensor_data": sensor_data,
                # Broker only reads action+margin from the response — opt out of
                # response-shaping enrichments (knowledge_surfacing, learning_context,
                # mirror_signals) the broker discards anyway. Gate is in unitares
                # PR #347 (lite_safe).
                "response_mode": "minimal",
            }

            # Add trajectory signature if available (enables lineage tracking in UNITARES)
            try:
                from .trajectory import compute_trajectory_signature
                from .anima_history import get_anima_history
                from .self_model import get_self_model
                # Note: growth_system is global _growth in server.py, we get it if available
                growth_system = None
                try:
                    from . import _growth
                    growth_system = _growth
                except (ImportError, AttributeError):
                    pass

                trajectory_sig = compute_trajectory_signature(
                    growth_system=growth_system,
                    self_model=get_self_model(),
                    anima_history=get_anima_history(),
                )
                if trajectory_sig and trajectory_sig.observation_count > 0:
                    sig_dict = trajectory_sig.to_dict()
                    # Add identity_confidence for UNITARES
                    sig_dict["identity_confidence"] = getattr(trajectory_sig, 'identity_confidence', 0.0)
                    update_arguments["trajectory_signature"] = sig_dict
                    logger.debug("Including trajectory (obs=%d, conf=%.2f)", trajectory_sig.observation_count, sig_dict.get('identity_confidence', 0))
            except Exception as e:
                # Non-blocking - trajectory is optional enhancement
                logger.debug("Trajectory not available: %s", e)

            # MCP JSON-RPC request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "sync_state",  # advertised alias; raw twin dropped from wire (unitares c737b24c)
                    "arguments": update_arguments
                }
            }
            
            # Determine endpoint URL
            mcp_url = self._get_mcp_url()
            
            # Build headers with identity for proper UNITARES binding
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",  # Required by MCP SSE servers
                "X-Session-ID": self._session_id or "anima-creature"
            }
            # Add agent ID header if set (for proper identity binding in UNITARES)
            if self._agent_id:
                headers["X-Agent-Id"] = self._agent_id

            # Use shared session for connection pooling
            session = await self._get_session()
            async with session.post(
                mcp_url,
                json=mcp_request,
                headers=headers
            ) as response:
                if response.status == 200:
                    # Handle SSE or JSON response format
                    content_type = response.headers.get("Content-Type", "")
                    text = await response.text()
                    result = self._parse_mcp_response(text, content_type)
                    if not result:
                        raise Exception("No valid JSON data in SSE response")

                    # Parse MCP response
                    if "result" in result:
                        governance_result = result["result"]

                        # Check MCP-level error flag (tool returned isError)
                        if governance_result.get("isError"):
                            error_text = "unknown error"
                            if "content" in governance_result and governance_result["content"]:
                                c = governance_result["content"][0]
                                error_text = c.get("text", error_text)
                            raise Exception(f"UNITARES rejected check-in: {error_text}")

                        # MCP wraps tool results in content[0]["text"] as JSON string
                        if "content" in governance_result and governance_result["content"]:
                            content = governance_result["content"][0]
                            if content.get("type") == "text" and content.get("text"):
                                try:
                                    governance_result = json.loads(content["text"])
                                except json.JSONDecodeError:
                                    pass  # Keep original if not JSON

                        # Typed identity refusal (STRICT_IDENTITY_REQUIRED). This
                        # payload carries NEITHER success:false NOR an action, so
                        # without this check it parses as a silent default
                        # "proceed" — exactly how the 2026-06-30→07-02 outage
                        # stayed invisible bridge-side (anima-mcp #97).
                        if _is_identity_refusal(governance_result):
                            raise IdentityRefusedError(
                                f"UNITARES refused identity for {self._client_session_id()}: "
                                f"{governance_result.get('hint') or governance_result.get('error') or 'session binding unresolved'}"
                            )

                        # Check application-level error (success: false)
                        if governance_result.get("success") is False:
                            error_code = governance_result.get("error_code", "UNKNOWN")
                            error_msg = governance_result.get("error") or governance_result.get("reason") or "update rejected"
                            logger.warning("UNITARES check-in rejected: code=%s msg=%s", error_code, error_msg)
                            raise Exception(f"UNITARES check-in failed [{error_code}]: {error_msg}")

                        logger.debug("Response keys: %s", list(governance_result.keys()))
                        # Log agent binding info from UNITARES
                        bound_id = governance_result.get("resolved_agent_id") or governance_result.get("agent_signature", {}).get("agent_id") or governance_result.get("agent_signature", {}).get("uuid")
                        logger.debug("Bound to agent: %s", bound_id[:8] if bound_id else 'not specified')

                        # Extract action and margin from UNITARES response
                        # UNITARES returns: {"action": "proceed", "margin": "comfortable", ...}
                        return {
                            "action": governance_result.get("action", "proceed"),
                            "margin": governance_result.get("margin", "comfortable"),
                            "reason": governance_result.get("reason", "Governance check completed"),
                            "source": "unitares",
                            "unitares_agent_id": bound_id,  # For display identification
                            "raw_response": governance_result,
                            **_state_space_fields(
                                body_projection, governance_result
                            ),
                        }
                    elif "error" in result:
                        raise Exception(f"MCP error: {result['error']}")
                else:
                    # HTTP error - fallback to local
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                        
        except ImportError:
            # aiohttp not available
            raise Exception("aiohttp not installed - cannot connect to UNITARES")
        except asyncio.TimeoutError:
            raise Exception("Timeout connecting to UNITARES server")
        except IdentityRefusedError:
            raise  # typed — check_in attempts the re-anchor rescue
        except Exception as e:
            raise Exception(f"Error calling UNITARES: {e}")
    
    def _local_governance(
        self,
        anima: Anima,
        readings: SensorReadings,
        body_projection: BodyEISVProjection,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Local governance decision (fallback if UNITARES unavailable).
        
        Uses simple thresholds based on the body EISV projection.
        """
        # Compute signed margins (positive = within bounds, negative = crossed)
        # UNITARES thresholds (from governance_config.py):
        RISK_THRESHOLD = 0.60
        COHERENCE_THRESHOLD = 0.40

        # Signed margins: positive = room to threshold, negative = past threshold.
        # V (Valence) is telemetry, not a gate — it carries little outcome signal
        # and gating on it homogenises agents, so the local fallback no longer
        # pauses on V (reported in eisv for observability only).
        margins = {
            "risk": RISK_THRESHOLD - body_projection.entropy,
            "coherence": body_projection.integrity - COHERENCE_THRESHOLD,
        }

        # Check if any threshold crossed
        crossed = {k: v for k, v in margins.items() if v < 0}
        valid = {k: v for k, v in margins.items() if v >= 0}

        if crossed:
            # At least one threshold crossed
            worst_edge = min(crossed.items(), key=lambda x: x[1])[0]
            distance_past = abs(crossed[worst_edge])

            # warning: just crossed (< 0.1 past), critical: deep past (>= 0.1)
            if distance_past >= 0.1:
                margin = "critical"
            else:
                margin = "warning"

            action = "pause"
            reason = f"Crossed {worst_edge} threshold by {distance_past:.2f}"
            nearest_edge = worst_edge
        else:
            # All within bounds - find nearest edge
            nearest_edge = min(valid.items(), key=lambda x: x[1])[0]
            distance_to = valid[nearest_edge]

            # comfortable: > 0.15 from edge, tight: <= 0.15
            if distance_to > 0.15:
                margin = "comfortable"
            else:
                margin = "tight"

            action = "proceed"
            reason = f"State healthy (margin: {margin})"
        
        if error:
            reason += f" [UNITARES unavailable: {error}]"
        
        return {
            "action": action,
            "margin": margin,
            "reason": reason,
            "source": "local",
            "nearest_edge": nearest_edge,
            **_state_space_fields(body_projection),
        }
    
    def set_agent_id(self, agent_id: str):
        """Set agent ID for UNITARES."""
        self._agent_id = agent_id
    
    def set_session_id(self, session_id: str):
        """Set session ID for UNITARES connection."""
        self._session_id = session_id

    async def resolve_caller_identity(self, session_id: Optional[str] = None) -> Optional[str]:
        """Resolve caller's verified display_name from UNITARES.

        Uses the ``identity()`` tool to look up who the current session
        belongs to, returning their display label if found.

        Args:
            session_id: Optional session ID to resolve. Uses bridge's
                        session ID if not provided.

        Returns:
            Verified display name string, or None if unavailable.
        """
        if not self._url or self._available is False:
            return None  # Skip when UNITARES known unavailable

        sid = session_id or self._session_id
        if not sid:
            return None

        try:
            import aiohttp
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "identity",
                    "arguments": {}
                }
            }

            mcp_url = self._get_mcp_url()
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "X-Session-ID": sid,
            }

            session = await self._get_session()
            async with session.post(
                mcp_url,
                json=mcp_request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as response:
                if response.status != 200:
                    return None
                content_type = response.headers.get("Content-Type", "")
                text = await response.text()
                result = self._parse_mcp_response(text, content_type)
                if not result or "result" not in result:
                    return None

                # MCP wraps in content[0]["text"]
                content = result["result"].get("content", [])
                if content and content[0].get("type") == "text":
                    try:
                        identity_data = json.loads(content[0]["text"])
                    except (json.JSONDecodeError, KeyError):
                        return None
                    return identity_data.get("display_name") or identity_data.get("label")

        except Exception as e:
            logger.debug("resolve_caller_identity failed: %s", e)
        return None

    async def sync_name(self, name: str) -> bool:
        """
        Sync Lumen's name to UNITARES label.
        
        Args:
            name: Lumen's chosen name
            
        Returns:
            True if synced successfully, False otherwise
        """
        if not self._url or not self._agent_id:
            return False
        
        try:
            # Call UNITARES identity tool to set label
            # Note: agent(action="update") doesn't set label directly
            # We need to use identity(name=...) tool instead
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "identity",
                    "arguments": {
                        "client_session_id": self._client_session_id(),
                        "name": name
                    }
                }
            }

            mcp_url = self._get_mcp_url()
            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": self._session_id or "anima-creature"
            }
            if self._agent_id:
                headers["X-Agent-Id"] = self._agent_id

            # Use shared session for connection pooling
            session = await self._get_session()
            async with session.post(mcp_url, json=mcp_request, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    text = await response.text()
                    result = self._parse_mcp_response(text, content_type)
                    return result is not None and "result" in result and "error" not in result
            return False
        except Exception:
            # Non-fatal - name sync is optional
            return False
    
    async def sync_identity_metadata(self, identity: 'CreatureIdentity') -> bool:
        """
        Sync Lumen's identity summary into its UNITARES agent notes.

        Called on first check-in so the governance record carries birth date
        and awakening count.

        Writes ``notes`` only. On ``/mcp/``, ``agent(action="update")`` accepts
        just ``tags`` and ``notes``: the SDK validates arguments against the
        advertised schema and silently drops ``purpose`` and ``preferences``
        before the handler sees them. ``tags`` is deliberately not sent — the
        handler REPLACES the tag list, and Lumen's row holds server-granted
        tags (``pinned``, ``pioneer``, ``persistent``) that gate archival
        immunity and delete refusal. The old payload listed five tags, so the
        first sync to actually land would have stripped the rest.

        Args:
            identity: CreatureIdentity object

        Returns:
            True if synced successfully, False otherwise
        """
        if not self._url or not self._agent_id:
            return False

        try:
            born_at = identity.born_at.isoformat() if hasattr(identity, 'born_at') else None
            awakenings = identity.total_awakenings if hasattr(identity, 'total_awakenings') else 0

            # Get creature name for labeling
            creature_name = identity.name if hasattr(identity, 'name') and identity.name else "Anima"
            creature_id = identity.creature_id if hasattr(identity, 'creature_id') else "unknown"

            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    # update_agent_metadata is a pre-consolidation name that
                    # /mcp/ does not resolve ("Unknown tool", isError).
                    "name": "agent",
                    "arguments": {
                        "action": "update",
                        # client_session_id ensures stable identity binding across restarts
                        "client_session_id": self._client_session_id(),
                        "notes": f"{creature_name} identity: creature_id={creature_id}, born={born_at}, awakenings={awakenings}"
                    }
                }
            }

            mcp_url = self._get_mcp_url()
            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": self._session_id or "anima-creature",
                "Accept": "application/json, text/event-stream"
            }
            if self._agent_id:
                headers["X-Agent-Id"] = self._agent_id

            logger.info("Syncing identity metadata for %s", creature_name)

            # Use shared session for connection pooling
            session = await self._get_session()
            async with session.post(mcp_url, json=mcp_request, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    text = await response.text()
                    result = self._parse_mcp_response(text, content_type)

                    refusal = mcp_tool_refusal(result)
                    if refusal is None:
                        logger.info("Identity sync SUCCESS - %s labeled in UNITARES", creature_name)
                        return True
                    logger.warning("Identity sync failed: %s", refusal)
                else:
                    logger.warning("Identity sync HTTP error: %d", response.status)
            return False
        except Exception as e:
            # Non-fatal - metadata sync is optional
            logger.warning("Identity sync error: %s", e)
            return False

    async def report_outcome(
        self,
        outcome_type: str,
        outcome_score: Optional[float] = None,
        is_bad: Optional[bool] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Report an outcome event to UNITARES for EISV validation.

        Fire-and-forget, non-blocking. Non-fatal on failure.

        Args:
            outcome_type: e.g. "drawing_completed", "task_completed"
            outcome_score: 0.0-1.0 quality metric
            is_bad: Whether negative outcome (inferred from type if None)
            detail: Type-specific metadata

        Returns:
            True if reported successfully, False otherwise
        """
        if not self._url or not self._agent_id:
            return False

        try:
            arguments = {
                "client_session_id": self._client_session_id(),
                "outcome_type": outcome_type,
            }
            if outcome_score is not None:
                arguments["outcome_score"] = outcome_score
            if is_bad is not None:
                arguments["is_bad"] = is_bad
            if detail:
                arguments["detail"] = detail

            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "record_result",  # advertised alias; raw twin dropped from wire (unitares c737b24c)
                    "arguments": arguments,
                }
            }

            mcp_url = self._get_mcp_url()
            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": self._session_id or "anima-creature",
                "Accept": "application/json, text/event-stream",
            }
            if self._agent_id:
                headers["X-Agent-Id"] = self._agent_id

            session = await self._get_session()
            async with session.post(mcp_url, json=mcp_request, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    text = await response.text()
                    result = self._parse_mcp_response(text, content_type)
                    if result and "result" in result and "error" not in result:
                        logger.info("Outcome reported: %s score=%.2f", outcome_type, outcome_score or 0)
                        return True
            return False
        except Exception as e:
            logger.debug("Outcome report failed (non-fatal): %s", e)
            return False


# Convenience function for common use case
async def check_governance(
    anima: Anima,
    readings: SensorReadings,
    unitares_url: Optional[str] = None,
    neural_weight: float = 0.3,
    physical_weight: float = 0.7
) -> Dict[str, Any]:
    """
    Convenience function to check governance.
    
    Args:
        anima: Anima state
        readings: Sensor readings
        unitares_url: Optional UNITARES server URL
        neural_weight: Weight for neural signals
        physical_weight: Weight for physical signals
    
    Returns:
        Governance decision dict
    """
    bridge = UnitaresBridge(unitares_url=unitares_url)
    return await bridge.check_in(anima, readings, neural_weight, physical_weight)
