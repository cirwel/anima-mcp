"""
Bridge between anima-mcp and unitares-governance MCP server.

Enables creature to check in with UNITARES governance system via HTTP/SSE.
Provides fallback local governance if UNITARES server is unavailable.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .identity.store import CreatureIdentity

from .eisv_mapper import (
    EISVMetrics,
    anima_to_eisv,
    estimate_complexity,
    generate_status_text,
    compute_ethical_drift,
    compute_confidence,
)
from .anima import Anima
from .sensors.base import SensorReadings

logger = logging.getLogger(__name__)


class _AnimaSnapshot:
    """Lightweight snapshot of anima state for delta computation between check-ins."""
    __slots__ = ('warmth', 'clarity', 'stability', 'presence')
    def __init__(self, w, c, s, p):
        self.warmth, self.clarity, self.stability, self.presence = w, c, s, p


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
        timeout: float = 5.0
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
        if '/mcp' in self._url:
            return self._url
        elif '/sse' in self._url:
            return self._url.replace('/sse', '/mcp')
        return f"{self._url}/mcp"

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
            health_url = self._url.replace('/sse', '/health') if '/sse' in self._url else f"{self._url}/health"
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
            return False

        except ImportError:
            # aiohttp not available
            self._available = False
            return False
        except Exception:
            self._available = False
            self._circuit_failures += 1
            self._maybe_open_circuit(time.time())
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
    ) -> Dict[str, Any]:
        """
        Check in with UNITARES governance.

        Maps anima state to EISV metrics and requests governance decision.

        Args:
            anima: Anima state
            readings: Sensor readings (physical + neural)
            neural_weight: Weight for neural signals in EISV mapping
            physical_weight: Weight for physical signals in EISV mapping
            identity: Optional CreatureIdentity for metadata sync
            is_first_check_in: If True, syncs identity metadata to UNITARES
            drawing_eisv: Optional DrawingEISV state from ScreenRenderer (None when not drawing)

        Returns:
            Governance decision dict with:
            - action: "proceed" | "pause" | "halt"
            - margin: "comfortable" | "tight" | "critical"
            - reason: Human-readable explanation
            - eisv: EISV metrics used
            - source: "unitares" | "local" (which governance system responded)
        """
        logger.debug("check_in called: is_first_check_in=%s, identity=%s", is_first_check_in, identity is not None)

        # Map anima to EISV first (always needed)
        eisv = anima_to_eisv(anima, readings, neural_weight, physical_weight)

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
                result = await self._call_unitares(anima, readings, eisv, identity=identity, drawing_eisv=drawing_eisv, experiential_summary=experiential_summary)
                logger.info("UNITARES responded: %s", result.get('source', 'unknown'))
                self._circuit_failures = 0  # Success resets circuit
                self._circuit_current_backoff = self._circuit_backoff_base
                return result
            except Exception as e:
                # Fallback to local governance on error
                logger.warning("UNITARES error, falling back to local: %s", e)
                import time
                self._circuit_failures += 1
                self._maybe_open_circuit(time.time())
                return self._local_governance(anima, readings, eisv, error=str(e))
        else:
            # Use local governance
            logger.debug("UNITARES not available, using local governance")
            return self._local_governance(anima, readings, eisv)
    
    async def _call_unitares(
        self,
        anima: Anima,
        readings: SensorReadings,
        eisv: EISVMetrics,
        identity: Optional['CreatureIdentity'] = None,
        drawing_eisv: Optional[Dict[str, Any]] = None,
        experiential_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call UNITARES governance via HTTP/SSE."""
        try:
            # Prepare MCP request
            complexity = estimate_complexity(anima, readings)
            status_text = generate_status_text(anima, readings, eisv, experiential_summary=experiential_summary)
            
            # Build sensor_data payload — include raw sensors for dashboard visibility
            sensor_data = {
                "eisv": eisv.to_dict(),
                "anima": {
                    "warmth": anima.warmth,
                    "clarity": anima.clarity,
                    "stability": anima.stability,
                    "presence": anima.presence,
                },
                "environment": {
                    "cpu_temp_c": getattr(readings, 'cpu_temp_c', None),
                    "ambient_temp_c": getattr(readings, 'ambient_temp_c', None),
                    "humidity_pct": getattr(readings, 'humidity_pct', None),
                    "light_lux": getattr(readings, 'light_lux', None),
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
                "client_session_id": f"lumen-{self._agent_id}" if self._agent_id else "lumen-anima",
                "agent_name": "Lumen",  # Enables name-claim identity recovery after session key change
                "complexity": complexity,
                "confidence": confidence,
                "ethical_drift": ethical_drift,
                "response_text": status_text,
                "sensor_data": sensor_data,
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
                    "name": "process_agent_update",
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
                            "eisv": eisv.to_dict(),
                            "source": "unitares",
                            "unitares_agent_id": bound_id,  # For display identification
                            "raw_response": governance_result
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
        except Exception as e:
            raise Exception(f"Error calling UNITARES: {e}")
    
    def _local_governance(
        self,
        anima: Anima,
        readings: SensorReadings,
        eisv: EISVMetrics,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Local governance decision (fallback if UNITARES unavailable).
        
        Uses simple thresholds based on EISV metrics.
        """
        # Compute signed margins (positive = within bounds, negative = crossed)
        # UNITARES thresholds (from governance_config.py):
        RISK_THRESHOLD = 0.60
        COHERENCE_THRESHOLD = 0.40
        VOID_THRESHOLD = 0.15

        # Signed margins: positive = room to threshold, negative = past threshold
        margins = {
            "risk": RISK_THRESHOLD - eisv.entropy,        # Higher entropy is worse
            "coherence": eisv.integrity - COHERENCE_THRESHOLD,  # Lower integrity is worse
            "void": VOID_THRESHOLD - eisv.void            # Higher void is worse
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
            "eisv": eisv.to_dict(),
            "source": "local",
            "nearest_edge": nearest_edge
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
            # Note: update_agent_metadata doesn't set label directly
            # We need to use identity(name=...) tool instead
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "identity",
                    "arguments": {
                        "client_session_id": f"lumen-{self._agent_id}" if self._agent_id else "lumen-anima",
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
        Sync Lumen's identity metadata to UNITARES.
        
        Includes birth date, runtime metrics, and name history.
        Called on first check-in to ensure UNITARES has full context.
        
        Args:
            identity: CreatureIdentity object
            
        Returns:
            True if synced successfully, False otherwise
        """
        if not self._url or not self._agent_id:
            return False
        
        try:
            # Build metadata payload
            metadata = {
                "born_at": identity.born_at.isoformat() if hasattr(identity, 'born_at') else None,
                "total_awakenings": identity.total_awakenings if hasattr(identity, 'total_awakenings') else 0,
                "total_alive_seconds": identity.total_alive_seconds if hasattr(identity, 'total_alive_seconds') else 0.0,
                "alive_ratio": identity.alive_ratio() if hasattr(identity, 'alive_ratio') else 0.0,
                "name_history": identity.name_history if hasattr(identity, 'name_history') else [],
                "current_awakening_at": identity.current_awakening_at.isoformat() if hasattr(identity, 'current_awakening_at') and identity.current_awakening_at else None,
            }

            # Get creature name for labeling
            creature_name = identity.name if hasattr(identity, 'name') and identity.name else "Anima"
            creature_id = identity.creature_id if hasattr(identity, 'creature_id') else "unknown"

            # Call UNITARES update_agent_metadata tool - label ourselves!
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "update_agent_metadata",
                    "arguments": {
                        # client_session_id ensures stable identity binding across restarts
                        "client_session_id": f"lumen-{self._agent_id}" if self._agent_id else "lumen-anima",
                        "purpose": f"{creature_name} - embodied digital creature (creature_id: {creature_id[:8]}...)",
                        "tags": [creature_name.lower(), "anima", "creature", "embodied", "autonomous"],
                        "preferences": metadata,
                        "notes": f"{creature_name} identity: creature_id={creature_id}, born={metadata.get('born_at')}, awakenings={metadata.get('total_awakenings')}"
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

                    if result and "result" in result and "error" not in result:
                        logger.info("Identity sync SUCCESS - %s labeled in UNITARES", creature_name)
                        return True
                    else:
                        error = result.get('error', 'unknown') if result else 'no response'
                        logger.warning("Identity sync failed: %s", error)
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
                "client_session_id": f"lumen-{self._agent_id}" if self._agent_id else "lumen-anima",
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
                    "name": "outcome_event",
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
