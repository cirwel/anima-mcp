"""
UNITARES Knowledge Graph Integration for Lumen.

Bridges Lumen's local insights to the shared UNITARES knowledge graph.
Allows Lumen to contribute discoveries that persist across sessions
and can be accessed by other agents.

Design principle: Local memory stays local (embodied, fast, personal).
UNITARES gets significant insights (shared, persistent, collective).
"""

import os
import asyncio
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime


# Track what we've already shared to avoid duplicates
_shared_insights: set = set()
_last_share_time: float = 0.0
MIN_SHARE_INTERVAL = 60.0  # Don't share more than once per minute

# Reusable session — avoids allocating a new ClientSession per call
_http_session: Optional[Any] = None
_session_loop: Optional[Any] = None


async def share_insight_to_unitares(
    insight: str,
    discovery_type: str = "insight",
    tags: Optional[list] = None,
    identity: Optional[Any] = None,
    client_session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Share a Lumen insight to the UNITARES knowledge graph.

    Args:
        insight: The insight text (e.g., "In evenings with warm temperature, I feel content")
        discovery_type: Type of discovery (insight, observation, pattern, note)
        tags: Optional tags for categorization
        identity: Optional CreatureIdentity for agent binding
        client_session_id: Lumen's governance binding key (the bridge's
            ``client_session_id()``). Required — UNITARES writes need a bound
            caller, and without one a store is either refused or recorded
            under an anonymous writer id, so an unattributed share is skipped.

    Returns:
        Result dict if successful, None if skipped or failed
    """
    import time
    global _shared_insights, _last_share_time

    # Get UNITARES URL
    unitares_url = os.environ.get("UNITARES_URL")
    if not unitares_url:
        return None

    # Deduplication: Don't share the same insight twice
    insight_hash = hash(insight)
    if insight_hash in _shared_insights:
        return None

    # Rate limiting: Don't flood UNITARES
    now = time.time()
    if now - _last_share_time < MIN_SHARE_INTERVAL:
        return None

    if not client_session_id:
        print("[UNITARES Knowledge] Share skipped: no client_session_id to attribute it to",
              file=sys.stderr, flush=True)
        return None

    try:
        import aiohttp
        
        # Get agent ID from identity or environment
        agent_id = None
        if identity and hasattr(identity, 'creature_id'):
            agent_id = identity.creature_id
        else:
            agent_id = os.environ.get("ANIMA_ID")
        
        if not agent_id:
            return None
        
        # Build tags
        final_tags = ["lumen", "embodied", "autonomous"]
        if tags:
            final_tags.extend(tags)
        
        # store_knowledge_graph is a pre-consolidation name that /mcp/ does
        # not resolve ("Unknown tool", isError) — every share was refused.
        # `details` is the canonical field; `content` only reached it through
        # a parameter alias.
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "knowledge",
                "arguments": {
                    "action": "store",
                    "client_session_id": client_session_id,
                    "summary": insight,
                    "discovery_type": discovery_type,
                    "tags": final_tags,
                    "details": json.dumps({
                        "source": "lumen_autonomous",
                        "timestamp": datetime.now().isoformat(),
                    })
                }
            }
        }
        
        # Determine endpoint URL
        if '/mcp' in unitares_url:
            mcp_url = unitares_url
        elif '/sse' in unitares_url:
            mcp_url = unitares_url.replace('/sse', '/mcp')
        else:
            mcp_url = f"{unitares_url}/mcp"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Agent-Id": agent_id,
            "X-Session-ID": f"anima-{agent_id[:8]}"
        }
        
        global _http_session, _session_loop

        # Reuse session if still valid for this event loop
        current_loop = asyncio.get_running_loop()
        if (_http_session is None
                or _http_session.closed
                or _session_loop is not current_loop):
            if _http_session is not None and not _http_session.closed:
                await _http_session.close()
            _http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0),
            )
            _session_loop = current_loop

        from .unitares_bridge import UnitaresBridge, mcp_tool_refusal

        async with _http_session.post(mcp_url, json=mcp_request, headers=headers) as response:
            if response.status != 200:
                print(f"[UNITARES Knowledge] Share failed: HTTP {response.status}",
                      file=sys.stderr, flush=True)
                return None
            content_type = response.headers.get("Content-Type", "")
            result = UnitaresBridge._parse_mcp_response(await response.text(), content_type)

        # A refusal is not a share: leave the insight unrecorded so it can be
        # offered again, and say why instead of returning a refusal as a result.
        refusal = mcp_tool_refusal(result)
        if refusal is not None:
            print(f"[UNITARES Knowledge] Share refused: {refusal}", file=sys.stderr, flush=True)
            return None

        _shared_insights.add(insight_hash)
        _last_share_time = now
        # Keep set bounded
        if len(_shared_insights) > 1000:
            _shared_insights.clear()
        return result["result"]

    except ImportError:
        # aiohttp not available
        return None
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        # Log but don't crash - this is optional
        print(f"[UNITARES Knowledge] Share error (non-fatal): {e}", file=sys.stderr, flush=True)
        return None


def _close_shared_session_if_owned(loop: asyncio.AbstractEventLoop) -> None:
    """Close the cached aiohttp session if it belongs to the given loop."""
    global _http_session, _session_loop
    if _session_loop is not loop:
        return
    try:
        if _http_session is not None and not _http_session.closed:
            loop.run_until_complete(_http_session.close())
    finally:
        _http_session = None
        _session_loop = None


def share_insight_sync(
    insight: str,
    discovery_type: str = "insight",
    tags: Optional[list] = None,
    identity: Optional[Any] = None,
    client_session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for sharing insights.

    Safe to call from sync code - creates event loop if needed.
    """
    try:
        # Try to get existing loop
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - schedule as task
            asyncio.create_task(share_insight_to_unitares(
                insight, discovery_type, tags, identity, client_session_id=client_session_id
            ))
            return None  # Can't wait for result in this case
        except RuntimeError:
            # No running loop - create one
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    asyncio.wait_for(
                        share_insight_to_unitares(
                            insight, discovery_type, tags, identity,
                            client_session_id=client_session_id,
                        ),
                        timeout=5.0
                    )
                )
            finally:
                _close_shared_session_if_owned(loop)
                # Drain pending callbacks (aiohttp connector cleanup)
                loop.run_until_complete(asyncio.sleep(0))
                loop.close()
    except Exception:
        return None


def should_share_insight(text: str) -> bool:
    """
    Determine if an insight is significant enough to share to UNITARES.
    
    Filters out routine observations, keeping only meaningful discoveries.
    """
    # Too short - probably not meaningful
    if len(text) < 20:
        return False
    
    # Keywords that suggest significant insight
    significant_keywords = [
        "learned", "discovered", "noticed", "pattern",
        "often feel", "tends to", "I've found",
        "interesting", "surprising", "unexpected",
        "when", "because", "relationship",
    ]
    
    text_lower = text.lower()
    
    # Check for significant keywords
    for keyword in significant_keywords:
        if keyword in text_lower:
            return True
    
    # Memory-based insights (from the advocate) are always significant
    if "I often feel" in text or "I want to explore" in text:
        return True
    
    return False
