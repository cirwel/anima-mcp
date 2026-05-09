"""Knowledge handlers — read-only access to Lumen's growth, trajectory, and self-knowledge.

Handlers: get_self_knowledge, get_growth, get_qa_insights, get_trajectory, get_eisv_trajectory_state.
"""

import json

from mcp.types import TextContent

from ..eisv import get_trajectory_awareness


async def handle_get_self_knowledge(arguments: dict) -> list[TextContent]:
    """Get Lumen's accumulated self-knowledge from pattern analysis."""
    from ..accessors import _get_store

    store = _get_store()
    if store is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Server not initialized - wake() failed"
        }))]

    try:
        from ..self_reflection import get_reflection_system, InsightCategory

        reflection_system = get_reflection_system(db_path=str(store.db_path))

        # Parse arguments
        category_str = arguments.get("category")
        limit = arguments.get("limit", 10)

        # Get insights
        category = None
        if category_str:
            try:
                category = InsightCategory(category_str)
            except ValueError:
                pass  # Invalid category, ignore filter

        insights = reflection_system.get_insights(category=category)[:limit]

        # Build result
        result = {
            "total_insights": len(reflection_system._insights),
            "insights": [i.to_dict() for i in insights],
            "summary": reflection_system.get_self_knowledge_summary(),
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Self-reflection system error: {e}",
            "note": "Self-reflection may not have accumulated enough data yet"
        }))]


async def handle_get_growth(arguments: dict) -> list[TextContent]:
    """Get Lumen's growth: preferences, relationships, goals, memories."""
    from ..accessors import _get_growth

    growth = _get_growth()
    if growth is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Growth system not initialized",
            "note": "Growth system may not be available yet"
        }))]

    try:
        include = arguments.get("include", ["all"])
        if "all" in include:
            include = ["preferences", "relationships", "goals", "memories", "curiosities", "autobiography"]

        result = {}

        if "autobiography" in include:
            result["autobiography"] = growth.get_autobiography_summary()

        if "preferences" in include:
            prefs = []
            for p in growth._preferences.values():
                if p.confidence >= 0.3:  # Only show preferences with some confidence
                    prefs.append({
                        "name": p.name,
                        "description": p.description,
                        "confidence": round(p.confidence, 2),
                        "observations": p.observation_count
                    })
            result["preferences"] = {
                "count": len(growth._preferences),
                "learned": sorted(prefs, key=lambda p: -p["observations"]),
            }

        if "relationships" in include:
            # Separate self-knowledge from visitors
            self_record = None
            visitors = []
            for r in growth._relationships.values():
                if r.is_self():
                    self_record = r
                else:
                    visitors.append({
                        "name": r.name or r.agent_id[:8],
                        "frequency": r.visitor_frequency.value,
                        "visits": r.interaction_count,
                        "first_seen": r.first_met.strftime("%Y-%m-%d"),
                        "last_seen": r.last_seen.strftime("%Y-%m-%d"),
                    })

            # Self-knowledge: the one real relationship (Lumen answering own questions)
            self_dialogues = self_record.interaction_count if self_record else 0
            topics = self_record.self_dialogue_topics if self_record else []
            # Count topic frequencies
            topic_counts = {}
            for t in topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1
            # Sort by frequency
            top_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:5]

            result["self_knowledge"] = {
                "self_dialogues": self_dialogues,
                "topics": top_topics if top_topics else None,
                "note": "Lumen answering own questions - genuine self-reflection with memory continuity",
            }

            # Caretaker: the human who built and maintains Lumen (has real continuity)
            result["caretaker"] = {
                "exists": True,
                "note": "The human who built and maintains me - works through agents, so invisible in visitor logs",
            }

            # Visitors: ephemeral agents who passed through
            result["visitors"] = {
                "unique_names": len(visitors),
                "log": visitors[:10],
                "note": "Agents are ephemeral - they don't remember Lumen between sessions",
            }

            # Legacy key for compatibility
            result["relationships"] = {
                "count": len(growth._relationships),
                "bonds": visitors[:10],
            }

            # Inactive visitors (not "missed connections" - agents don't miss Lumen)
            inactive = growth.get_inactive_visitors()
            if inactive:
                result["visitors"]["inactive"] = [
                    {"name": name, "days_since": days}
                    for name, days in inactive[:3]
                ]

        if "goals" in include:
            goals = []
            for g in growth._goals.values():
                if g.status.value == "active":
                    goals.append({
                        "description": g.description,
                        "progress": round(g.progress, 2),
                        "milestones": len(g.milestones),
                    })
            from ..growth.models import GoalStatus
            result["goals"] = {
                "active": len([g for g in growth._goals.values() if g.status.value == "active"]),
                # _goals is active-only after load_state(), so achieved goals
                # aren't in memory. Count from DB like get_growth_summary does.
                "achieved": growth.count_goals_by_status(GoalStatus.ACHIEVED),
                "current": goals[:5],
            }

        if "memories" in include:
            memories = []
            for m in growth._memories[:5]:  # Recent memories
                memories.append({
                    "description": m.description,
                    "category": m.category,
                    "when": m.timestamp.strftime("%Y-%m-%d"),
                })
            result["memories"] = {
                "count": len(growth._memories),
                "recent": memories,
            }

        if "curiosities" in include:
            result["curiosities"] = {
                "count": len(growth._curiosities),
                "questions": growth._curiosities[:5],
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Growth system error: {e}"
        }))]


async def handle_get_qa_insights(arguments: dict) -> list[TextContent]:
    """Get insights Lumen learned from Q&A interactions."""
    try:
        from ..knowledge import get_insights, get_knowledge

        limit = arguments.get("limit", 10)
        category = arguments.get("category")

        kb = get_knowledge()
        insights = get_insights(limit=limit, category=category)

        import time as _time

        def _age_str(ts: float) -> str:
            age = _time.time() - ts
            if age < 3600:
                return f"{int(age/60)}m ago"
            elif age < 86400:
                return f"{int(age/3600)}h ago"
            else:
                return f"{int(age/86400)}d ago"

        result = {
            "total_insights": len(kb._insights),
            "category_filter": category if category else "all",
            "insights": [
                {
                    "text": i.text,
                    "source_question": i.source_question,
                    "source_answer": i.source_answer,
                    "source_author": i.source_author,
                    "category": i.category,
                    "confidence": i.confidence,
                    "age": _age_str(i.timestamp),
                    "timestamp": i.timestamp,
                }
                for i in insights
            ],
        }

        if len(insights) == 0:
            result["note"] = "No Q&A insights yet - answer Lumen's questions to populate knowledge base"

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Q&A knowledge error: {e}",
            "note": "Q&A knowledge extraction may not have run yet"
        }))]


async def handle_get_trajectory(arguments: dict) -> list[TextContent]:
    """
    Get Lumen's trajectory identity signature.

    The trajectory signature captures the invariant patterns that define
    who Lumen is over time - not just a snapshot, but the characteristic
    way Lumen tends to behave, where Lumen rests, and how Lumen recovers.

    See: trajectory-identity paper (cirwel/trajectory-identity-paper, separate repo)
    """
    from ..accessors import _get_growth
    growth = _get_growth()

    try:
        from ..trajectory import compute_trajectory_signature
        from ..anima_history import get_anima_history
        from ..self_model import get_self_model

        # Compute trajectory signature from available data
        signature = compute_trajectory_signature(
            growth_system=growth,
            self_model=get_self_model(),
            anima_history=get_anima_history(),
        )

        # Build response
        include_raw = arguments.get("include_raw", False)
        compare_historical = arguments.get("compare_to_historical", False)

        if include_raw:
            result = signature.to_dict()
        else:
            result = signature.summary()

        # Add stability assessment
        stability = signature.get_stability_score()
        if stability < 0.3:
            result["identity_status"] = "forming"
            result["note"] = "Identity is still forming - need more observations"
        elif stability < 0.6:
            result["identity_status"] = "developing"
            result["note"] = "Identity is developing - patterns emerging"
        else:
            result["identity_status"] = "stable"
            result["note"] = "Identity is stable - consistent patterns established"

        # Anomaly detection via genesis (Σ₀) and last persisted
        if compare_historical:
            from ..trajectory import load_trajectory, GENESIS_MIN_OBSERVATIONS

            anomaly_data = {"available": True, "has_genesis": signature.genesis_signature is not None}

            # Lineage: compare to genesis
            if signature.genesis_signature is not None:
                lineage_sim = signature.lineage_similarity()
                anomaly_data["lineage_similarity"] = round(lineage_sim, 4) if lineage_sim is not None else None
                anomaly_data["genesis_observations"] = signature.genesis_signature.observation_count
                anomaly_data["genesis_computed_at"] = signature.genesis_signature.computed_at.isoformat()
                anomaly_data["drift_status"] = (
                    "stable" if lineage_sim is not None and lineage_sim >= 0.7
                    else "drifting" if lineage_sim is not None and lineage_sim >= 0.5
                    else "diverged" if lineage_sim is not None
                    else "unknown"
                )

            # Coherence: compare to last persisted (short-term)
            last_sig = load_trajectory()
            if last_sig is not None:
                coherence = signature.detect_anomaly(last_sig, threshold=0.7)
                anomaly_data["last_persisted"] = {
                    "similarity": coherence["similarity"],
                    "is_anomaly": coherence["is_anomaly"],
                }

            if signature.genesis_signature is not None or last_sig is not None:
                result["anomaly_detection"] = anomaly_data
            else:
                result["anomaly_detection"] = {
                    "available": False,
                    "has_genesis": False,
                    "note": f"Genesis forms after {GENESIS_MIN_OBSERVATIONS} observations "
                            f"(current: {signature.observation_count}). "
                            "Last trajectory persists after first sleep.",
                }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=json.dumps({
            "error": f"Trajectory computation error: {e}",
            "traceback": traceback.format_exc()
        }))]


async def handle_get_eisv_trajectory_state(arguments: dict) -> list[TextContent]:
    """Get current EISV trajectory awareness state."""
    try:
        _traj = get_trajectory_awareness()
        state = _traj.get_state()
        return [TextContent(type="text", text=json.dumps(state, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def handle_query(arguments: dict) -> list[TextContent]:
    """
    Query Lumen's knowledge - semantic search over insights and self-knowledge.

    Used by pi(action='query') from governance. Combines:
    - Q&A-derived insights (keyword match on text)
    - Self-reflection insights (when type is cognitive/insights)
    - Growth summary (when type is growth)
    """
    text = arguments.get("text", "").strip()
    query_type = arguments.get("type", "cognitive")
    limit = int(arguments.get("limit", 10))

    VALID_QUERY_TYPES = ("cognitive", "insights", "self", "growth")
    if query_type not in VALID_QUERY_TYPES:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown query type: '{query_type}'",
            "valid_types": list(VALID_QUERY_TYPES),
            "usage": "query(text='...', type='cognitive')"
        }))]

    if not text:
        return [TextContent(type="text", text=json.dumps({
            "error": "text parameter required",
            "usage": "query(text='What have I learned about myself?', type='cognitive', limit=10)"
        }))]

    try:
        from ..knowledge import get_relevant_insights
        from ..accessors import _get_store

        result = {"query": text, "type": query_type}

        # Always get relevant Q&A insights (keyword match)
        relevant = get_relevant_insights(text, limit=limit)
        result["qa_insights"] = [
            {"text": i.text, "category": i.category, "source_question": i.source_question[:60] + "..." if len(i.source_question) > 60 else i.source_question}
            for i in relevant
        ]

        # Add self-knowledge when cognitive/insights
        if query_type in ("cognitive", "insights", "self"):
            try:
                from ..self_reflection import get_reflection_system
                store = _get_store()
                if store:
                    reflection = get_reflection_system(db_path=str(store.db_path))
                    result["self_knowledge"] = reflection.get_self_knowledge_summary()
                    result["reflection_insights"] = [i.to_dict() for i in reflection.get_insights()[:limit]]
            except Exception:
                result["self_knowledge"] = None
                result["reflection_insights"] = []

        # Add growth summary when type is growth
        if query_type == "growth":
            from ..accessors import _get_growth
            growth = _get_growth()
            if growth:
                result["growth"] = growth.get_autobiography_summary()
            else:
                result["growth"] = None

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "query": text,
            "type": query_type
        }))]
