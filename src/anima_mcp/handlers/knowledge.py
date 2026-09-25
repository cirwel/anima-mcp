"""Knowledge handlers — read-only access to Lumen's growth, trajectory, and self-knowledge.

Handlers: get_self_knowledge, get_growth, get_qa_insights, get_trajectory, get_eisv_trajectory_state.
"""

import json
from datetime import datetime, timezone

from mcp.types import TextContent

from ..eisv import get_trajectory_awareness


def _insight_view(
    insight,
    growth=None,
    self_model=None,
    light_attribution=None,
) -> dict:
    """Attach source-specific epistemic labels without rewriting stored history."""
    view = insight.to_dict()
    insight_id = getattr(insight, "id", view.get("id", ""))
    evidence_supports = int(view.get("validation_count", 0) or 0)
    evidence_contradictions = int(view.get("contradiction_count", 0) or 0)
    evidence_support_label = "validation"
    evidence_contradiction_label = "challenge"
    pref_name = (
        insight_id.removeprefix("pref_")
        if insight_id.startswith("pref_")
        else insight_id
    )
    pref = None
    historical_preference_claim = False
    led_causal_model_ready = None
    if growth is not None:
        pref = growth._preferences.get(pref_name)
    is_preference_insight = insight_id.startswith("pref_") or (
        insight_id.startswith("drawing_") and pref is not None
    )

    if is_preference_insight:
        source_kind = "preference_observation"
        confidence_kind = "95% Wilson lower bound on signed evidence windows"
        evidence_basis = "independent preference evidence windows"
        evidence_support_label = "positive direction"
        evidence_contradiction_label = "negative direction"
        if pref is not None:
            evidence_supports = int(getattr(pref, "supporting_count", 0) or 0)
            evidence_contradictions = int(
                getattr(pref, "contradicting_count", 0) or 0
            )
            evidence_origin = getattr(
                pref, "evidence_origin", "legacy_unclassified"
            )
            historical_preference_claim = (
                evidence_origin == "retired_qa_claim_bridge_v1"
            )
            view.update({
                "confidence": pref.confidence,
                "sample_count": getattr(
                    pref, "independent_evidence_count", pref.observation_count
                ),
                "raw_observation_count": pref.observation_count,
                "supporting_count": evidence_supports,
                "contradicting_count": evidence_contradictions,
                "evidence_origin": evidence_origin,
            })
            if historical_preference_claim:
                description = view.get("description", "")
                for prefix in (
                    "i know this about myself: ",
                    "observational pattern: ",
                    "from q&a: ",
                ):
                    if description.startswith(prefix):
                        description = description.removeprefix(prefix)
                view.update({
                    "description": f"historical Q&A claim: {description}",
                    "record_role": "historical_claim",
                    "historical_as_of": pref.last_confirmed.isoformat(),
                    "current_state_authority": "none",
                })
            evidence_basis = {
                "legacy_hourly_reconstruction": (
                    "conservative hourly reconstruction of legacy broker ticks"
                ),
                "legacy_event_count": "legacy event episodes",
                "native_hourly_windows": "signed hourly state windows",
                "native_events": "distinct event episodes",
                "reset_external_light_gate_v2": (
                    "cold-started after raw/self-glow contamination; awaiting gated residual"
                ),
                "retired_qa_claim_bridge_v1": (
                    "retired textual Q&A claim; preserved for historical audit only"
                ),
            }.get(evidence_origin, evidence_basis)
            if (
                not historical_preference_claim
                and view["description"].startswith("i know this about myself: ")
            ):
                view["description"] = view["description"].replace(
                    "i know this about myself: ", "observational pattern: ", 1
                )
    elif insight_id.startswith("belief_"):
        source_kind = "self_model_belief"
        confidence_kind = "episode-bucketed belief confidence"
        evidence_basis = "supporting and contradicting belief episodes"
        evidence_support_label = "support"
        evidence_contradiction_label = "challenge"
        belief_id = insight_id.removeprefix("belief_")
        belief = None
        if self_model is not None:
            belief = (getattr(self_model, "beliefs", None) or {}).get(belief_id)
        if belief is not None:
            # Reflection rows preserve the wording and validation counters that
            # existed when they were written.  They are useful as an audit, but
            # they must not masquerade as the current belief after a reset or
            # evidence migration.  Keep the old projection explicitly nested
            # and render the live SelfBelief fields at the top level.
            stored_reflection_audit = {
                key: view.get(key)
                for key in (
                    "description",
                    "last_validated",
                    "validation_count",
                    "contradiction_count",
                    "strength",
                )
                if key in view
            }
            stored_reflection_audit.update({
                "record_role": "historical_projection",
                "current_state_authority": "none",
            })
            evidence_supports = int(getattr(belief, "supporting_count", 0) or 0)
            evidence_contradictions = int(
                getattr(belief, "contradicting_count", 0) or 0
            )
            live_confidence = float(getattr(belief, "confidence", 0.0))
            last_tested = getattr(belief, "last_tested", None)
            view.update({
                "description": (
                    f"Testable hypothesis: {getattr(belief, 'description', belief_id)}"
                ),
                "confidence": live_confidence,
                "belief_value": float(getattr(belief, "value", 0.5)),
                "sample_count": evidence_supports + evidence_contradictions,
                "supporting_count": evidence_supports,
                "contradicting_count": evidence_contradictions,
                "validation_count": evidence_supports,
                "contradiction_count": evidence_contradictions,
                "strength": live_confidence,
                "last_validated": (
                    last_tested.isoformat()
                    if isinstance(last_tested, datetime)
                    else None
                ),
                "stored_reflection_audit": stored_reflection_audit,
            })
        if insight_id == "belief_my_leds_affect_lux":
            stats = None
            stats_source = None
            if isinstance(light_attribution, dict):
                live_model = light_attribution.get("model")
                if isinstance(live_model, dict):
                    stats = live_model
                    stats_source = "broker_shm_live"
            if stats is None:
                stats_getter = getattr(
                    self_model, "get_light_attribution_model_stats", None
                )
                stats = stats_getter() if callable(stats_getter) else None
                if isinstance(stats, dict):
                    stats_source = "persisted_self_model_snapshot"
            if isinstance(stats, dict):
                led_causal_model_ready = bool(stats.get("ready"))
                try:
                    causal_confidence = float(stats.get("confidence", 0.0))
                except (TypeError, ValueError):
                    causal_confidence = 0.0
                view["causal_test"] = {
                    "source": stats_source,
                    "model_kind": stats.get("kind"),
                    "identification_status": stats.get(
                        "identification_status"
                    ),
                    "ready": led_causal_model_ready,
                    "confidence": causal_confidence,
                    "transition_count": stats.get("transition_count"),
                    "latest_transition_at_unix": stats.get(
                        "latest_transition_at_unix"
                    ),
                    "up_transitions": stats.get("up_transitions"),
                    "down_transitions": stats.get("down_transitions"),
                    "unknown_reasons": stats.get("unknown_reasons", []),
                }
                view["description"] = (
                    "Causally supported hypothesis: my own LEDs affect my light "
                    "sensor readings"
                    if led_causal_model_ready
                    else "Hypothesis under causal test: my own LEDs affect my "
                    "light sensor readings"
                )
                view["interpretation"] = (
                    "The raw closed-loop correlation pathway is retired. "
                    f"The causal breathing-pulse test is "
                    f"{stats.get('identification_status', 'unknown')} at "
                    f"{causal_confidence:.0%} confidence; "
                    "only a ready model may reinforce this belief."
                )
            else:
                view["description"] = (
                    "Unverified hypothesis: my own LEDs may affect my light "
                    "sensor readings"
                )
                view["interpretation"] = (
                    "Narrative hypothesis only. The separately gated "
                    "light_attribution residual determines whether self-glow can "
                    "be estimated; raw lux remains physical telemetry, not room light."
                )
        elif insight_id == "belief_light_warmth_correlation":
            view["interpretation"] = (
                "Historical raw-lux evidence was cold-started because it mixed "
                "room light with self-glow. New episodes are admitted only when "
                "the learned external-light residual is ready."
            )
    elif insight_id.startswith("qa_"):
        source_kind = "qa_claim"
        confidence_kind = "claim confidence with later validation/retraction"
        evidence_basis = "Q&A assertion plus independent re-derivations"
        view.update({
            "record_role": "historical_claim",
            "historical_as_of": view.get("discovered_at"),
            "current_state_authority": "none",
        })
    elif insight_id.startswith("trend_"):
        source_kind = "long_term_trend"
        confidence_kind = "summary-window heuristic, not a probability"
        evidence_basis = "distinct rest or daily summaries"
    else:
        source_kind = "state_association"
        confidence_kind = "association heuristic, not a probability"
        evidence_basis = "state-history pattern samples; observational, not causal"

    confidence = max(0.0, min(1.0, float(view.get("confidence", 0.0))))
    reported_samples = max(0, int(view.get("sample_count", 0) or 0))
    if source_kind == "preference_observation":
        evidence_majority = max(evidence_supports, evidence_contradictions)
        evidence_minority = min(evidence_supports, evidence_contradictions)
        actionability_evidence = evidence_supports + evidence_contradictions
        contested = (
            evidence_minority > 0
            and evidence_minority * 3 >= max(1, evidence_majority)
        )
    elif source_kind == "self_model_belief":
        actionability_evidence = evidence_supports + evidence_contradictions
        contested = (
            evidence_contradictions > 0
            and evidence_contradictions * 3 >= max(1, evidence_supports)
        )
    else:
        # Q&A `sample_count` is its source-reference count while validation and
        # contradiction counters represent later checks. Use the larger, not
        # their sum: they can refer to the same underlying claim episode.
        actionability_evidence = max(
            reported_samples,
            evidence_supports + evidence_contradictions,
        )
        contested = (
            evidence_contradictions > 0
            and evidence_contradictions * 3 >= max(1, evidence_supports)
        )
    minimum_evidence = {
        "preference_observation": 10,
        "self_model_belief": 5,
        "qa_claim": 3,
        "long_term_trend": 3,
        "state_association": 10,
    }[source_kind]
    if historical_preference_claim:
        actionability = "historical_claim"
        review_reason = (
            "This textual Q&A row was retired from preference learning; it "
            "has no current-state or behavioral authority"
        )
    elif source_kind == "qa_claim":
        actionability = "historical_claim"
        review_reason = (
            "Q&A re-derivations can strengthen a stored claim but do not "
            "revalidate it against Lumen's current wiring or telemetry"
        )
    elif (
        insight_id == "belief_my_leds_affect_lux"
        and led_causal_model_ready is False
    ):
        actionability = "review"
        review_reason = (
            "the causal breathing-pulse model has not established a stable "
            "positive LED-to-lux response"
        )
    elif contested:
        actionability = "review"
        review_reason = (
            "both preference directions have material evidence"
            if source_kind == "preference_observation"
            else "contradictions are material relative to supporting evidence"
        )
    elif actionability_evidence < minimum_evidence:
        actionability = "review"
        review_reason = (
            f"only {actionability_evidence} source evidence item(s); "
            f"{minimum_evidence} required for established status"
        )
    elif confidence < 0.8:
        actionability = "review"
        review_reason = "reported confidence is below the established threshold"
    else:
        actionability = "established"
        review_reason = ""

    view.update({
        "confidence_kind": confidence_kind,
        "evidence_basis": evidence_basis,
        "source_kind": source_kind,
        "causal_claim": False,
        "reported_uncertainty": round(1.0 - confidence, 3),
        "actionability": actionability,
        "review_reason": review_reason,
        "evidence_supporting_count": evidence_supports,
        "evidence_contradicting_count": evidence_contradictions,
        "evidence_support_label": evidence_support_label,
        "evidence_contradiction_label": evidence_contradiction_label,
        "actionability_evidence_count": actionability_evidence,
        "minimum_established_evidence": minimum_evidence,
    })
    return view


async def handle_get_self_knowledge(arguments: dict) -> list[TextContent]:
    """Get Lumen's accumulated self-knowledge from pattern analysis."""
    from ..accessors import _get_growth, _get_last_shm_data, _get_store

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

        active_insights = reflection_system.get_insights()
        insights = (
            reflection_system.get_insights(category=category)
            if category else active_insights
        )
        growth = _get_growth()
        try:
            from ..self_model import get_self_model

            self_model = get_self_model()
        except Exception:
            self_model = None
        shm_data = _get_last_shm_data() or {}
        light_attribution = shm_data.get("light_attribution")

        # The reflection store retains the historical wording/confidence that
        # existed when an insight was written.  `_insight_view` reconciles that
        # history with live preference and belief evidence (including resets),
        # so every part of this response must be built from the same views.
        # Otherwise `insights` can correctly call a claim cold-started while
        # `summary.strongest` still presents its pre-reset confidence as 1.0.
        active_view_pairs = [
            (
                insight,
                _insight_view(
                    insight,
                    growth,
                    self_model,
                    light_attribution,
                ),
            )
            for insight in active_insights
        ]
        active_views_by_id = {
            view.get("id", getattr(insight, "id", "")): view
            for insight, view in active_view_pairs
        }

        def current_view(insight):
            insight_id = getattr(insight, "id", "")
            view = active_views_by_id.get(insight_id)
            if view is not None:
                return view
            return _insight_view(
                insight,
                growth,
                self_model,
                light_attribution,
            )

        summary = reflection_system.get_self_knowledge_summary()
        if isinstance(summary, dict):
            # Preserve source-aware ordering within each authority tier. A
            # timestamped Q&A record never outranks live evidence merely
            # because its own claim score is high.
            actionability_order = {
                "established": 0,
                "review": 1,
                "historical_claim": 2,
            }
            ranked_pairs = sorted(
                active_view_pairs,
                key=lambda pair: actionability_order.get(
                    pair[1].get("actionability"), 1
                ),
            )
            by_category = {}
            for insight_category in InsightCategory:
                category_views = [
                    view
                    for insight, view in ranked_pairs
                    if getattr(insight, "category", None) == insight_category
                ]
                if category_views:
                    by_category[insight_category.value] = [
                        view.get("description", "") for view in category_views[:3]
                    ]
            summary = {
                **summary,
                "total_insights": len(active_insights),
                "strongest": [view for _, view in ranked_pairs[:3]],
                "by_category": by_category,
                "ranking_note": (
                    "Established evidence precedes review items, and historical "
                    "Q&A claims rank last. Confidence remains source-specific."
                ),
            }

        # Build result
        result = {
            "total_insights": len(active_insights),
            "insights": [current_view(i) for i in insights[:limit]],
            "summary": summary,
            "epistemic_note": (
                "Confidence fields have source-specific meanings; inspect "
                "confidence_kind and evidence_basis. None implies causality."
            ),
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
            from ..growth.models import preference_evidence_status

            prefs = []
            for p in growth._preferences.values():
                evidence_status = preference_evidence_status(p)
                pref_view = {
                    "name": p.name,
                    "description": p.description,
                    "confidence": round(p.confidence, 2),
                    "evidence_status": evidence_status,
                    "evidence_windows": getattr(
                        p, "independent_evidence_count", p.observation_count
                    ),
                    "raw_observation_calls": p.observation_count,
                    "supporting_windows": getattr(p, "supporting_count", None),
                    "contradicting_windows": getattr(p, "contradicting_count", None),
                    "positive_direction_windows": getattr(
                        p, "supporting_count", None
                    ),
                    "negative_direction_windows": getattr(
                        p, "contradicting_count", None
                    ),
                    "evidence_origin": getattr(
                        p, "evidence_origin", "legacy_unclassified"
                    ),
                }
                if evidence_status == "historical_claim":
                    pref_view.update({
                        "record_role": "historical_claim",
                        "historical_as_of": p.last_confirmed.isoformat(),
                        "current_state_authority": "none",
                    })
                prefs.append(pref_view)
            established = [
                pref for pref in prefs if pref["evidence_status"] == "established"
            ]
            cold_start = [
                pref
                for pref in prefs
                if pref["evidence_status"] == "tracked"
            ]
            review = [
                pref
                for pref in prefs
                if pref["evidence_status"] == "review"
            ]
            historical_claims = [
                pref
                for pref in prefs
                if pref["evidence_status"] == "historical_claim"
            ]

            def preference_order(pref):
                return -pref["confidence"], -pref["evidence_windows"]

            result["preferences"] = {
                "tracked_count": len(prefs),
                "established_count": len(established),
                "review_count": len(review),
                "cold_start_count": len(cold_start),
                "historical_claim_count": len(historical_claims),
                "established": sorted(established, key=preference_order),
                "review": sorted(review, key=preference_order),
                "cold_start": sorted(cold_start, key=preference_order),
                "historical_claims": sorted(
                    historical_claims, key=preference_order
                ),
                # Compatibility aliases now use the honest established meaning.
                "count": len(prefs),
                "learned": sorted(established, key=preference_order),
                "count_semantics": (
                    "tracked rows are not learned; established requires >=10 "
                    "independent evidence items and confidence >=0.8"
                ),
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
    """Get claims Lumen was told in Q&A (extracted from answers, not verified)."""
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
                    "references": i.references,
                    "conviction_score": round(i.conviction_score(), 3),
                    "rederived_from": len(i.derived_from),
                    "contradicted_by": len(i.contradicted_by),
                    "age": _age_str(i.timestamp),
                    "timestamp": i.timestamp,
                    "historical_as_of": datetime.fromtimestamp(
                        i.timestamp, timezone.utc
                    ).isoformat(),
                    "record_role": "historical_claim",
                    "current_state_authority": "none",
                }
                for i in insights
            ],
            "epistemic_note": (
                "These are timestamped Q&A claims, not current telemetry or "
                "sensor-observed preferences. Re-derivation affects claim "
                "ranking only and has no direct behavioral effect."
            ),
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
        from ..accessors import _get_growth, _get_store

        result = {"query": text, "type": query_type}

        # Always get relevant Q&A insights (keyword match)
        relevant = get_relevant_insights(text, limit=limit)
        result["qa_insights"] = [
            {
                "text": i.text,
                "category": i.category,
                "source_question": (
                    i.source_question[:60] + "..."
                    if len(i.source_question) > 60
                    else i.source_question
                ),
                "historical_as_of": datetime.fromtimestamp(
                    i.timestamp, timezone.utc
                ).isoformat(),
                "record_role": "historical_claim",
                "current_state_authority": "none",
            }
            for i in relevant
        ]

        # Add self-knowledge when cognitive/insights
        if query_type in ("cognitive", "insights", "self"):
            try:
                store = _get_store()
                if store:
                    # Reuse the evidence-reconciled view so the semantic query
                    # pipe cannot resurrect stale reflection wording or an old
                    # persisted causal snapshot that `/self-knowledge` already
                    # superseded with live broker state.
                    knowledge_content = await handle_get_self_knowledge({
                        "limit": limit,
                    })
                    knowledge_payload = json.loads(knowledge_content[0].text)
                    result["self_knowledge"] = knowledge_payload.get("summary")
                    result["reflection_insights"] = knowledge_payload.get(
                        "insights", []
                    )
                    result["self_knowledge_epistemic_note"] = (
                        "Q&A-derived reflection rows are historical claims, "
                        "not current-state evidence."
                    )
            except Exception:
                result["self_knowledge"] = None
                result["reflection_insights"] = []

        # Add growth summary when type is growth
        if query_type == "growth":
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
