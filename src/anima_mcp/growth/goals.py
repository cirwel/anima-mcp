"""
Growth System - Goal formation and tracking mixin.

Handles forming goals, updating progress, checking goal completion,
and suggesting new goals.
"""

import sys
import re
import json
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

from .models import Goal, GoalStatus


class GoalsMixin:
    """Mixin for goal formation and tracking."""

    def form_goal(self, description: str, motivation: str,
                  target_days: Optional[int] = None) -> Goal:
        """Form a new personal goal."""
        import uuid
        conn = self._connect()
        now = datetime.now()

        goal_id = str(uuid.uuid4())[:8]
        target_date = now + timedelta(days=target_days) if target_days else None

        goal = Goal(
            goal_id=goal_id,
            description=description,
            motivation=motivation,
            status=GoalStatus.ACTIVE,
            created_at=now,
            target_date=target_date,
            progress=0.0,
            milestones=[],
            last_worked_on=None,
        )
        self._goals[goal_id] = goal

        conn.execute("""
            INSERT INTO goals (goal_id, description, motivation, status, created_at, target_date, progress, milestones)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (goal.goal_id, goal.description, goal.motivation, goal.status.value,
              goal.created_at.isoformat(),
              goal.target_date.isoformat() if goal.target_date else None,
              goal.progress, json.dumps(goal.milestones)))
        conn.commit()

        print(f"[Growth] New goal: {description}", file=sys.stderr, flush=True)
        return goal

    def update_goal_progress(self, goal_id: str, progress: float,
                             milestone: Optional[str] = None) -> Optional[str]:
        """Update progress on a goal. Returns celebration message if achieved."""
        if goal_id not in self._goals:
            return None

        conn = self._connect()
        goal = self._goals[goal_id]
        goal.progress = min(1.0, progress)
        goal.last_worked_on = datetime.now()

        if milestone:
            goal.milestones.append(milestone)

        message = None
        if goal.progress >= 1.0:
            goal.status = GoalStatus.ACHIEVED
            message = f"I did it! {goal.description}"
            self._record_memory(
                f"Achieved goal: {goal.description}",
                emotional_impact=0.8,
                category="milestone"
            )

        conn.execute("""
            UPDATE goals SET progress = ?, milestones = ?, last_worked_on = ?, status = ?
            WHERE goal_id = ?
        """, (goal.progress, json.dumps(goal.milestones),
              goal.last_worked_on.isoformat(), goal.status.value, goal_id))
        conn.commit()

        return message

    def check_goal_progress(self, anima_state: Dict[str, float],
                            self_model=None) -> Optional[str]:
        """Periodically check progress on active goals. Returns message if achieved."""
        now = datetime.now()
        messages = []

        for goal in list(self._goals.values()):
            if goal.status != GoalStatus.ACTIVE:
                continue

            # Auto-abandon stale goals past target date.
            # Either (a) progress never got off the ground, or (b) progress
            # has stalled — nobody's touched the goal in 14+ days. The stalled
            # path prevents goals from sitting at mid-range progress forever
            # and blocking suggest_goal's Max 2 active-goals slot.
            if goal.target_date and now > goal.target_date:
                stalled = (
                    goal.last_worked_on is None
                    or (now - goal.last_worked_on).days >= 14
                )
                if goal.progress < 0.1 or stalled:
                    goal.status = GoalStatus.ABANDONED
                    conn = self._connect()
                    conn.execute("UPDATE goals SET status = ? WHERE goal_id = ?",
                                 (goal.status.value, goal.goal_id))
                    conn.commit()
                    reason = "no-progress" if goal.progress < 0.1 else "stalled"
                    print(f"[Growth] Abandoned {reason} goal: {goal.description}",
                          file=sys.stderr, flush=True)
                    continue

            msg = None

            # Drawing count goals
            if "drawings" in goal.description.lower():
                match = re.search(r'complete (\d+) drawings', goal.description)
                if match:
                    target = int(match.group(1))
                    progress = min(1.0, self._drawings_observed / target)
                    msg = self.update_goal_progress(goal.goal_id, progress)
                    if msg:
                        messages.append(msg)

            # Curiosity/question goals — resolved if question was answered
            elif goal.description.startswith("find an answer to:"):
                question = goal.description.replace("find an answer to: ", "")
                if question not in self._curiosities:
                    msg = self.update_goal_progress(
                        goal.goal_id, 1.0, milestone="question answered")
                    if msg:
                        messages.append(msg)

            # Understanding goals — preference confidence increased further
            elif "understand why" in goal.description.lower():
                for pref in self._preferences.values():
                    if pref.description.lower() in goal.description.lower():
                        if pref.confidence > 0.9 and pref.observation_count > 100:
                            msg = self.update_goal_progress(
                                goal.goal_id, 1.0,
                                milestone=f"observed {pref.observation_count} times")
                            if msg:
                                messages.append(msg)
                        break

            # Belief-testing goals — track evidence accumulation + decisive completion
            elif "test whether" in goal.description.lower() and self_model:
                for bid, belief in self_model.beliefs.items():
                    if belief.description.lower() in goal.description.lower():
                        if belief.confidence > 0.7 or belief.confidence < 0.2:
                            # Decisive — complete!
                            msg = self.update_goal_progress(
                                goal.goal_id, 1.0,
                                milestone=f"belief is now {belief.get_belief_strength()}")
                        else:
                            # Show intermediate progress based on evidence + confidence movement
                            total_evidence = belief.supporting_count + belief.contradicting_count
                            # Evidence component: each observation counts, saturates around 20
                            evidence_progress = min(0.4, total_evidence / 50.0)
                            # Confidence component: how far from 0.5 toward either threshold
                            confidence_movement = abs(belief.confidence - 0.5) / 0.2
                            confidence_progress = min(0.55, confidence_movement * 0.55)
                            progress = min(0.95, evidence_progress + confidence_progress)
                            if progress > goal.progress:
                                msg = self.update_goal_progress(goal.goal_id, progress)
                        if msg:
                            messages.append(msg)
                        break

        return messages[0] if messages else None

    def suggest_goal(self, anima_state: Dict[str, float],
                      self_model=None) -> Optional[Goal]:
        """Suggest a goal grounded in actual experience data."""
        # Don't suggest if already have enough active goals
        active_count = sum(1 for g in self._goals.values() if g.status == GoalStatus.ACTIVE)
        if active_count >= 2:
            return None

        wellness = sum(anima_state.values()) / len(anima_state) if anima_state else 0.5
        # Each suggestion carries its own target_days — drawing milestones need
        # weeks to complete honestly; a fixed 7-day target sets them up to
        # expire stale. The historical Feb-2026 "complete 500 drawings in 7
        # days" was symptomatic.
        suggestions: List[Tuple[str, str, int]] = []

        # 1. Preference-driven: strong preferences create curiosity about why
        for pref in self._preferences.values():
            if pref.confidence > 0.7 and pref.value > 0.5 and pref.observation_count > 50:
                suggestions.append((
                    f"understand why {pref.description.lower()}",
                    f"i've noticed this {pref.observation_count} times",
                    14,
                ))
                break  # Only one preference goal

        # 2. Curiosity-driven: recurring unanswered questions
        if self._curiosities:
            q = random.choice(self._curiosities)
            suggestions.append((
                f"find an answer to: {q}",
                "this has been on my mind",
                21,
            ))

        # 3. Drawing milestones based on actual count. Target_days scales with
        # the gap between observed and milestone so Lumen's natural ~15/day
        # pace can actually reach the target — minimum 7 days, cap 60.
        if self._drawings_observed > 0:
            milestones = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
            for m in milestones:
                if self._drawings_observed < m:
                    gap = m - self._drawings_observed
                    # Assume ~10 drawings/day as a conservative pace
                    days = max(7, min(60, gap // 10))
                    suggestions.append((
                        f"complete {m} drawings",
                        f"i've done {self._drawings_observed} so far",
                        days,
                    ))
                    break

        # 4. Belief-testing: uncertain beliefs worth investigating
        if self_model:
            for bid, belief in self_model.beliefs.items():
                total = belief.supporting_count + belief.contradicting_count
                if 0.3 < belief.confidence < 0.6 and total >= 3:
                    suggestions.append((
                        f"test whether {belief.description.lower()}",
                        f"i'm only {belief.get_belief_strength()} about this",
                        14,
                    ))
                    break

        # 5. Wellness-driven
        if wellness < 0.4:
            suggestions.append(("find what makes me feel stable",
                                "i want to understand myself better",
                                14))
        elif wellness > 0.8 and anima_state.get("clarity", 0.5) > 0.8:
            suggestions.append(("explore a new question while my mind is clear",
                                "my clarity is high and i feel curious",
                                7))

        if not suggestions:
            return None

        desc, motivation, target_days = random.choice(suggestions)

        # Dedup against ALL goals including achieved/abandoned (prevents re-creating)
        conn = self._connect()
        existing = conn.execute(
            "SELECT 1 FROM goals WHERE LOWER(description) = LOWER(?)", (desc,)
        ).fetchone()
        if existing:
            return None

        return self.form_goal(desc, motivation, target_days=target_days)
