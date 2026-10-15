"""Earned mark effects reach the self-model (the bonus was always 0 before)."""

import inspect
from types import SimpleNamespace

from anima_mcp import stable_creature
from anima_mcp.experiential_marks import ExperientialMarks
from anima_mcp.stable_creature import _apply_belief_update_bonus


def test_bonus_follows_the_earned_marks(tmp_path):
    marks = ExperientialMarks(db_path=str(tmp_path / "marks.db"))
    model = SimpleNamespace(belief_update_bonus=0.0)

    _apply_belief_update_bonus(model, marks)
    assert model.belief_update_bonus == 0.0

    marks.earn_mark("maturity_adolescent")
    _apply_belief_update_bonus(model, marks)
    assert model.belief_update_bonus == 0.15


def test_missing_parts_are_a_no_op():
    model = SimpleNamespace(belief_update_bonus=0.0)
    _apply_belief_update_bonus(model, None)
    _apply_belief_update_bonus(None, object())
    assert model.belief_update_bonus == 0.0


def test_run_creature_applies_it_after_marks_exist_and_after_earning():
    """The 2026-03 bug: the bonus was read before exp_marks was assigned."""
    source = inspect.getsource(stable_creature.run_creature)
    created = source.index("exp_marks = get_experiential_marks(")
    applied = [i for i in range(len(source))
               if source.startswith("_apply_belief_update_bonus(self_model, exp_marks)", i)]
    assert len(applied) == 2
    assert all(i > created for i in applied)
    assert applied[1] > source.index("exp_marks.check_and_earn(")
