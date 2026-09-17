import json

import numpy as np
import pytest

from sumo.tasks.spot.spot_jug_manipulation import SpotJugRoll, SpotJugRollConfig
from tools.jug_roll_targets import REFERENCE, SCENARIOS, gates, intervals, set_target
from tools.jug_roll_targets_report import all_positions_reached, validate


def task():
    return SpotJugRoll(SpotJugRollConfig(**json.loads((REFERENCE / "gentle_seed0.json").read_text())["config"]))


@pytest.mark.parametrize("scenario", ["left", "right", "behind"])
def test_target_change_preserves_fixed_robot_jug_water_state(scenario):
    t = task()
    original = np.r_[t.data.qpos.copy(), t.data.qvel.copy(), t.data.time]
    set_target(t, SCENARIOS[scenario][0])
    np.testing.assert_array_equal(original, np.r_[t.data.qpos, t.data.qvel, t.data.time])
    np.testing.assert_array_equal(t.config.goal_pos, SCENARIOS[scenario][0])
    assert np.linalg.norm(t.config.goal_pos[:2] - t.config.start_pos[:2]) == pytest.approx(1.2)


def test_goal_switch_resets_only_progress_metadata_not_physics():
    t = task()
    t.data.time = 4.0
    t.rolling_distance = 1.0
    t.axial_rotation = 10.0
    original = t.data.qpos.copy()
    water = t.data.qpos[t.get_joint_position_start_index("jug_water_0_joint") :].copy()
    set_target(t, [2.15, 1.2, 0.14], t.data.qpos[t.object_pose_start : t.object_pose_start + 3])
    np.testing.assert_array_equal(t.data.qpos, original)
    np.testing.assert_array_equal(t.data.qpos[t.get_joint_position_start_index("jug_water_0_joint") :], water)
    assert t.data.time == 4 and t._last_metric_time == 4
    assert t.rolling_distance == t.axial_rotation == 0


def test_arrival_gates_are_active_before_success_distance():
    t = task()
    g = gates(t, 0.356)
    assert g["push_gate"] == pytest.approx(0.106 / 0.3)
    assert g["effective_rest_weight"] > 6
    assert g["effective_reach_weight"] < 6
    assert gates(t, 0.6)["effective_rest_weight"] == 0
    assert gates(t, 0.2)["push_gate"] == 0


def test_contact_interval_edges():
    r = [dict(time=0.02 * (i + 1), contact=bool(v)) for i, v in enumerate([0, 1, 1, 0, 1])]
    np.testing.assert_allclose(intervals(r, "contact"), [[0.02, 0.06], [0.08, 0.1]])


def test_positions_can_succeed_without_strict_roll_and_require_every_goal():
    r = dict(goals=[[0, 0, 0], [1, 0, 0]], stages=[dict(end_position_settled=True, status="success")])
    assert not all_positions_reached(r)
    r["stages"].append(dict(end_position_settled=True, status="timeout"))
    assert all_positions_reached(r)
    r["stages"][1]["end_position_settled"] = False
    assert not all_positions_reached(r)


def test_report_rejects_changed_reward_physics_and_extra_plans():
    from copy import deepcopy

    ref = json.loads((REFERENCE / "gentle_seed0.json").read_text())
    r = dict(
        physical_initial_hash=ref["physical_initial_hash"],
        planning_settings=ref["planning_settings"],
        controller=ref["controller"],
        optimizer=ref["optimizer"],
        steps=1500,
        duration=30,
        planning_updates=600,
        candidate_evaluations=14400,
        goals=SCENARIOS["left"],
        scenario="left",
        initial_config=deepcopy(ref["config"]),
        stages=[dict(index=0, goal=SCENARIOS["left"][0], start_time=0, end_time=30, elapsed=30, status="timeout")],
        completed_all=False,
    )
    r["initial_config"]["goal_pos"] = SCENARIOS["left"][0]
    validate(r, ref)
    for key, value in [("water_fill_ratio", 0), ("w_arm_rest", 0)]:
        altered = deepcopy(r)
        altered["initial_config"][key] = value
        with pytest.raises(AssertionError):
            validate(altered, ref)
    r["planning_updates"] = 601
    with pytest.raises(AssertionError):
        validate(r, ref)


def test_cli_forwards_ground_friction_and_reward_sets(monkeypatch, tmp_path):
    """main() must forward --ground-friction and --set to run().

    Regression (2026-09-18): the positional call silently dropped both options and a whole
    study ran the baseline nine times.
    """
    import sys

    from tools import jug_roll_targets as m

    captured = {}
    real_run = m.run

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(m, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "--scenario", "front", "--seed", "0", "--out", str(tmp_path),
                                      "--task", "spot_jug_roll_arm_gentle_coarse", "--ground-friction", "2.5",
                                      "--set", "w_torso_roll=200", "--set", "w_torso_pitch=50"])
    m.main()
    import inspect
    bound = inspect.signature(real_run).bind(*captured["args"], **captured["kwargs"])
    assert bound.arguments["ground_friction"] == 2.5
    assert bound.arguments["reward_sets"] == {"w_torso_roll": "200", "w_torso_pitch": "50"}
    assert bound.arguments["task_name"] == "spot_jug_roll_arm_gentle_coarse"


def test_study_options_require_a_registered_task_and_scalar_fields():
    import pytest

    from tools.jug_roll_targets import make_system

    with pytest.raises(ValueError, match="--task"):
        make_system(ground_friction=1.0)
    with pytest.raises(ValueError, match="--task"):
        make_system(reward_sets={"w_torso_roll": "1"})
    with pytest.raises(ValueError, match="int/float"):
        make_system(task_name="spot_jug_roll_arm_gentle_coarse", reward_sets={"goal_pos": "[0,0,0]"})
    with pytest.raises(ValueError, match="construction-time"):
        make_system(task_name="spot_jug_roll_arm_gentle_coarse", reward_sets={"jug_mass": "2"})
