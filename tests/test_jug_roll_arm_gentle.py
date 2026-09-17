"""The frozen gentle arm-roll deployment profile matches its authoritative offline run.

The fixture is the `initial_config` / `planning_settings` of
out/jug_roll_horizon_tradeoff_20260917/n32_h1p5/front/seed0.json (handoff §3). If the
profile drifts from it, the deployed task is no longer the one the 10/12 result was
measured on -- that is what this file is for, not a behaviour test.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from judo.config import _OVERRIDE_REGISTRY
from judo.controller import ControllerConfig
from judo.optimizers.cem import CrossEntropyMethodConfig
from judo.tasks import get_registered_tasks

import sumo.tasks  # noqa: F401  (registers the spot tasks)
from sumo.controller.optimizer_overrides import set_default_spot_optimizer_overrides
from sumo.controller.overrides import set_default_spot_overrides
from sumo.tasks.spot.spot_jug_manipulation import (
    SpotJugRoll,
    SpotJugRollArmGentle,
    SpotJugRollArmGentleCoarse,
    SpotJugRollArmGentleCoarseConfig,
    SpotJugRollArmGentleConfig,
    SpotJugRollArmGentleDry,
    SpotJugRollArmGentleDryConfig,
    SpotJugRollConfig,
)

FIXTURE = json.loads((Path(__file__).parent / "data" / "jug_roll_arm_gentle_profile.json").read_text())
# Offline scenario coordinates; on the robot A/B come from perception and the operator.
RUNTIME_GOAL_FIELDS = {"start_pos", "goal_pos"}


def _plain(value):
    return np.asarray(value).tolist() if isinstance(value, (np.ndarray, list, tuple)) else value


def test_profile_config_equals_authoritative_run():
    got = {k: _plain(v) for k, v in dataclasses.asdict(SpotJugRollArmGentleConfig()).items()}
    want = FIXTURE["initial_config"]
    assert set(got) == set(want), sorted(set(got) ^ set(want))
    diff = {k: (got[k], want[k]) for k in want if k not in RUNTIME_GOAL_FIELDS and got[k] != want[k]}
    assert diff == {}, diff
    # and the profile really is an opt-in on top of the deployed roll defaults
    base = {k: _plain(v) for k, v in dataclasses.asdict(SpotJugRollConfig()).items()}
    assert base["roll_use_arm"] is False and base["water_fill_ratio"] == 0.0
    assert got["roll_use_arm"] is True and got["water_fill_ratio"] == 0.1


@pytest.fixture(scope="module")
def task():
    return SpotJugRollArmGentle()


def test_profile_builds_the_arm_model(task):
    assert task.mode == "roll" and task.name == "spot_jug_roll_arm_gentle"
    assert task.use_arm and task.use_gripper and task.nu == 11
    assert task.roll_hand_sensors
    names = [task.model.sensor(i).name for i in range(task.model.nsensor)]
    assert "jug_roll_hand_position" in names and "jug_roll_hand_velocity" in names
    assert task.water_count == 19 and len(task.synthesized_joints) == 19
    assert task.water_mass == pytest.approx(1.903, abs=1e-3)
    # arm_rest pulls towards the EXTENDED neutral posture, not the stowed one
    assert np.allclose(np.asarray(task.reset_arm_pos)[:6], [0.0, -0.9, 1.8, 0.0, -0.9, 0.0])
    plain = SpotJugRoll()
    assert plain.nu == 3 and not plain.roll_hand_sensors


def test_profile_reward_has_the_arm_terms(task):
    model = task.model
    n = model.nq + model.nv
    states = np.tile(np.concatenate([task.reset_pose, np.zeros(model.nv)]), (2, 3, 1)).astype(float)
    sensors = np.zeros((2, 3, model.nsensordata))
    controls = np.zeros((2, 3, task.nu))
    terms = task.reward_terms(states, sensors, controls, task.config)
    for name in ("arm_speed", "arm_peak_speed", "arm_body_clearance", "hand_speed",
                 "hand_peak_speed", "hand_reach", "arm_rest", "standoff", "approach", "position"):
        assert name in terms, name
        assert np.all(np.asarray(terms[name]) <= 1e-9), (name, terms[name])
    assert states.shape[-1] == n


def test_registered_with_the_authoritative_mpc_shape():
    assert "spot_jug_roll_arm_gentle" in get_registered_tasks()
    set_default_spot_overrides()
    set_default_spot_optimizer_overrides()
    want = dict(FIXTURE["planning_settings"])
    assert want.pop("optimizer") == "cem"
    # The RESOLVED configs (defaults + this task's overrides), every fixture field covered.
    ctrl, cem = ControllerConfig(), CrossEntropyMethodConfig()
    ctrl.set_override("spot_jug_roll_arm_gentle")
    cem.set_override("spot_jug_roll_arm_gentle")
    resolved = {**{f: getattr(cem, f) for f in vars(cem)}, **{f: getattr(ctrl, f) for f in vars(ctrl)}}
    missing = [k for k in want if k not in resolved]
    assert not missing, missing
    diff = {k: (resolved[k], want[k]) for k in want if resolved[k] != want[k]}
    assert diff == {}, diff
    assert ctrl.horizon == 1.5 and cem.num_rollouts == 32
    # the plain roll task keeps the deployed nu=3 family shape
    plain = _OVERRIDE_REGISTRY[CrossEntropyMethodConfig]["spot_jug_roll"]
    assert plain["num_rollouts"] == 24 and _OVERRIDE_REGISTRY[ControllerConfig]["spot_jug_roll"]["horizon"] == 2.0


def test_dry_variant_differs_only_in_water_and_mass():
    wet = {k: _plain(v) for k, v in dataclasses.asdict(SpotJugRollArmGentleConfig()).items()}
    dry = {k: _plain(v) for k, v in dataclasses.asdict(SpotJugRollArmGentleDryConfig()).items()}
    assert {k for k in wet if wet[k] != dry[k]} == {"water_fill_ratio", "jug_mass"}
    assert dry["water_fill_ratio"] == 0.0 and dry["jug_mass"] == 3.4
    task = SpotJugRollArmGentleDry()
    assert task.nu == 11 and task.roll_hand_sensors and task.water_count == 0
    assert not getattr(task, "synthesized_joints", ())
    assert task.model.body("jug").mass[0] == pytest.approx(3.4, abs=1e-6)
    assert "spot_jug_roll_arm_gentle_dry" in get_registered_tasks()
    set_default_spot_overrides()
    set_default_spot_optimizer_overrides()
    ctrl, cem = ControllerConfig(), CrossEntropyMethodConfig()
    ctrl.set_override("spot_jug_roll_arm_gentle_dry")
    cem.set_override("spot_jug_roll_arm_gentle_dry")
    assert ctrl.horizon == 1.5 and cem.num_rollouts == 32 and cem.num_elites == 3


def test_coarse_variant_keeps_newton_with_five_balls():
    import mujoco

    from sumo.tasks.spot import jug_water

    wet = {k: _plain(v) for k, v in dataclasses.asdict(SpotJugRollArmGentleConfig()).items()}
    coarse = {k: _plain(v) for k, v in dataclasses.asdict(SpotJugRollArmGentleCoarseConfig()).items()}
    assert {k for k in wet if wet[k] != coarse[k]} == {"water_ball_radius"}
    task = SpotJugRollArmGentleCoarse()
    assert task.nu == 11 and task.roll_hand_sensors
    assert task.water_count == 5 and task.water_mass == pytest.approx(1.903, abs=1e-3)
    assert len(task.synthesized_joints) == 5
    assert task.model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert SpotJugRollArmGentle().model.opt.solver == mujoco.mjtSolver.mjSOL_CG
    assert jug_water.water_parameters(0.1, 0.06)[0] == 1
    with pytest.raises(ValueError):
        jug_water.water_parameters(0.1, 0.07)
    # the settled grid must have a slot for a ball this big (the axis)
    for radius in (0.045, 0.06):
        n, m = jug_water.water_parameters(0.1, radius)
        pos = jug_water.settled_positions(n, radius, m, (0.7071, 0.7071, 0.0, 0.0))
        assert pos.shape == (n, 3) and np.isfinite(pos).all()
    assert "spot_jug_roll_arm_gentle_coarse" in get_registered_tasks()
    set_default_spot_overrides()
    set_default_spot_optimizer_overrides()
    ctrl, cem = ControllerConfig(), CrossEntropyMethodConfig()
    ctrl.set_override("spot_jug_roll_arm_gentle_coarse")
    cem.set_override("spot_jug_roll_arm_gentle_coarse")
    assert ctrl.horizon == 1.5 and cem.num_rollouts == 32


def test_coarse_balls_never_overlap_when_pooled_per_tick():
    """Pooled placement keeps few big balls apart and inside at every tilt.

    The planner re-synthesizes the balls from the observed jug pose every tick; an
    overlap is a contact transient in every rollout (review 2026-09-17: 0.93 m/s
    within 50 ms at 30 deg).
    """
    from sumo.tasks.spot import jug_water

    (z_bot, wall), (z_top, _), *_ = jug_water.PROFILE
    for radius, counts in ((0.045, (1, 2, 3)), (0.04, (1, 3, 5)), (0.035, (5, 7))):
      for count in counts:
        for tilt_deg in (0, 10, 30, 45, 60, 90, 100, 120, 150, 180):
            a = np.radians(tilt_deg) / 2
            quat = (np.cos(a), np.sin(a), 0.0, 0.0)            # tilt about x
            pts = jug_water.pooled_local_positions(count, radius, quat)
            assert pts.shape == (count, 3), (radius, count, tilt_deg)
            assert np.all(np.hypot(pts[:, 0], pts[:, 1]) <= wall - radius + 1e-9), (radius, count, tilt_deg)
            assert np.all(pts[:, 2] >= z_bot + radius - 1e-9) and np.all(pts[:, 2] <= z_top - radius + 1e-9)
            for i in range(count):
                for j in range(i + 1, count):
                    assert np.linalg.norm(pts[i] - pts[j]) >= 2 * radius - 1e-9, (radius, count, tilt_deg, i, j)
    # fine balls keep the historical row placement (the frozen profile's behaviour)
    fine = jug_water.pooled_local_positions(19, 0.025, (0.7071, 0.7071, 0.0, 0.0))
    assert fine.shape == (19, 3)
