import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.spot.spot_barrel_look_at import YAW_RATE_INDEX, SpotBarrelLookAt
from sumo.tasks.spot.spot_barrel_perceive import SpotBarrelPerceive


@pytest.fixture(scope="module")
def task():
    return SpotBarrelLookAt()


def _state(task, yaw: float, barrel_xy=(2.0, 0.0)):
    """One (1, 1, nq+nv) state: robot at the origin with the given yaw, barrel at barrel_xy."""
    q = np.array(task.reset_pose, dtype=float)
    b = task.body_pose_idx
    q[b : b + 3] = [0.0, 0.0, 0.52]
    q[b + 3 : b + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # wxyz, yaw about z
    o = task.barrel_pose_idx
    q[o : o + 2] = barrel_xy
    return np.concatenate([q, np.zeros(task.model.nv)])[None, None, :]


def test_registered():
    assert "spot_barrel_look_at" in get_registered_tasks()


def test_is_a_perceive_task_with_the_deployed_morphology(task):
    assert isinstance(task, SpotBarrelPerceive)
    assert task.nu == 3
    assert task.perceived_object_joints == ("barrel_joint",)


def test_yaw_rate_is_hard_capped(task):
    base = SpotBarrelPerceive().actuator_ctrlrange
    capped = task.actuator_ctrlrange
    assert base[YAW_RATE_INDEX, 1] > task.config.max_yaw_rate  # the cap actually narrows
    np.testing.assert_allclose(capped[YAW_RATE_INDEX], [-task.config.max_yaw_rate, task.config.max_yaw_rate])


def test_lock_xy_default_turns_in_place_only(task):
    assert task.config.lock_xy
    capped = task.actuator_ctrlrange
    np.testing.assert_allclose(capped[:YAW_RATE_INDEX], 0.0)          # vx, vy cannot be commanded
    assert capped[YAW_RATE_INDEX, 1] > 0.0                             # yaw still can


def test_lock_xy_off_keeps_the_base_xy_bounds():
    from sumo.tasks.spot.spot_barrel_look_at import SpotBarrelLookAtConfig
    free = SpotBarrelLookAt(SpotBarrelLookAtConfig(lock_xy=False))
    base = SpotBarrelPerceive().actuator_ctrlrange
    np.testing.assert_allclose(free.actuator_ctrlrange[:YAW_RATE_INDEX], base[:YAW_RATE_INDEX])


def test_yaw_rate_index_is_the_torso_yaw_command(task):
    """controls[2] must land in the policy command's torso yaw-rate slot, or the cap is on the wrong axis."""
    u = np.zeros((1, 3))
    u[0, YAW_RATE_INDEX] = 0.3
    cmd = np.asarray(task.task_to_sim_ctrl(u)).reshape(-1)
    assert cmd[2] == pytest.approx(0.3)
    assert cmd[0] == pytest.approx(0.0) and cmd[1] == pytest.approx(0.0)


def test_heading_cos_geometry(task):
    q = _state(task, yaw=0.0)[..., : task.model.nq]
    assert task.heading_cos(q)[0, 0] == pytest.approx(1.0)            # facing +x, barrel at +x
    q = _state(task, yaw=np.pi)[..., : task.model.nq]
    assert task.heading_cos(q)[0, 0] == pytest.approx(-1.0)           # facing away
    q = _state(task, yaw=np.pi / 2)[..., : task.model.nq]
    assert task.heading_cos(q)[0, 0] == pytest.approx(0.0, abs=1e-9)  # side-on
    q = _state(task, yaw=0.0, barrel_xy=(0.0, 0.0))[..., : task.model.nq]
    assert task.heading_cos(q)[0, 0] == pytest.approx(1.0)            # degenerate: on top of it


def test_facing_the_barrel_scores_higher(task):
    sensors = np.zeros((1, 1, task.model.nsensordata))
    u = np.zeros((1, 1, task.nu))
    r_face = task.reward(_state(task, 0.0), sensors, u)[0]
    r_side = task.reward(_state(task, np.pi / 2), sensors, u)[0]
    r_away = task.reward(_state(task, np.pi), sensors, u)[0]
    assert r_face > r_side > r_away
    # 90 deg costs w_look, faded by the distance ramp at the barrel's (first-step) distance
    from sumo.tasks.spot.look_at import ramp
    q = _state(task, 0.0)[0, 0, : task.model.nq]
    b, o = task.body_pose_idx, task.barrel_pose_idx
    fade = ramp(np.hypot(q[o] - q[b], q[o + 1] - q[b + 1]), task.config.look_ramp_dist)
    assert r_face - r_side == pytest.approx(task.config.w_look * fade)
    assert r_face - r_away == pytest.approx(2 * task.config.w_look * fade)


def test_turning_fast_costs(task):
    sensors = np.zeros((1, 1, task.model.nsensordata))
    s = _state(task, 0.0)
    slow = np.zeros((1, 1, task.nu))
    slow[..., YAW_RATE_INDEX] = 0.1
    fast = np.zeros((1, 1, task.nu))
    fast[..., YAW_RATE_INDEX] = 0.4
    r_slow, r_fast = task.reward(s, sensors, slow)[0], task.reward(s, sensors, fast)[0]
    assert r_slow > r_fast
    assert r_slow - r_fast == pytest.approx(task.config.w_yaw_rate * 0.3)


def test_reward_shape(task):
    batch, horizon = 4, 6
    rng = np.random.default_rng(0)
    states = rng.normal(size=(batch, horizon, task.model.nq + task.model.nv))
    sensors = rng.normal(size=(batch, horizon, task.model.nsensordata))
    controls = rng.normal(size=(batch, horizon, task.nu))
    assert task.reward(states, sensors, controls).shape == (batch,)
