import mujoco
import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS
from sumo.tasks.spot.spot_jug_kick import JUG_REST_HEIGHT, RADIUS_MIN, SpotJugKick

# Quaternion (w, x, y, z) laying the jug on its side (90 deg about world y).
LYING_QUAT = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])


@pytest.fixture(scope="module")
def task():
    return SpotJugKick()


def _upright_state(task, robot_xy=(-5.0, -5.0), jug_xy=(0.0, 0.0)):
    """Deterministic qpos: robot standing at robot_xy, jug upright at jug_xy."""
    np.random.seed(0)
    qpos = np.array(task.reset_pose)
    qpos[task.body_pose_start : task.body_pose_start + 2] = robot_xy
    o = task.object_pose_start
    qpos[o : o + 7] = [*jug_xy, JUG_REST_HEIGHT, 1, 0, 0, 0]
    return qpos


def test_registered():
    assert "spot_jug_kick" in get_registered_tasks()


def test_action_space_is_base_velocity(task):
    """Deployment contract: locomotion-only, same action space as spot_navigate."""
    assert task.nu == 3
    assert not task.use_arm


def test_model_layout(task):
    assert task.model.joint("jug_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert len(task.reset_pose) == task.model.nq
    assert task.perceived_object_joints == ("jug_joint",)


def test_object_z_axis_sensor_wiring(task):
    """The sensor slice used by the reward must be the jug's world-frame z-axis."""
    data = mujoco.MjData(task.model)
    data.qpos[:] = _upright_state(task)
    mujoco.mj_forward(task.model, data)
    s = task.object_z_axis_start
    np.testing.assert_allclose(data.sensordata[s : s + 3], [0, 0, 1], atol=1e-9)

    data.qpos[task.object_pose_start + 3 : task.object_pose_start + 7] = LYING_QUAT
    mujoco.mj_forward(task.model, data)
    assert abs(data.sensordata[s + 2]) < 1e-9  # lying: z-axis is horizontal


def test_reward_prefers_tipped_jug(task):
    """With everything else equal, a rollout that tips the jug scores higher."""
    batch, horizon = 2, 5
    nq, nv = task.model.nq, task.model.nv
    states = np.zeros((batch, horizon, nq + nv))
    states[..., task.body_pose_start + 2] = 0.52  # standing torso height
    sensors = np.zeros((batch, horizon, task.model.nsensordata))
    sensors[0, :, task.object_z_axis_start + 2] = 1.0  # stays upright
    sensors[1, :, task.object_z_axis_start + 2] = 0.0  # knocked flat
    controls = np.zeros((batch, horizon, task.nu))

    reward = task.reward(states, sensors, controls)
    assert reward.shape == (batch,)
    assert reward[1] > reward[0]
    assert reward[1] - reward[0] == pytest.approx(task.config.w_tip)


def test_tip_reward_saturates_at_horizontal(task):
    """Inverted must NOT out-score horizontal: the tip term caps once the jug is flat."""
    batch, horizon = 3, 4
    nq, nv = task.model.nq, task.model.nv
    states = np.zeros((batch, horizon, nq + nv))
    states[..., task.body_pose_start + 2] = 0.52  # standing
    sensors = np.zeros((batch, horizon, task.model.nsensordata))
    z = task.object_z_axis_start + 2
    sensors[0, :, z] = 1.0  # upright  -> tip 0
    sensors[1, :, z] = 0.0  # horizontal -> tip w_tip
    sensors[2, :, z] = -1.0  # inverted -> capped at w_tip (NOT 2*w_tip)
    controls = np.zeros((batch, horizon, task.nu))

    reward = task.reward(states, sensors, controls)
    assert reward[1] > reward[0]
    assert reward[2] == pytest.approx(reward[1])  # inverted scores no more than flat


def test_reset_spawns_jug_at_goal(task):
    """The jug spawns at goal_pos xy so the navigate term always points at it."""
    goal_xy = np.asarray(task.config.goal_pos)[:2]
    for _ in range(5):
        qpos = np.array(task.reset_pose)
        jug_xy = qpos[task.object_pose_start : task.object_pose_start + 2]
        assert np.linalg.norm(jug_xy - goal_xy) < 0.5  # within the reset jitter
        robot_xy = qpos[task.body_pose_start : task.body_pose_start + 2]
        # Robot starts a walk away from the jug (annulus), so it must locomote in.
        assert np.linalg.norm(robot_xy - jug_xy) >= RADIUS_MIN - 0.5


def test_reward_shape(task):
    batch, horizon = 4, 6
    rng = np.random.default_rng(0)
    states = rng.normal(size=(batch, horizon, task.model.nq + task.model.nv))
    sensors = rng.normal(size=(batch, horizon, task.model.nsensordata))
    controls = rng.normal(size=(batch, horizon, task.nu))
    assert task.reward(states, sensors, controls).shape == (batch,)


def test_success_requires_tip_and_standing(task):
    data = mujoco.MjData(task.model)
    data.qpos[:] = _upright_state(task)
    mujoco.mj_forward(task.model, data)
    assert not task.success(task.model, data)  # upright jug: no success
    assert not task.failure(task.model, data)

    o = task.object_pose_start
    data.qpos[o + 3 : o + 7] = LYING_QUAT
    assert task.success(task.model, data)  # tipped + standing

    data.qpos[task.body_pose_start + 2] = 0.2  # robot fell while tipping
    assert not task.success(task.model, data)
    assert task.failure(task.model, data)


def _settle(task, data, seconds):
    """Step physics while position-servoing the robot to its standing pose.

    Without ctrl the leg servos drive to 0 and the (far-away) robot collapses,
    which would spuriously fail success() assertions on the settled state.
    """
    data.ctrl[:] = [*LEGS_STANDING_POS, *task.reset_arm_pos]
    steps = int(round(seconds / task.model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(task.model, data)


def test_physics_upright_jug_is_stable(task):
    """An untouched jug must not fall over on its own (collision/inertia sanity)."""
    data = mujoco.MjData(task.model)
    data.qpos[:] = _upright_state(task)  # robot 7 m away, cannot interfere
    _settle(task, data, 1.0)
    assert task._jug_upright_cos(data) > 0.95


def test_physics_shove_moves_or_tips_jug(task):
    """A kick-magnitude shove must visibly displace or topple the jug."""
    data = mujoco.MjData(task.model)
    data.qpos[:] = _upright_state(task)
    # get_joint_velocity_start_index addresses the concatenated [qpos | qvel]
    # rollout state; subtract nq for the index into data.qvel.
    v = task.get_joint_velocity_start_index("jug_joint") - task.model.nq
    data.qvel[v : v + 3] = [3.0, 0.0, 0.0]
    _settle(task, data, 1.5)
    displaced = np.linalg.norm(data.qpos[task.object_pose_start : task.object_pose_start + 2]) > 0.3
    tipped = task._jug_upright_cos(data) < task.config.tip_success_cos
    assert displaced or tipped


def test_physics_lying_jug_stays_down(task):
    """A jug laid on its side settles there; success holds on the settled state."""
    data = mujoco.MjData(task.model)
    data.qpos[:] = _upright_state(task)
    o = task.object_pose_start
    data.qpos[o + 2] = 0.145  # roughly the lying rest height (jug radius)
    data.qpos[o + 3 : o + 7] = LYING_QUAT
    _settle(task, data, 1.0)
    assert task._jug_upright_cos(data) < 0.3
    assert task.success(task.model, data)
