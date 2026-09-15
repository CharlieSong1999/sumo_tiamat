import mujoco
import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.spot.spot_jug_manipulation import (
    SpotJugLayDown,
    SpotJugMove,
    SpotJugRoll,
    SpotJugUpright,
    disable_velocity_rewards,
)


@pytest.fixture(params=[SpotJugUpright, SpotJugLayDown, SpotJugRoll, SpotJugMove], scope="module")
def task(request):
    return request.param()


def test_registration_reset_and_action_space(task):
    task.reset()
    assert get_registered_tasks()[task.name].task_type is type(task)
    # Arm/gripper mode appends a gripper-selection channel to the ten commands.
    assert task.nu == (11 if task.mode == "upright" else 3)
    assert task.reset_pose.shape == (task.model.nq,)
    assert task.perceived_object_joints == ("jug_joint",)
    assert not task.success(task.model, task.data)
    assert not task.failure(task.model, task.data)
    assert not task.data.qvel.any()


def test_reward_shape_and_fall_penalty(task):
    task.reset()
    states = np.tile(np.r_[task.data.qpos, task.data.qvel], (3, 6, 1))
    sensors = np.tile(task.data.sensordata, (3, 6, 1))
    controls = np.zeros((3, 6, task.nu))
    healthy = task.reward(states, sensors, controls)
    assert healthy.shape == (3,)
    assert np.isfinite(healthy).all()
    states[1, 2, task.body_pose_start + 2] = 0.2
    terms = task.reward_terms(states, sensors, controls)
    assert terms["fall"][1] == -task.config.fall_penalty
    assert terms["fall"][0] == 0


def test_upright_orientation_and_airborne_rejection():
    task = SpotJugUpright()
    o = task.object_pose_start
    task.data.qpos[o : o + 7] = [*task.config.goal_pos[:2], 0.2415, 1, 0, 0, 0]
    mujoco.mj_forward(task.model, task.data)
    assert task.success(task.model, task.data)
    task.data.qpos[o + 2] = 0.6
    assert not task.success(task.model, task.data)
    task.data.qpos[o + 2] = 0.2415
    task.data.qpos[o + 3 : o + 7] = [0, 1, 0, 0]
    assert not task.success(task.model, task.data)


def test_lay_down_requires_horizontal_and_slow():
    task = SpotJugLayDown()
    o = task.object_pose_start
    v = task.object_vel_start - task.model.nq
    task.data.qpos[o : o + 7] = [*task.config.goal_pos[:2], 0.14, np.sqrt(0.5), np.sqrt(0.5), 0, 0]
    assert task.success(task.model, task.data)
    task.data.qvel[v] = 1
    assert not task.success(task.model, task.data)
    task.data.qvel[:] = 0
    task.data.qpos[o + 3 : o + 7] = [0, 1, 0, 0]
    assert not task.success(task.model, task.data)


def test_roll_rejects_sliding_and_spinning_in_place():
    task = SpotJugRoll()
    pos = np.tile([1.2, 0, 0.14], (4, 1, 1))
    axis = np.tile([0, -1, 0], (4, 1, 1))
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    vel[0, :, 0] = 0.5  # sliding only
    omega[1, :, 2] = -0.5 / task.config.radius  # spinning only
    vel[2:, :, 0] = 0.5
    omega[2:, :, 2] = -0.5 / task.config.radius
    pos[3, :, 2] = 0.5  # airborne translation and rotation
    coupled, slip = task._roll_speeds(pos, vel, omega, axis)
    np.testing.assert_allclose(coupled[:, 0], [0, 0, 0.5, 0])
    assert slip[2, 0] == pytest.approx(0)


def test_roll_success_requires_history_and_reset_clears_it():
    task = SpotJugRoll()
    o = task.object_pose_start
    task.data.qpos[o : o + 3] = [*task.config.goal_pos[:2], 0.14]
    assert not task.success(task.model, task.data)
    task.rolling_distance = 0.6
    task.axial_rotation = 5
    assert task.success(task.model, task.data)
    task.reset()
    assert task.rolling_distance == task.axial_rotation == 0
    assert not task.success(task.model, task.data)


@pytest.mark.parametrize("task_type", [SpotJugUpright, SpotJugLayDown, SpotJugRoll, SpotJugMove])
def test_no_velocity_reward_is_independent_of_jug_velocity(task_type):
    task = task_type()
    disable_velocity_rewards(task.config)
    states = np.tile(np.r_[task.data.qpos, task.data.qvel], (5, 6, 1))
    sensors = np.tile(task.data.sensordata, (5, 6, 1))
    controls = np.zeros((5, 6, task.nu))
    stationary_reward = task.reward(states, sensors, controls)
    rng = np.random.default_rng(17)
    states[..., task.object_vel_start : task.object_vel_start + 6] = rng.normal(size=(5, 6, 6)) * 7
    np.testing.assert_array_equal(task.reward(states, sensors, controls), stationary_reward)
    terms = task.reward_terms(states, sensors, controls)
    for name in ("linear_velocity", "angular_velocity", "roll", "slip"):
        if name in terms:
            assert not terms[name].any()


def test_move_has_three_metre_goal_and_no_orientation_preference():
    task = SpotJugMove()
    assert np.linalg.norm(task.config.goal_pos[:2] - task.config.start_pos[:2]) == pytest.approx(3.0)
    o = task.object_pose_start
    rewards = []
    for quat, height in [([1, 0, 0, 0], 0.2415), ([np.sqrt(0.5), np.sqrt(0.5), 0, 0], 0.14)]:
        task.data.qpos[o : o + 7] = [*task.config.goal_pos[:2], height, *quat]
        mujoco.mj_forward(task.model, task.data)
        assert task.success(task.model, task.data)
        state = np.r_[task.data.qpos, task.data.qvel][None, None]
        rewards.append(task.reward(state, task.data.sensordata[None, None], np.zeros((1, 1, task.nu)))[0])
    assert rewards[0] == pytest.approx(rewards[1])


def test_jug_mass_scales_body_mass_and_inertia():
    import mujoco

    from sumo.tasks.spot.spot_jug_manipulation import SpotJugRoll, SpotJugRollConfig

    heavy = SpotJugRoll(SpotJugRollConfig(jug_mass=3.0))
    light = SpotJugRoll(SpotJugRollConfig(jug_mass=1.0))
    bid_h = mujoco.mj_name2id(heavy.model, mujoco.mjtObj.mjOBJ_BODY, "jug")
    bid_l = mujoco.mj_name2id(light.model, mujoco.mjtObj.mjOBJ_BODY, "jug")
    assert heavy.model.body_mass[bid_h] == pytest.approx(3.0)
    assert light.model.body_mass[bid_l] == pytest.approx(1.0)
    assert heavy.model.body_inertia[bid_h] == pytest.approx(3.0 * light.model.body_inertia[bid_l])
    # Default is the 2026-09-14 real jug estimate, not the XML's 1.0 kg.
    default = SpotJugRoll()
    assert default.model.body_mass[mujoco.mj_name2id(default.model, mujoco.mjtObj.mjOBJ_BODY, "jug")] == pytest.approx(1.5)


def test_rolling_friction_and_speed_cap():
    import mujoco

    from sumo.tasks.spot.spot_jug_manipulation import SpotJugRoll, SpotJugRollConfig, SpotJugUpright

    task = SpotJugRoll(SpotJugRollConfig(rolling_friction=0.02, max_base_speed=0.3))
    gid = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_GEOM, "jug_collision")
    assert task.model.geom_friction[gid, 2] == pytest.approx(0.02)
    assert task.model.geom_condim[gid] == 6
    limits = task.actuator_ctrlrange
    assert limits[0].tolist() == [-0.3, 0.3] and limits[1].tolist() == [-0.3, 0.3]
    assert abs(limits[2, 1]) > 0.3  # yaw rate untouched
    up = SpotJugUpright()
    assert up.actuator_ctrlrange[0, 1] == pytest.approx(0.7)  # upright keeps the base bounds


@pytest.mark.parametrize("field,value", [("jug_mass", 0.0), ("jug_mass", -1.0), ("jug_mass", float("nan")), ("rolling_friction", -0.1)])
def test_invalid_construction_parameters_are_rejected(field, value):
    from sumo.tasks.spot.spot_jug_manipulation import SpotJugRoll, SpotJugRollConfig

    with pytest.raises(ValueError):
        SpotJugRoll(SpotJugRollConfig(**{field: value}))
