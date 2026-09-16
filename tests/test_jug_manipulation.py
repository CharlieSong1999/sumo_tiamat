import mujoco
import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.spot.spot_jug_manipulation import (
    SpotJugLayDown,
    SpotJugMove,
    SpotJugRoll,
    SpotJugRollConfig,
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
    np.testing.assert_allclose(coupled[:, 0], [0, 0, min(0.5, task.config.roll_speed_cap), 0])   # capped
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


def test_arm_idle_is_a_neutral_nu11_task():
    from sumo.tasks.spot.spot_jug_manipulation import SpotJugArmIdle, SpotJugUpright

    idle, up = SpotJugArmIdle(), SpotJugUpright()
    assert idle.nu == up.nu == 11
    assert idle.model.nq == up.model.nq          # same scene (jug_joint), switchable layout
    s = np.zeros((2, 1, idle.model.nq + idle.model.nv))
    s[..., : idle.model.nq] = idle.reset_pose[: idle.model.nq]
    b = idle.body_pose_start
    s[..., b : b + 2] = 0.0
    s[1, :, b + 7 + 12] += 0.8                    # arm_sh0 swung away from stowed
    sens = np.zeros((2, 1, idle.model.nsensordata))
    r = idle.reward(s, sens, np.zeros((2, 1, idle.nu)))
    assert r[0] > r[1]                            # stowed arm wins
    # moving the jug changes nothing for the idle task
    s2 = s.copy()
    s2[..., idle.object_pose_start : idle.object_pose_start + 2] += 1.0
    assert idle.reward(s2, sens, np.zeros((2, 1, idle.nu))) == pytest.approx(r)


def test_water_default_and_synthesized_pooling():
    """Opt-in water: five 0.1 kg balls (~0.5 kg of stones) the planner places inside the perceived jug."""
    from sumo.tasks.spot.jug_water import PROFILE, pooled_local_positions
    assert SpotJugRoll().synthesized_joints == ()                                # deployed default: no water
    task = SpotJugRoll(SpotJugRollConfig(water_fill_ratio=0.026))
    assert task.water_count == 5 and task.water_mass == pytest.approx(0.495, abs=0.01)
    assert task.synthesized_joints == tuple(f"jug_water_{i}_joint" for i in range(5))
    wall, z_bot, z_top = PROFILE[0][1], PROFILE[0][0], PROFILE[1][0]
    r = task.water_radius
    for quat in ([1, 0, 0, 0], [np.sqrt(0.5), 0, -np.sqrt(0.5), 0], [np.sqrt(0.5), np.sqrt(0.5), 0, 0]):
        local = pooled_local_positions(5, r, quat)
        assert (np.hypot(local[:, 0], local[:, 1]) <= wall - r + 1e-6).all()     # inside the side wall
        assert (local[:, 2] >= z_bot + r - 1e-6).all() and (local[:, 2] <= z_top - r + 1e-6).all()
        d = np.linalg.norm(local[1:] - local[:-1], axis=-1)
        assert (d >= 2 * r - 1e-9).all()                                         # no overlap
    # synthesize_qpos: balls land inside the jug at the jug's world pose
    q = np.array(task.reset_pose, dtype=float)
    o = task.object_pose_start
    q[o : o + 3] = [1.3, -0.5, 0.14]
    q[o + 3 : o + 7] = [np.sqrt(0.5), 0, -np.sqrt(0.5), 0]                        # lying
    task.synthesize_qpos(q)
    for name in task.synthesized_joints:
        adr = task.model.jnt_qposadr[task.model.joint(name).id]
        assert np.linalg.norm(q[adr : adr + 3] - q[o : o + 3]) < 0.25
        assert q[adr + 3 : adr + 7].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_pushing_modes_arrive_deadband_and_standoff():
    """Inside arrive_tolerance the jug is left alone: flat position cost, no approach/roll, standoff on."""
    task = SpotJugRoll()
    c = task.config
    c.goal_pos = np.array([1.5, -0.5, 0.14])
    sensors = np.zeros((1, 1, task.model.nsensordata))
    u = np.zeros((1, 1, task.nu))
    def state(jug_xy, body_xy):
        q = np.array(task.reset_pose, dtype=float)
        b, o = task.body_pose_start, task.object_pose_start
        q[b : b + 3] = [body_xy[0], body_xy[1], 0.52]
        q[o : o + 3] = [jug_xy[0], jug_xy[1], 0.14]
        q[o + 3 : o + 7] = [np.sqrt(0.5), 0, -np.sqrt(0.5), 0]
        return np.concatenate([q, np.zeros(task.model.nv)])[None, None, :]
    lying_axis = np.zeros((1, 1, task.model.nsensordata))
    # jug 15 cm from the goal (arrived): position flat, approach/roll zero, standoff active when close
    t_far = task.reward_terms(state((1.35, -0.5), (0.2, -0.5)), sensors, u)
    t_near = task.reward_terms(state((1.35, -0.5), (1.0, -0.5)), sensors, u)
    assert t_far["position"] == pytest.approx(0.0) and t_near["position"] == pytest.approx(0.0)
    assert t_far["approach"] == pytest.approx(0.0) and t_far["roll"] == pytest.approx(0.0)
    assert t_far["standoff"] == pytest.approx(0.0)                         # 1.15 m away: fine
    assert t_near["standoff"] == pytest.approx(-c.w_standoff * (c.standoff_dist - 0.35))   # 0.35 m: penalised
    # jug 1 m from the goal (not arrived): position cost past the tolerance, no standoff
    t_go = task.reward_terms(state((0.5, -0.5), (0.2, -0.5)), sensors, u)
    assert t_go["position"] == pytest.approx(-c.w_position * (1.0 - c.arrive_tolerance))
    assert t_go["standoff"] == pytest.approx(0.0)
    # 5 cm past the tolerance: the push terms are only 1/6 armed, the standoff 5/6
    t_edge = task.reward_terms(state((1.2, -0.5), (0.85, -0.5)), sensors, u)
    assert t_edge["standoff"] == pytest.approx(-c.w_standoff * (c.standoff_dist - 0.35) * (1 - 0.05 / c.arrive_ramp))
    assert abs(t_edge["approach"]) < abs(task.reward_terms(state((0.5, -0.5), (0.85, -0.5)), sensors, u)["approach"])


def test_roll_speed_is_capped_and_overspeed_costs():
    task = SpotJugRoll()
    c = task.config
    sensors = np.zeros((1, 1, task.model.nsensordata))
    sensors[..., task.object_z_axis_start : task.object_z_axis_start + 3] = [1.0, 0.0, 0.0]   # lying, axis along x
    c.goal_pos = np.array([3.0, 0.0, 0.14])
    c.start_pos = np.array([0.95, 0.0, 0.0])
    u = np.zeros((1, 1, task.nu))
    def state(vx):
        q = np.array(task.reset_pose, dtype=float)
        o, vo = task.object_pose_start, task.object_vel_start      # both index the full state
        q[o : o + 3] = [1.5, 0.0, 0.14]
        st = np.concatenate([q, np.zeros(task.model.nv)])
        st[vo] = vx                                  # jug sliding/rolling toward the goal at vx
        return st[None, None, :]
    slow, fast = task.reward_terms(state(0.2), sensors, u), task.reward_terms(state(0.8), sensors, u)
    assert slow["overspeed"] == pytest.approx(0.0)
    assert fast["overspeed"] == pytest.approx(-c.w_overspeed * (0.8 - c.roll_speed_cap))
    assert fast["roll"] <= c.w_roll * c.roll_speed_cap + 1e-9              # nothing extra for speed past the cap


def test_roll_term_is_gated_by_arrival_with_a_really_rolling_jug():
    task = SpotJugRoll()
    c = task.config
    c.goal_pos = np.array([3.0, 0.0, 0.14])
    c.start_pos = np.array([0.95, 0.0, 0.0])
    sensors = np.zeros((1, 1, task.model.nsensordata))
    sensors[..., task.object_z_axis_start : task.object_z_axis_start + 3] = [0.0, -1.0, 0.0]   # axis along -y: rolls along x
    u = np.zeros((1, 1, task.nu))
    def rolling_state(x):
        q = np.array(task.reset_pose, dtype=float)
        o, vo = task.object_pose_start, task.object_vel_start
        q[o : o + 3] = [x, 0.0, 0.14]
        st = np.concatenate([q, np.zeros(task.model.nv)])
        st[vo] = 0.25                                  # forward 0.25 m/s ...
        st[vo + 5] = -0.25 / c.radius                  # ... coupled spin about the body z axis
        return st[None, None, :]
    far, arrived = task.reward_terms(rolling_state(1.5), sensors, u), task.reward_terms(rolling_state(2.9), sensors, u)
    assert far["roll"] > 0.0                              # coupled rolling toward B is rewarded ...
    assert arrived["roll"] == pytest.approx(0.0)          # ... until the jug has arrived
    assert far["overspeed"] == pytest.approx(0.0)


def test_arrive_ramp_must_be_positive():
    with pytest.raises(ValueError):
        SpotJugRoll(SpotJugRollConfig(arrive_ramp=0.0))


def test_pooled_water_stays_inside_the_cavity_at_any_tilt():
    from sumo.tasks.spot.jug_water import PROFILE, pooled_local_positions
    wall, z_bot, z_top = PROFILE[0][1], PROFILE[0][0], PROFILE[1][0]
    r = 0.025
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        for n in (5, 19):
            p = pooled_local_positions(n, r, q)
            assert (np.hypot(p[:, 0], p[:, 1]) <= wall - r + 1e-6).all()
            assert (p[:, 2] >= z_bot + r - 1e-6).all() and (p[:, 2] <= z_top - r + 1e-6).all()
