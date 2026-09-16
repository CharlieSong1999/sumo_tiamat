"""look_at terms: heading cosine, distance ramp, the operator point on the deployed tasks."""

import numpy as np
import pytest

from sumo.tasks.spot.look_at import heading_cos, look_term, ramp


def _qpos(nq, body_idx, xy, yaw):
    q = np.zeros(nq)
    q[body_idx : body_idx + 2] = xy
    q[body_idx + 3] = np.cos(yaw / 2)
    q[body_idx + 6] = np.sin(yaw / 2)
    return q


def test_heading_cos_geometry():
    q = _qpos(7, 0, (0.0, 0.0), 0.0)[None, None]        # (1 rollout, 1 step, nq)
    cos, d = heading_cos(q, 0, np.array([2.0, 0.0]))
    assert cos[0, 0] == pytest.approx(1.0) and d[0, 0] == pytest.approx(2.0)
    cos, _ = heading_cos(q, 0, np.array([0.0, 2.0]))
    assert cos[0, 0] == pytest.approx(0.0, abs=1e-9)
    cos, _ = heading_cos(q, 0, np.array([-2.0, 0.0]))
    assert cos[0, 0] == pytest.approx(-1.0)
    q90 = _qpos(7, 0, (0.0, 0.0), np.pi / 2)[None, None]
    cos, _ = heading_cos(q90, 0, np.array([0.0, 2.0]))
    assert cos[0, 0] == pytest.approx(1.0)
    # target under the body: degenerate direction reads as facing it
    cos, _ = heading_cos(q, 0, np.array([0.0, 0.0]))
    assert cos[0, 0] == 1.0


def test_ramp_fades_under_the_robot():
    assert ramp(np.array(0.0), 0.6) == pytest.approx(0.0)
    assert ramp(np.array(0.6), 0.6) == pytest.approx(1 - np.exp(-1))
    assert ramp(np.array(2.0), 0.6) > 0.99
    assert ramp(np.array(0.0), 0.0) == 1.0


def test_look_term_shape_and_sign():
    q = np.stack([_qpos(7, 0, (0.0, 0.0), 0.0)] * 5)[None]      # (1 rollout, 5 steps, nq)
    q = np.concatenate([q, q * 0 + _qpos(7, 0, (0.0, 0.0), np.pi)[None, None]])  # 2 rollouts
    r = look_term(q, 0, np.array([2.0, 0.0]), 30.0, 0.6)
    assert r.shape == (2,)
    assert r[0] == pytest.approx(0.0, abs=1e-6)
    assert r[1] == pytest.approx(-30.0 * 2.0 * ramp(np.array(2.0), 0.6))
    # per-state target (an object read from the state) broadcasts too
    tgt = np.zeros(q.shape[:-1] + (2,))
    tgt[..., 0] = 2.0
    assert look_term(q, 0, tgt, 30.0, 0.6) == pytest.approx(r)


def _controls(task, n=1):
    return np.zeros((n, 1, task.nu))


def test_navigate_look_point_off_by_default_and_on_when_set():
    from sumo.tasks.spot.spot_navigate_look import SpotNavigateLook

    task = SpotNavigateLook()
    s = np.zeros((2, 1, task.model.nq + task.model.nv))
    s[..., : task.model.nq] = task.reset_pose[: task.model.nq]
    b = task.body_pose_idx
    s[..., b : b + 2] = 0.0
    s[0, :, b + 3 : b + 7] = [1.0, 0.0, 0.0, 0.0]   # facing +x (reset_pose may carry a random yaw)
    s[1, :, b + 3 : b + 7] = [0.0, 0.0, 0.0, 1.0]   # facing -x
    sens = np.zeros((2, 1, task.model.nsensordata))
    off = task.reward(s, sens, _controls(task, 2))
    assert off[0] == pytest.approx(off[1])          # heading is free when the point is off
    task.config.look_at_enabled = True
    task.config.look_at_pos = np.array([2.0, 0.0, 0.0]) + task.config.look_at_pos * 0
    on = task.reward(s, sens, _controls(task, 2))
    assert on[0] > on[1]                            # facing +x (toward the point) wins
    assert on[0] == pytest.approx(off[0])           # facing it costs nothing extra


def test_ramp_is_taken_from_the_first_step_only():
    # Walking onto the target inside the horizon must not shrink the penalty.
    q_far = np.stack([_qpos(7, 0, (0.0, 0.0), np.pi)] * 4)          # facing away, 1 m off
    q_walk = q_far.copy()
    q_walk[1:, 0] = [0.5, 0.9, 1.0]                                    # steps onto the target
    r_far = look_term(q_far[None], 0, np.array([1.0, 0.0]), 10.0, 0.6)
    r_walk = look_term(q_walk[None], 0, np.array([1.0, 0.0]), 10.0, 0.6)
    assert r_walk[0] == pytest.approx(r_far[0])   # no gain from approaching: heading unchanged
    assert r_far[0] == pytest.approx(-10.0 * 2.0 * ramp(np.array(1.0), 0.6))


def test_look_at_task_point_overrides_barrel_without_double_count():
    from sumo.tasks.spot.spot_barrel_look_at import SpotBarrelLookAt

    task = SpotBarrelLookAt()
    s = np.zeros((2, 1, task.model.nq + task.model.nv))
    s[..., : task.model.nq] = task.reset_pose[: task.model.nq]
    b, o = task.body_pose_idx, task.barrel_pose_idx
    s[..., b : b + 2] = 0.0
    s[..., o : o + 2] = [2.0, 0.0]                        # barrel at +x
    s[0, :, b + 3 : b + 7] = [1.0, 0.0, 0.0, 0.0]           # facing +x
    s[1, :, b + 3 : b + 7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]   # facing +y
    sens = np.zeros((2, 1, task.model.nsensordata))
    u = _controls(task, 2)
    r_barrel = task.reward(s, sens, u)
    assert r_barrel[0] > r_barrel[1]                       # barrel: facing +x wins
    task.config.look_at_enabled = True
    task.config.look_at_pos = np.array([0.0, 2.0, 0.0])    # operator point at +y
    r_point = task.reward(s, sens, u)
    assert r_point[1] > r_point[0]                         # point: facing +y wins
    # exactly one look term is paid: reward == navigate part + w_look * ramp * (1 - cos)
    nav = task.navigate_reward(s, sens, u)
    fade = ramp(np.array(2.0), task.config.look_ramp_dist)
    assert r_point[1] == pytest.approx(nav[1], abs=1e-6)                       # facing it: no cost
    assert r_point[0] == pytest.approx(nav[0] - task.config.w_look * fade * 1.0, abs=1e-6)


def test_perceive_and_look_at_tasks_honour_the_point():
    from sumo.tasks.spot.spot_barrel_look_at import SpotBarrelLookAt
    from sumo.tasks.spot.spot_barrel_perceive import SpotBarrelPerceive

    for cls in (SpotBarrelPerceive, SpotBarrelLookAt):
        task = cls()
        assert hasattr(task.config, "look_at_enabled") and not task.config.look_at_enabled
        s = np.zeros((2, 1, task.model.nq + task.model.nv))
        s[..., : task.model.nq] = task.reset_pose[: task.model.nq]
        b = task.body_pose_idx
        s[..., b : b + 2] = 0.0
        s[0, :, b + 3 : b + 7] = [1.0, 0.0, 0.0, 0.0]
        s[1, :, b + 3 : b + 7] = [0.0, 0.0, 0.0, 1.0]
        sens = np.zeros((2, 1, task.model.nsensordata))
        task.config.look_at_enabled = True
        task.config.look_at_pos = np.array([3.0, 0.0, 0.0])
        r = task.reward(s, sens, _controls(task, 2))
        assert r[0] > r[1], cls.__name__


def test_jug_tasks_face_the_jug_unless_it_is_underfoot():
    from sumo.tasks.spot.spot_jug_manipulation import SpotJugRoll

    task = SpotJugRoll()
    s = np.zeros((3, 1, task.model.nq + task.model.nv))
    s[..., : task.model.nq] = task.reset_pose[: task.model.nq]
    b, o = task.body_pose_start, task.object_pose_start
    s[..., b : b + 2] = 0.0
    s[..., o : o + 2] = [0.95, 0.0]
    s[0, :, b + 3 : b + 7] = [1.0, 0.0, 0.0, 0.0]       # facing the jug at +x
    s[1, :, b + 3 : b + 7] = [0.0, 0.0, 0.0, 1.0]       # facing away from the jug at +x
    s[2, :, o : o + 2] = [0.05, 0.0]                      # jug under the robot, facing away
    s[2, :, b + 3 : b + 7] = [0.0, 0.0, 0.0, 1.0]
    sens = np.zeros((3, 1, task.model.nsensordata))
    terms = task.reward_terms(s, sens, np.zeros((3, 1, task.nu)))
    look = terms["look"]
    assert look[0] == pytest.approx(0.0, abs=1e-6)
    expected = -task.config.w_look_object * 2.0 * ramp(np.array(0.95), task.config.look_ramp_dist)
    assert look[1] == pytest.approx(expected, rel=1e-6)   # 2w at 180 deg, ramped at 0.95 m
    assert abs(look[2]) < 0.2                              # faded: jug 5 cm away


def test_yaw_command_floor_mapping():
    from sumo.tasks.spot.spot_base import yaw_command_floor

    wz = np.array([0.0, 0.05, 0.12, 0.3, -0.2, -0.6, 0.9])
    out = yaw_command_floor(wz, floor=0.45, deadzone=0.1)
    assert out.tolist() == [0.0, 0.0, 0.45, 0.45, -0.45, -0.6, 0.9]
    assert yaw_command_floor(wz, floor=0.0, deadzone=0.1).tolist() == wz.tolist()   # off


def test_yaw_floor_reaches_the_policy_command_only_when_enabled():
    from sumo.tasks.spot.spot_navigate_look import SpotNavigateLook

    task = SpotNavigateLook()
    u = np.zeros((2, 3, task.nu))
    u[..., 2] = 0.2
    assert (task.config.yaw_rate_min, task.config.yaw_rate_deadzone) == (0.4, 0.1)   # deployed default
    assert task.task_to_sim_ctrl(u)[..., 2].tolist() == [[0.4] * 3] * 2     # navigate_look: floor, point or not
    task.config.yaw_rate_min = 0.0
    assert task.task_to_sim_ctrl(u)[..., 2].tolist() == [[0.2] * 3] * 2     # off: raw command
    task.config.yaw_rate_min, task.config.yaw_rate_deadzone = 0.45, 0.1
    assert task.task_to_sim_ctrl(u)[..., 2].tolist() == [[0.45] * 3] * 2
    u[..., 2] = 0.05
    assert task.task_to_sim_ctrl(u)[..., 2].tolist() == [[0.0] * 3] * 2
    assert task.task_to_sim_ctrl(u)[..., 0].tolist() == [[0.0] * 3] * 2     # vx untouched


def test_yaw_floor_never_exceeds_the_task_cap_and_kick_shares_it():
    from sumo.tasks.spot.spot_barrel_look_at import SpotBarrelLookAt, SpotBarrelLookAtConfig
    from sumo.tasks.spot.spot_jug_kick import SpotJugKick

    capped = SpotBarrelLookAt(SpotBarrelLookAtConfig(max_yaw_rate=0.2))
    u = np.zeros((1, 1, capped.nu))
    u[..., 2] = 0.15
    assert capped.task_to_sim_ctrl(u)[..., 2].item() == pytest.approx(0.2)   # floor clamped to the cap
    kick = SpotJugKick()
    assert (kick.config.yaw_rate_min, kick.config.yaw_rate_deadzone) == (0.4, 0.1)
    u = np.zeros((1, 1, kick.nu))
    u[..., 2] = 0.2
    assert kick.task_to_sim_ctrl(u)[..., 2].item() == pytest.approx(0.4)


def _standing_state(task, xy):
    q = np.array(task.reset_pose, dtype=float)
    b = task.body_pose_idx
    q[b : b + 3] = [xy[0], xy[1], 0.52]
    q[b + 3 : b + 7] = [1.0, 0.0, 0.0, 0.0]
    return np.concatenate([q, np.zeros(task.model.nv)])[None, None, :]


def test_hold_tasks_have_a_flat_goal_well_and_prefer_zero_commands():
    """Real robot 2026-09-15: a 7 cm goal offset kept the legs shuffling at 0.1 m/s commands."""
    from sumo.tasks.spot.spot_barrel_perceive import SpotBarrelPerceive
    from sumo.tasks.spot.spot_navigate_look import SpotNavigateLook

    for task in (SpotBarrelPerceive(), SpotNavigateLook()):
        task.config.goal_pos = np.array([0.0, 0.0, 0.52])
        sensors = np.zeros((1, 1, task.model.nsensordata))
        zero = np.zeros((1, 1, task.nu))
        r_on = task.reward(_standing_state(task, (0.0, 0.0)), sensors, zero)[0]
        r_7cm = task.reward(_standing_state(task, (0.07, 0.0)), sensors, zero)[0]
        r_20cm = task.reward(_standing_state(task, (0.20, 0.0)), sensors, zero)[0]
        assert r_7cm == pytest.approx(r_on)                                   # inside the well: flat
        assert r_on - r_20cm == pytest.approx(task.config.w_goal * 0.10)      # outside: full gradient past 10 cm
        small = np.zeros((1, 1, task.nu))
        small[..., :2] = 0.1
        assert r_7cm - task.reward(_standing_state(task, (0.07, 0.0)), sensors, small)[0] == pytest.approx(
            task.config.w_controls * np.hypot(0.1, 0.1))                        # planar commands cost
        yaw_only = np.zeros((1, 1, task.nu))
        yaw_only[..., 2] = 0.3
        r_yaw = task.reward(_standing_state(task, (0.07, 0.0)), sensors, yaw_only)[0]
        assert r_7cm - r_yaw == pytest.approx(task.config.w_yaw_hold * 0.3)      # holding: yaw costs
        task.config.look_at_enabled = True                                        # point set: yaw is free
        r_on_pt = task.reward(_standing_state(task, (0.07, 0.0)), sensors, zero)[0]
        r_yaw_pt = task.reward(_standing_state(task, (0.07, 0.0)), sensors, yaw_only)[0]
        assert r_yaw_pt == pytest.approx(r_on_pt)
        task.config.look_at_enabled = False


def test_hold_yaw_deadband_on_the_waiting_task():
    from sumo.tasks.spot.spot_barrel_perceive import SpotBarrelPerceive
    from sumo.tasks.spot.spot_navigate_look import SpotNavigateLook

    u = np.zeros((2, 3))
    u[0, 2], u[1, 2] = 0.25, -0.35
    out = np.asarray(SpotBarrelPerceive().task_to_sim_ctrl(u))
    assert out[:, 2].tolist() == pytest.approx([0.0, -0.35])               # waiting: under 0.3 is noise
    out = np.asarray(SpotNavigateLook().task_to_sim_ctrl(u))
    assert out[:, 2].tolist() == pytest.approx([0.4, -0.4])                # steering: floored


def test_planar_speed_deadband_reaches_the_policy_command():
    from sumo.tasks.spot.spot_barrel_perceive import SpotBarrelPerceive
    from sumo.tasks.spot.spot_jug_manipulation import SpotJugRoll

    for task in (SpotBarrelPerceive(), SpotJugRoll()):
        assert task.config.xy_speed_deadzone == 0.08
        u = np.zeros((3, task.nu))
        u[0, :2] = (0.03, 0.03)      # |v| 0.042: noise -> both zero
        u[1, :2] = (0.10, 0.00)      # above: untouched
        u[2, :2] = (0.06, 0.06)      # |v| 0.085: kept
        out = np.asarray(task.task_to_sim_ctrl(u))
        assert out[0, :2].tolist() == [0.0, 0.0]
        assert out[1, :2].tolist() == pytest.approx([0.10, 0.0])
        assert out[2, :2].tolist() == pytest.approx([0.06, 0.06])
