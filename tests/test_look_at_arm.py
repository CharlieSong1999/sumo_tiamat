"""spot_barrel_look_at_arm / spot_navigate_look_arm: the nu=11 variants of the two look tasks.

Arm + gripper in the action space so they can share a policy session with the arm tasks.
"""

import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.spot.arm_hold import ARM_DOF, ARM_QPOS_OFFSET, FINGER_CTRL_INDEX, arm_hold_pose
from sumo.tasks.spot.spot_barrel_look_at import YAW_RATE_INDEX, SpotBarrelLookAt, SpotBarrelLookAtArm
from sumo.tasks.spot.spot_constants import ARM_STOWED_POS, ARM_UNSTOWED_POS, GRIPPER_CLOSED_POS
from sumo.tasks.spot.spot_jug_manipulation import SpotJugArmIdle
from sumo.tasks.spot.spot_navigate_look import SpotNavigateLook, SpotNavigateLookArm

ARM = ["spot_barrel_look_at_arm", "spot_navigate_look_arm"]


@pytest.fixture(scope="module")
def tasks():
    return {"spot_barrel_look_at_arm": SpotBarrelLookAtArm(), "spot_navigate_look_arm": SpotNavigateLookArm()}


@pytest.fixture(scope="module")
def base_tasks():
    return {"spot_barrel_look_at_arm": SpotBarrelLookAt(), "spot_navigate_look_arm": SpotNavigateLook()}


def _state(task, arm=None, yaw=0.0):
    q = np.array(task.reset_pose, dtype=float)
    b = task.body_pose_idx
    q[b : b + 3] = [0.0, 0.0, 0.52]
    q[b + 3 : b + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    a = b + ARM_QPOS_OFFSET
    q[a : a + ARM_DOF] = arm_hold_pose(task) if arm is None else arm
    return np.concatenate([q, np.zeros(task.model.nv)])[None, None, :]


@pytest.mark.parametrize("name", ARM)
def test_registered_nu11_same_scene_as_the_nu3_task(name, tasks, base_tasks):
    assert name in get_registered_tasks()
    t, b = tasks[name], base_tasks[name]
    assert b.nu == 3 and t.nu == SpotJugArmIdle().nu == 11
    assert (t.model.nq, t.model.nv) == (b.model.nq, b.model.nv)  # the model is the same
    assert getattr(t, "perceived_object_joints", ()) == getattr(b, "perceived_object_joints", ())
    assert t.body_pose_idx == b.body_pose_idx
    assert t.yaw_floor_enabled == b.yaw_floor_enabled


def test_arm_qpos_slice_is_the_arm(tasks):
    t = tasks["spot_barrel_look_at_arm"]
    a = t.body_pose_idx + ARM_QPOS_OFFSET
    names = [
        t.model.joint(t.model.dof_jntid[t.model.jnt_dofadr[j]]).name
        for j in range(t.model.njnt)
        if a <= t.model.jnt_qposadr[j] < a + ARM_DOF
    ]
    assert len(names) == ARM_DOF and all("arm" in n for n in names), names


@pytest.mark.parametrize("name", ARM)
def test_arm_actions_reach_the_policy_command(name, tasks):
    t = tasks[name]
    u = np.zeros((1, t.nu))
    u[0, 3:10] = 0.2
    cmd = np.asarray(t.task_to_sim_ctrl(u)).reshape(-1)
    assert cmd.shape == (25,)
    np.testing.assert_allclose(cmd[3:10], 0.2)  # arm_cmd slot
    assert np.allclose(cmd[:3], 0.0)


@pytest.mark.parametrize("name", ARM)
def test_holding_the_reset_arm_scores_higher(name, tasks):
    t = tasks[name]
    u = np.zeros((1, 1, t.nu))
    held = t.reward(_state(t), np.zeros((1, 1, 0)), u)
    moved = t.reward(_state(t, arm=arm_hold_pose(t) + 0.3), np.zeros((1, 1, 0)), u)
    assert held.shape == (1,) and moved > -np.inf
    assert held[0] > moved[0]
    hold = arm_hold_pose(t)
    np.testing.assert_allclose(hold[:6], np.asarray(ARM_UNSTOWED_POS)[:6])  # the family's neutral pose
    assert hold[6] == GRIPPER_CLOSED_POS


@pytest.mark.parametrize("name", ARM)
def test_hold_pose_is_commandable_and_stow_is_not(name, tasks):
    """The reset pose must lie inside judo's arm command bounds; ARM_STOWED_POS does not.

    Holding the stowed pose would make the planner pull the arm to the clipped bounds (codex, 2026-09-18).
    """
    t = tasks[name]
    lim = t.actuator_ctrlrange
    hold = arm_hold_pose(t)
    assert np.all(lim[3:9, 0] <= hold[:6]) and np.all(hold[:6] <= lim[3:9, 1])
    stow = np.asarray(ARM_STOWED_POS)[:6]
    assert not (np.all(lim[3:9, 0] <= stow) and np.all(stow <= lim[3:9, 1]))


@pytest.mark.parametrize("name", ARM)
def test_finger_command_is_pinned_closed(name, tasks):
    t = tasks[name]
    lim = t.actuator_ctrlrange
    np.testing.assert_allclose(lim[FINGER_CTRL_INDEX], [GRIPPER_CLOSED_POS, GRIPPER_CLOSED_POS])
    assert lim.shape == (11, 2)
    # a control at the bounds' midpoint with the selector positive (no forced closure) still closes
    u = (lim[:, 0] + lim[:, 1])[None] / 2.0
    u[0, 10] = 1.0
    cmd = np.asarray(t.task_to_sim_ctrl(u)).reshape(-1)
    assert cmd[9] == GRIPPER_CLOSED_POS


@pytest.mark.parametrize("name", ARM)
def test_rewards_agree_with_the_nu3_task_when_the_arm_is_held(name, tasks, base_tasks):
    t, b = tasks[name], base_tasks[name]
    u = np.zeros((1, 1, t.nu))
    u3 = u[..., :3]
    r_arm = t.reward(_state(t, yaw=0.5), np.zeros((1, 1, 0)), u)
    r_base = b.reward(_state(t, yaw=0.5), np.zeros((1, 1, 0)), u3)
    np.testing.assert_allclose(r_arm, r_base)  # hold term is zero at the hold pose


def test_look_at_arm_keeps_the_yaw_limits(tasks):
    t = tasks["spot_barrel_look_at_arm"]
    lim = t.actuator_ctrlrange
    assert lim.shape[0] == t.nu
    np.testing.assert_allclose(lim[:YAW_RATE_INDEX], 0.0)
    np.testing.assert_allclose(lim[YAW_RATE_INDEX], [-t.config.max_yaw_rate, t.config.max_yaw_rate])


def test_facing_the_barrel_scores_higher_with_the_arm(tasks):
    t = tasks["spot_barrel_look_at_arm"]
    u = np.zeros((1, 1, t.nu))
    s0, s1 = _state(t, yaw=0.0), _state(t, yaw=1.2)
    o = t.barrel_pose_idx
    s0[..., o : o + 2] = s1[..., o : o + 2] = [2.0, 0.0]
    assert t.reward(s0, np.zeros((1, 1, 0)), u)[0] > t.reward(s1, np.zeros((1, 1, 0)), u)[0]
