"""Arm-hold reward term for the nu=11 variants of the locomotion tasks.

`spot_barrel_look_at` / `spot_navigate_look` are nu=3 (base velocity only); the deployed
policy honours task switches only within one action width, so they cannot share a policy
session with the arm tasks (spot_jug_arm_idle, spot_jug_roll_arm_gentle_*). The `_arm`
variants (2026-09-18) keep the rewards and add the arm + gripper to the action space
(nu=11 = base 3 + arm 6 + finger 1 + gripper selector 1, judo's SpotBase layout), paying a
quadratic cost for the arm joints leaving a hold pose so the planner keeps it still.

The hold pose is the task's reset pose (ARM_UNSTOWED_POS under use_arm, what
spot_jug_arm_idle holds and the roll profiles start from), with the finger closed. Not
ARM_STOWED_POS: the stowed joint angles lie OUTSIDE judo's arm command bounds, so a planner
told to hold them would command the clipped bounds and pull the arm away from stow
(codex review, 2026-09-18). The finger command is pinned closed in the bounds as well:
judo's gripper selector only forces closure when negative, and CEM averages the raw
controls, so an unpinned finger can be commanded open by the average of closed elites.
"""

import numpy as np

from sumo.tasks.spot.spot_constants import GRIPPER_CLOSED_POS

ARM_QPOS_OFFSET = 7 + 12  # after the base free joint (7) and the 12 leg joints
ARM_DOF = 7  # sh0 sh1 el0 el1 wr0 wr1 f1x (spot_jug_manipulation._arm_idle_reward)
FINGER_CTRL_INDEX = 9  # compact control: base [0:3], arm joints [3:9], finger [9], selector [10]


def arm_hold_pose(task) -> np.ndarray:
    """The 7 arm joint angles the reward holds: the task's reset arm pose, finger closed."""
    hold = np.array(task.reset_arm_pos, dtype=float)
    hold[6] = float(GRIPPER_CLOSED_POS)
    return hold


def arm_hold_term(qpos: np.ndarray, body_pose_idx: int, hold: np.ndarray, w_arm_hold: float) -> np.ndarray:
    """-w * sum((arm - hold)^2), averaged over the horizon. Shape = qpos.shape[:-2]."""
    a = body_pose_idx + ARM_QPOS_OFFSET
    arm = qpos[..., a : a + ARM_DOF]
    return -float(w_arm_hold) * np.square(arm - np.asarray(hold)[None, None]).sum(-1).mean(-1)


def pin_finger_closed(limits: np.ndarray) -> np.ndarray:
    """Copy of the task's control bounds with the finger row fixed at GRIPPER_CLOSED_POS."""
    out = np.array(limits, dtype=float, copy=True)
    out[FINGER_CTRL_INDEX] = [float(GRIPPER_CLOSED_POS), float(GRIPPER_CLOSED_POS)]
    return out
