"""Yaw-only "look at" reward terms shared by the deployed Spot tasks.

Two targets, one cost. The cost is on the heading error between the body's forward
axis (+x, projected to the ground plane) and the body->target direction:
``w * (1 - cos(error)) * ramp(d)``. Smooth everywhere (no atan2 wrap), 0 when facing
the target, ``w`` at 90 degrees, ``2w`` facing away. It reads only the base
orientation, so nothing here asks for a lean or a step -- yaw is the only thing that
changes it (the base-velocity action still decides whether the robot turns in place
or while walking; spot_barrel_look_at additionally locks vx/vy).

``ramp(d0) = 1 - exp(-(d0 / ramp_dist)^2)`` fades the term out as the target comes
under the robot: with the target 5 cm off the body axis the "direction to it" flips
with every centimetre of body motion, and a full-weight heading cost would have the
robot spin on the spot (asked for 2026-09-16, jug tasks: the jug ends up between the
feet). At d0 = ramp_dist the weight is 63 %, at 2 x ramp_dist 98 %. ``d0`` is the
distance at the FIRST step of the horizon -- the measured state -- so the weight is a
constant per plan: scaling by the per-step distance would pay the planner for walking
onto the target (it zeroes the term), which is not what "look at it" means.

1. LOOK-AT POINT (`LookAtPointFields`): an operator-set world point, off by default.
   The planner sets `look_at_pos` / `look_at_enabled` from the monitor's "look at"
   click; tasks that carry these fields honour it, tasks that do not simply ignore
   the command (the planner logs that). Lets the operator steer heading and
   destination independently, including while a perceived object is lost.
2. LOOK-AT OBJECT (`object_look_term`): a task-side target read from the state (the
   perceived jug), used by the jug manipulation tasks.
"""

from dataclasses import dataclass

import numpy as np
from judo.utils.fields import np_1d_field


@dataclass
class LookAtPointFields:
    """Config fields for the operator look-at point. Mix into a task config."""

    look_at_enabled: bool = False
    look_at_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, 0.0]),
        names=["x", "y", "z"],
        mins=[-10.0, -10.0, 0.0],
        maxs=[10.0, 10.0, 3.0],
        vis_name="look_at_pos",
        xyz_vis_indices=[0, 1, None],
    )
    w_look_point: float = 30.0      # 90 deg off costs 30 (= 0.5 m of a w_goal=60 goal)
    look_ramp_dist: float = 0.6     # m; see ramp() above
    # Yaw-rate command floor (spot_base.yaw_command_floor): the locomotion policy ignores
    # small yaw rates, so a commanded |wz| in [deadzone, floor) is raised to `floor`
    # and below the deadzone it is 0. 0 = off. Values set from the 2026-09-16 sweep.
    # Real Spot, 2026-09-14: 0.13-0.21 rad/s -> 0 deg/s, 0.26-0.31 -> 2-7 deg/s, ~0.4 turns.
    # MuJoCo with the same policy (2026-09-16 sweep): 0.15 -> 30 %, 0.30 -> 85 %, 0.45 -> 100 %.
    # NOT a --task-set field: the planner's rollouts and the policy node map the command
    # independently and must agree, so it lives in the task defaults on both sides.
    yaw_rate_min: float = 0.4
    yaw_rate_deadzone: float = 0.1


def heading_cos(qpos: np.ndarray, body_idx: int, target_xy: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(cos(heading error), planar distance) for every state in `qpos` (..., T, nq).

    The BEARING to the target is taken once per rollout, from the first step (the
    measured state): body position and target position at t = 0. Every later step
    compares its body forward axis (+x, in the ground plane, from the wxyz quaternion)
    against that fixed bearing. Two reasons: the bearing is undefined when the body
    stands on the target, and a per-step bearing would let the planner zero the term by
    walking onto the target inside the horizon instead of turning toward it. The
    planner replans at 20 Hz, so the bearing refreshes as the robot moves.

    `target_xy` is a fixed (2,) point or a per-state (..., T, 2) array (an object read
    from the same state vector); only its first step is used.
    """
    b = body_idx
    qw, qx, qy, qz = (qpos[..., b + 3], qpos[..., b + 4], qpos[..., b + 5], qpos[..., b + 6])
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)      # body +x axis in the world frame (w, x, y, z)
    fy = 2.0 * (qx * qy + qw * qz)
    fn = np.maximum(np.hypot(fx, fy), 1e-9)
    target_xy = np.asarray(target_xy, dtype=float)
    t0 = target_xy[..., :1, :] if target_xy.ndim >= 2 else target_xy
    dx = t0[..., 0] - qpos[..., :1, b]           # (..., 1): broadcasts over the horizon
    dy = t0[..., 1] - qpos[..., :1, b + 1]
    dn = np.hypot(dx, dy)
    cos = (fx * dx + fy * dy) / (fn * np.maximum(dn, 1e-9))
    cos = np.where(dn < 1e-6, 1.0, cos)          # on the target: no heading to face
    return cos, np.broadcast_to(dn, cos.shape)


def ramp(distance: np.ndarray, ramp_dist: float) -> np.ndarray:
    """0 with the target under the robot, -> 1 beyond ~2 x ramp_dist. ramp_dist <= 0: 1."""
    distance = np.asarray(distance, dtype=float)
    if ramp_dist <= 0:
        return np.ones_like(distance)
    return 1.0 - np.exp(-np.square(distance / ramp_dist))


def look_term(qpos: np.ndarray, body_idx: int, target_xy: np.ndarray, weight: float,
              ramp_dist: float) -> np.ndarray:
    """-weight * ramp(d at step 0) * mean_t(1 - cos): shape qpos.shape[:-2].

    `qpos` is (..., T, nq); the ramp is taken from the first step so it is one number
    per rollout (see the module docstring for why not per step).
    """
    cos, d = heading_cos(qpos, body_idx, target_xy)
    return -float(weight) * ramp(d[..., 0], ramp_dist) * (1.0 - cos).mean(-1)


def point_look_term(qpos: np.ndarray, body_idx: int, config) -> np.ndarray:
    """The operator look-at point term, or zeros when disabled / not finite."""
    p = np.asarray(config.look_at_pos, dtype=float)
    if not config.look_at_enabled or not np.all(np.isfinite(p[:2])):
        return np.zeros(qpos.shape[:-2])
    return look_term(qpos, body_idx, p[:2], config.w_look_point, config.look_ramp_dist)


def object_look_term(qpos: np.ndarray, body_idx: int, object_idx: int, weight: float,
                     ramp_dist: float) -> np.ndarray:
    """Look at a free-joint object whose qpos block starts at `object_idx`."""
    target = qpos[..., object_idx : object_idx + 2]
    return look_term(qpos, body_idx, target, weight, ramp_dist)


__all__ = ["LookAtPointFields", "heading_cos", "look_term", "object_look_term",
           "point_look_term", "ramp"]
