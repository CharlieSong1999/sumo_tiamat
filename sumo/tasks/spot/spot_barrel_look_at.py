"""spot_barrel_look_at: stand where you are and keep facing the perceived barrel.

Built on spot_barrel_perceive (same scene, same perceived free-joint barrel, same nu=3
base-velocity action space as the deployed spot_navigate). Two things are added:

1. A LOOK-AT term in the reward: the body's forward axis (its +x, projected to the
   ground plane) should point at the barrel. Paid as a penalty on the heading error,
   ``-w_look * (1 - cos(error))``: 0 when facing the barrel, ``w_look`` at 90 degrees,
   ``2 * w_look`` when facing away. Smooth everywhere, no atan2 wrap.
2. A CAP on the yaw rate the planner may command, two ways at once: a HARD bound on the
   yaw-rate action (``max_yaw_rate``, narrowing the +-0.7 rad/s the base task allows), so
   no sampled rollout can even try a fast turn; and a SOFT cost on |yaw rate| in the
   reward (``w_yaw_rate``), so among legal turns the slower one wins.
3. YAW ONLY (``lock_xy``, default on): the vx and vy actions are hard-bounded to 0, so the
   only thing the planner can do is turn in place. Asked for on the real robot
   (2026-09-07): with a goal at the current position the navigate term still let the
   optimiser trade small steps and a forward lean against the heading error; turning
   in place is the whole task.

Everything else -- the navigate-to-goal term (constant under lock_xy), the fall penalty,
the perceived-object plumbing -- is inherited unchanged. With the planner's
``--goal-relative 0 0`` this is "stand still, turn to face the barrel, slowly".
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from sumo.tasks.spot.spot_barrel_perceive import SpotBarrelPerceive, SpotBarrelPerceiveConfig

YAW_RATE_INDEX = 2  # compact control = [vx, vy, wz]; see SpotBase.task_to_sim_ctrl [0:3]


@dataclass
class SpotBarrelLookAtConfig(SpotBarrelPerceiveConfig):
    """spot_barrel_perceive's config plus the look-at weight and the yaw-rate limits."""

    w_look: float = 30.0          # heading-error weight: 90 deg off costs 30 (= 0.5 m of goal)
    w_yaw_rate: float = 5.0       # soft cost per rad/s of commanded yaw rate
    max_yaw_rate: float = 0.4     # hard bound on the yaw-rate action, rad/s (base task: 0.7)
    lock_xy: bool = True          # vx = vy = 0: turn in place, never step or lean into it


class SpotBarrelLookAt(SpotBarrelPerceive):
    """Perceived-barrel scene whose reward wants the robot facing the barrel, turning slowly."""

    name = "spot_barrel_look_at"
    config_t: type[SpotBarrelLookAtConfig] = SpotBarrelLookAtConfig
    config: SpotBarrelLookAtConfig

    def __init__(self, config: SpotBarrelLookAtConfig | None = None) -> None:
        super().__init__(config=config)
        self.barrel_pose_idx = self.get_joint_position_start_index("barrel_joint")

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """The base task's bounds with the yaw-rate row narrowed to +-max_yaw_rate and,
        under lock_xy, the vx/vy rows collapsed to [0, 0]."""
        limits = np.array(super().actuator_ctrlrange, dtype=float, copy=True)
        cap = float(self.config.max_yaw_rate)
        limits[YAW_RATE_INDEX] = [max(limits[YAW_RATE_INDEX, 0], -cap),
                                  min(limits[YAW_RATE_INDEX, 1], cap)]
        if self.config.lock_xy:
            limits[:YAW_RATE_INDEX] = 0.0
        return limits

    def heading_cos(self, qpos: np.ndarray) -> np.ndarray:
        """cos(angle between the body's forward axis and the body->barrel direction), in
        the ground plane. Shape = qpos.shape[:-1]. 1 = facing the barrel, -1 = facing away.
        Standing ON the barrel (degenerate direction) reads as facing it."""
        b = self.body_pose_idx
        qw, qx, qy, qz = (qpos[..., b + 3], qpos[..., b + 4], qpos[..., b + 5], qpos[..., b + 6])
        # Body +x axis in the world frame, from the (w, x, y, z) quaternion.
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qw * qz)
        fn = np.maximum(np.hypot(fx, fy), 1e-9)
        o = self.barrel_pose_idx
        dx = qpos[..., o] - qpos[..., b]
        dy = qpos[..., o + 1] - qpos[..., b + 1]
        dn = np.hypot(dx, dy)
        cos = (fx * dx + fy * dy) / (fn * np.maximum(dn, 1e-9))
        return np.where(dn < 1e-6, 1.0, cos)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Navigate reward (inherited) - heading error to the barrel - |yaw rate|."""
        base = super().reward(states, sensors, controls, system_metadata)
        qpos = states[..., : self.model.nq]
        look_reward = -self.config.w_look * (1.0 - self.heading_cos(qpos)).mean(-1)
        yaw_rate_reward = -self.config.w_yaw_rate * np.abs(controls[..., YAW_RATE_INDEX]).mean(-1)
        assert look_reward.shape == base.shape
        assert yaw_rate_reward.shape == base.shape
        return base + look_reward + yaw_rate_reward
