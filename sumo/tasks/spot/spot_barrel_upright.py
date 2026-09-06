# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""SpotBarrelUpright task - upright a barrel that starts lying on its side.

The barrel resets lying flat on the ground (its cylinder axis horizontal); the robot must
nudge / lever it back to a standing (upright) pose. Reward = object-z-axis-upright + gripper
proximity, mirroring spot_cone_upright.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from sumo.tasks.spot.spot_upright import (
    Z_AXIS,
    SpotUprightConfig,
    gripper_distance_reward,
    quat_to_mat,
    random_object_pose,
    sample_annulus_xy,
    z_axis_orientation_reward,
)

# Reuse the existing barrel scene (already exposes object_z_axis + sensor_arm_link_fngr via shared sensors).
XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_barrel_v1.xml")

RADIUS_MIN = 0.2
RADIUS_MAX = 0.6

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
ORIENTATION_TOLERANCE = 0.1

# At reset the barrel must be LYING DOWN: its cylinder axis (body z) is near-horizontal,
# i.e. the world-z component of the body z-axis is small.
LYING_TOLERANCE = 0.25


def _barrel_not_lying(quat: np.ndarray) -> bool:
    """Reject orientation (return True) unless the barrel axis is near-horizontal (lying flat)."""
    object_z_axis = quat_to_mat(quat) @ Z_AXIS
    return bool(abs(object_z_axis[2]) > LYING_TOLERANCE)


@dataclass
class SpotBarrelUprightConfig(SpotUprightConfig):
    """Config for the Spot barrel upright task."""


class SpotBarrelUpright(SpotBase[SpotBarrelUprightConfig]):
    """Task getting Spot to upright a barrel that starts lying on the ground."""

    name: str = "spot_barrel_upright"
    config_t: type[SpotBarrelUprightConfig] = SpotBarrelUprightConfig  # type: ignore[assignment]
    config: SpotBarrelUprightConfig

    def __init__(self, config: SpotBarrelUprightConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("barrel_joint")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")
        self.end_effector_to_object_idx = self.get_sensor_start_index("sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward using object uprightness and gripper proximity.

        Inputs (all batched over candidate rollouts):
            states:   (batch, horizon, nq + nv) -- concatenated [qpos | qvel] per timestep.
            sensors:  (batch, horizon, nsensordata) -- all <sensor> outputs per timestep.
            controls: (batch, horizon, nu) -- the commanded controls per timestep.
        Each term is reduced to (batch,) -- one score per rollout. Errors/distances are
        negated inside the shared helpers because reward is "higher is better".
        """
        batch_size = states.shape[0]

        # object_z_axis: the barrel's "up" axis (cylinder axis) in world frame, (batch, horizon, 3).
        # Equals world Z when the barrel is upright; horizontal while it lies on its side. Drives uprighting.
        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        # gripper_to_object: vector from the barrel (site_object) to the gripper finger, (batch, horizon, 3).
        gripper_to_object = sensors[..., self.end_effector_to_object_idx : self.end_effector_to_object_idx + 3]
        # gripper_distance: gripper->barrel distance, (batch, horizon). Norm of the vector above.
        gripper_distance = np.linalg.norm(gripper_to_object, axis=-1)

        # orientation_reward: (batch,). Negated mean (1 - object_z·world_z) -> maximized when barrel stands up.
        orientation_reward = z_axis_orientation_reward(self.config, object_z_axis)
        # proximity_reward: (batch,). Negated mean gripper->barrel distance -> keep the arm on the barrel.
        proximity_reward = gripper_distance_reward(self.config, gripper_distance)

        assert orientation_reward.shape == (batch_size,)
        assert proximity_reward.shape == (batch_size,)
        # Total reward = sum of the (already weighted) terms, shape (batch,).
        return (
            orientation_reward  # objective: stand the barrel upright (z-axis -> world z)
            + proximity_reward  # shaping: keep the gripper near the barrel so it can act on it
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset with the barrel lying flat on the ground at a random position/yaw."""
        reset_object_pose = random_object_pose(
            self.model,
            "barrel",
            sample_annulus_xy(RADIUS_MIN, RADIUS_MAX),
            reject_orientation=_barrel_not_lying,
        )
        spot_pos = DEFAULT_SPOT_POS + np.random.randn(2) * 0.001
        return np.array(
            [
                *spot_pos,
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *reset_object_pose,
            ]
        )

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the barrel is upright (its z-axis aligned with world z)."""
        object_z_axis = data.sensordata[self.object_z_axis_idx : self.object_z_axis_idx + 3]
        orientation_alignment = np.dot(object_z_axis, Z_AXIS)
        return bool(orientation_alignment >= (1.0 - ORIENTATION_TOLERANCE))


__all__ = ["SpotBarrelUpright", "SpotBarrelUprightConfig"]
