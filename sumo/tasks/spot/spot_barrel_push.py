# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from sumo.tasks.spot.spot_push import (
    SpotPushConfig,
    goal_distance_reward,
    gripper_distance_reward,
    object_linear_velocity_reward,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_barrel_v1.xml")

# Object position is sampled in an annulus around the origin on reset.
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0

# Resting height of the barrel center.
# Upright barrel: half its height (0.45). To make it lie on its side and ROLL,
# set this to the barrel radius (0.29) and use an on-side quat in reset_pose (see note below).
DEFAULT_BARREL_HEIGHT = 0.45

# Success condition tolerances.
POSITION_TOLERANCE = 0.2
VELOCITY_TOLERANCE = 0.05
SPOT_FALLEN_THRESHOLD = 0.35


@dataclass
class SpotBarrelPushConfig(SpotPushConfig):
    """Config for the Spot barrel pushing/moving task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, DEFAULT_BARREL_HEIGHT], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.0],
        steps=[0.1, 0.1, 0.05],
        vis_name="barrel_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, DEFAULT_BARREL_HEIGHT],
    )


class SpotBarrelPush(SpotBase[SpotBarrelPushConfig]):
    """Task getting Spot to push/move a plain barrel to a goal location.

    Uses the arm as a pushing stick (no gripper). Reward mirrors spot_cone_push:
    drive the object toward the goal, keep the end-effector near the object, and
    penalize the object's linear velocity (discourage knocking it away).
    """

    name = "spot_barrel_push"
    config_t: type[SpotBarrelPushConfig] = SpotBarrelPushConfig
    config: SpotBarrelPushConfig

    # Joints whose state comes from perception rather than from simulation, i.e. what a
    # real-robot deployment has to supply every tick. Only the task knows this -- the
    # model cannot say which of its free joints we can actually see. Consumed by
    # sumo_server.scene.SceneLayout.from_task.
    perceived_object_joints: tuple[str, ...] = ("barrel_joint",)

    def __init__(self, config: SpotBarrelPushConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)

        self.body_pose_start = self.get_joint_position_start_index("base")
        self.object_pose_start = self.get_joint_position_start_index("barrel_joint")
        self.object_vel_start = self.get_joint_velocity_start_index("barrel_joint")
        self.end_effector_to_object_start = self.get_sensor_start_index("sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward using goal distance, end-effector proximity, and object velocity penalty.

        Inputs (all batched over candidate rollouts):
            states:   (batch, horizon, nq + nv) -- concatenated [qpos | qvel] per timestep.
            sensors:  (batch, horizon, nsensordata) -- all <sensor> outputs per timestep.
            controls: (batch, horizon, nu) -- the commanded controls per timestep.
        Each term below is reduced to (batch,) -- one score per rollout. Distances are
        negated inside the shared helpers because reward is "higher is better".
        """
        batch_size = states.shape[0]

        # qpos: the position half of the state, shape (batch, horizon, nq).
        qpos = states[..., : self.model.nq]
        # object_pos: barrel free-joint [x, y, z], shape (batch, horizon, 3). Drives the goal term.
        object_pos = qpos[..., self.object_pose_start : self.object_pose_start + 3]
        # object_linear_velocity: barrel [vx, vy, vz], (batch, horizon, 3). Read from the qvel half of
        # `states` (object_vel_start already points past nq). Penalized so the barrel isn't knocked away.
        object_linear_velocity = states[..., self.object_vel_start : self.object_vel_start + 3]

        # end_effector_to_object: vector from the barrel (site_object) to the gripper finger, (batch, horizon, 3).
        # Its norm is the gripper->barrel distance -- the "keep the hand on the barrel while pushing" signal.
        end_effector_to_object = sensors[..., self.end_effector_to_object_start : self.end_effector_to_object_start + 3]
        # gripper_proximity_reward: (batch,). Negated mean gripper->barrel distance (helper applies -w).
        gripper_proximity_reward = gripper_distance_reward(
            self.config,
            np.linalg.norm(end_effector_to_object, axis=-1),
        )
        # goal_reward: (batch,). Negated mean distance from the barrel to the goal xyz (the task objective).
        goal_reward = goal_distance_reward(self.config, object_pos)
        # object_linear_velocity_penalty: (batch,). Penalize barrel speed -> discourage shoving it away.
        object_linear_velocity_penalty = object_linear_velocity_reward(self.config, object_linear_velocity)

        assert gripper_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_penalty.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        # Total reward = sum of the (already weighted) terms, shape (batch,).
        return (
            goal_reward  # objective: barrel close to the goal
            + gripper_proximity_reward  # shaping: keep the hand near the barrel
            + object_linear_velocity_penalty  # shaping: push gently, don't knock it away
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and barrel.

        The barrel starts upright (quat [1, 0, 0, 0]). To roll it on its side instead,
        use quat [0.7071, 0.7071, 0, 0] and set DEFAULT_BARREL_HEIGHT to the radius (0.29).
        """
        # Sample object position in an annulus around the origin.
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + 0.1 * np.random.randn(2)
        object_pose = np.array([*object_pos, DEFAULT_BARREL_HEIGHT, 1, 0, 0, 0])

        # Place robot at a random pose near the origin.
        robot_pose_xy = np.random.uniform(-0.5, 0.5, 2)
        random_yaw_robot = np.random.uniform(0, 2 * np.pi)
        robot_pose_orientation = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])
        robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the barrel reached the goal and is nearly at rest."""
        object_pos = data.qpos[self.object_pose_start : self.object_pose_start + 3]
        object_vel = data.qvel[self.object_vel_start - self.model.nq : self.object_vel_start - self.model.nq + 3]
        goal_pos = np.array(self.config.goal_position)
        position_check = np.linalg.norm(object_pos - goal_pos, axis=-1, ord=np.inf) < POSITION_TOLERANCE
        velocity_check = np.linalg.norm(object_vel, axis=-1) < VELOCITY_TOLERANCE
        return position_check and velocity_check

    def failure(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if Spot has fallen."""
        body_height = data.qpos[self.body_pose_start + 2]
        return body_height <= SPOT_FALLEN_THRESHOLD
