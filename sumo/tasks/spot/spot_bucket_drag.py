# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import (
    GRIPPER_CLOSED_POS,
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_bucket.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
Y_AXIS = np.array([1.0, 0.0, 0.0])
X_AXIS = np.array([0.0, -1.0, 0.0])  # flipped for hardware

# annulus object position sampling
RADIUS_MIN = 1.5
RADIUS_MAX = 3.0
USE_LEGS = False
USE_GRIPPER = True
USE_TORSO = False

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
DEFAULT_OBJECT_POS = np.array([2.0, 0.0])

# Bucket rests on the ground with its origin at z = 0.22 (bottom of the body cylinder touches z=0).
DEFAULT_OBJECT_HEIGHT = 0.22


@dataclass
class SpotBucketDragConfig(SpotBaseConfig):
    """Config for the Spot bucket dragging task (grab the handle, drag to the goal)."""

    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, -1.0],
        maxs=[5.0, 5.0, 2.0],
        steps=[0.1, 0.1, 0.1],
        vis_name="bucket_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[-1.0, 1.3, 0.0],
    )
    w_fence: float = 1000.0
    w_goal: float = 500.0
    w_orientation: float = 50.0  # Reward for keeping bucket aligned with global x, y, z axes
    object_fallen_threshold: float = 0.1  # Bucket center-of-mass height (tipped/knocked over)
    w_torso_proximity: float = 50.0
    torso_proximity_threshold: float = 1.0
    w_object_fallen: float = 2500.0
    w_gripper_to_grasp_proximity: float = 250.0
    w_gripper_orientation: float = 250.0
    w_approach_site_proximity: float = 100.0
    w_controls: float = 2.0
    w_object_velocity: float = 10.0
    # Resistance-based grasp detection parameters
    resistance_threshold: float = 0.15
    fully_closed_tolerance: float = 0.05
    w_false_grasp_penalty: float = 1000.0
    w_grasp_quality: float = 100.0
    position_tolerance: float = 0.2


class SpotBucketDrag(SpotBase[SpotBucketDragConfig]):
    """Task getting Spot to grab a bucket by its handle and drag it to a goal location."""

    name = "spot_bucket_drag"
    config_t = SpotBucketDragConfig
    config: SpotBucketDragConfig

    def __init__(self, config: SpotBucketDragConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, use_gripper=True, config=config)

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("bucket_joint")
        self.object_vel_idx = self.get_joint_velocity_start_index("bucket_joint")
        self.object_x_axis_idx = self.get_sensor_start_index("object_x_axis")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")

        # Single grasp point: the handle top bar.
        self.gripper_to_grasp_idx = self.get_sensor_start_index("sensor_gripper_to_grasp_handle")

        # Gripper orientation sensors
        self.gripper_x_axis_idx = self.get_sensor_start_index("sensor_gripper_x_axis")
        self.gripper_y_axis_idx = self.get_sensor_start_index("sensor_gripper_y_axis")
        self.gripper_z_axis_idx = self.get_sensor_start_index("sensor_gripper_z_axis")

        # Torso to approach site proximity sensors (left, middle, right)
        self.torso_to_approach_left_idx = self.get_sensor_start_index("sensor_torso_to_approach_left")
        self.torso_to_approach_mid_idx = self.get_sensor_start_index("sensor_torso_to_approach_mid")
        self.torso_to_approach_right_idx = self.get_sensor_start_index("sensor_torso_to_approach_right")

        # Gripper joint position index for the resistance-based grasp detection.
        self.gripper_joint_idx = self.get_joint_position_start_index("arm_f1x")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot bucket dragging task.

        Inputs (all batched over candidate rollouts):
            states:   (batch, horizon, nq + nv) -- concatenated [qpos | qvel] per timestep.
            sensors:  (batch, horizon, nsensordata) -- all <sensor> outputs per timestep.
            controls: (batch, horizon, nu) -- the commanded controls per timestep.
        Every intermediate below keeps the leading (batch, horizon) dims and is reduced
        to (batch,) at the end via .mean(-1) / .any(-1), so the final reward is one score
        per rollout. Distances/errors are negated because reward is "higher is better".
        """
        batch_size = states.shape[0]

        # qpos: the position half of the state, shape (batch, horizon, nq).
        qpos = states[..., : self.model.nq]

        # --- Robot base (Spot) pose, read from qpos ---
        # body_height: z of the robot base, shape (batch, horizon). Used for the "fell over" check.
        body_height = qpos[..., self.body_pose_idx + 2]
        # body_pos: robot base [x, y, z], shape (batch, horizon, 3). Used for fence + (unused) proximity.
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]

        # --- Bucket (object) state ---
        # object_pos: bucket free-joint [x, y, z], shape (batch, horizon, 3). Drives goal + fallen terms.
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        # object_linear_velocity: bucket linear velocity [vx, vy, vz], shape (batch, horizon, 3).
        # Read from the qvel half of `states` (object_vel_idx already points past nq). (Unused below.)
        object_linear_velocity = states[..., self.object_vel_idx : self.object_vel_idx + 3]

        # --- Bucket orientation: its three body axes expressed in world frame, each (batch, horizon, 3) ---
        # When the bucket is perfectly upright these equal the world X/Y/Z axes. Used by orientation reward.
        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]  # bucket "up" axis
        object_x_axis = sensors[..., self.object_x_axis_idx : self.object_x_axis_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]

        # gripper_to_grasp: vector from the handle grasp site to the gripper wrist site, (batch, horizon, 3).
        # Its norm is the gripper->handle distance (the core "reach the handle" signal).
        gripper_to_grasp = sensors[..., self.gripper_to_grasp_idx : self.gripper_to_grasp_idx + 3]

        # gripper_z_axis: the gripper wrist "up" axis in world frame, (batch, horizon, 3).
        # Compared against object_z_axis so the gripper approaches the handle with the right roll.
        gripper_z_axis = sensors[..., self.gripper_z_axis_idx : self.gripper_z_axis_idx + 3]

        # --- Torso -> approach-site vectors, each (batch, horizon, 3) ---
        # Vectors from the robot torso to the three "stand here" sites placed around the bucket;
        # their norms are how far the torso is from each candidate standing spot.
        torso_to_approach_left = sensors[..., self.torso_to_approach_left_idx : self.torso_to_approach_left_idx + 3]
        torso_to_approach_mid = sensors[..., self.torso_to_approach_mid_idx : self.torso_to_approach_mid_idx + 3]
        torso_to_approach_right = sensors[..., self.torso_to_approach_right_idx : self.torso_to_approach_right_idx + 3]

        # === Resistance-Based Grasp Detection ===
        # There is no direct "am I holding the handle" sensor, so we infer it from the gripper joint.
        # gripper_joint_pos: actual opening of the gripper finger joint (arm_f1x), (batch, horizon).
        gripper_joint_pos = qpos[..., self.gripper_joint_idx]
        # gripper_joint_cmd: the commanded opening for that same joint. ARM_CMD_INDS[6] -> controls index 9.
        gripper_joint_cmd = controls[..., 9]

        # position_error = actual - commanded. If we command "close" but the handle physically blocks the
        # fingers, the actual opening stays larger than commanded -> positive error -> the gripper is
        # meeting resistance -> it is holding something. (batch, horizon).
        position_error = gripper_joint_pos - gripper_joint_cmd

        # has_resistance: True where the fingers can't close as far as commanded (something is in the way).
        has_resistance = position_error > self.config.resistance_threshold
        # is_fully_closed: True where the fingers reached the fully-closed pose (i.e. grabbed nothing).
        is_fully_closed = np.abs(gripper_joint_pos - GRIPPER_CLOSED_POS) < self.config.fully_closed_tolerance
        # is_not_empty: the complement -- the hand is not closed all the way, so it may be around the handle.
        is_not_empty = ~is_fully_closed
        # is_grasping: resistance AND not fully closed -> a real grasp on the handle. (batch, horizon).
        is_grasping = has_resistance & is_not_empty
        # is_false_grasp: commanded a firm close (< -0.5) yet fingers shut completely -> closed on empty air.
        is_false_grasp = is_fully_closed & (gripper_joint_cmd < -0.5)

        # --- Fence penalty: keep the robot inside the hardware-safe rectangle ---
        # fence_violated_x/y: per-timestep bool, True where the base leaves the allowed x/y range.
        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        # spot_fence_reward: (batch,). A flat penalty if the base ever (.any over time) breaks the fence.
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        # spot_fallen_reward: (batch,). Large penalty if the base height ever drops below the fall threshold.
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # object_fallen_reward: (batch,). Penalty if the bucket ever drops below a height (knocked over / off).
        object_fallen_reward = -self.config.w_object_fallen * (
            object_pos[..., 2] <= self.config.object_fallen_threshold
        ).any(axis=-1)

        # --- Goal term: the actual task objective ---
        # goal_reward: (batch,). Negated mean (over time) distance from the bucket to the goal xyz.
        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        # --- Bucket orientation term: keep the bucket upright while dragging ---
        # *_alignment: dot of each bucket axis with its target world axis, in [-1, 1] (1 == aligned).
        x_alignment = np.sum(object_x_axis * X_AXIS, axis=-1)
        y_alignment = np.sum(object_y_axis * Y_AXIS, axis=-1)
        z_alignment = np.sum(object_z_axis * Z_AXIS, axis=-1)
        # orientation_error: sum of the three per-axis misalignments (0 == perfectly upright). (batch, horizon).
        orientation_error = (1.0 - x_alignment) + (1.0 - y_alignment) + (1.0 - z_alignment)
        # object_orientation_reward: (batch,). Negated mean misalignment -> penalize tipping/spinning.
        object_orientation_reward = -self.config.w_orientation * orientation_error.mean(axis=-1)

        # --- Approach term: encourage the torso to reach one of the standing spots beside the bucket ---
        # distance_to_*_approach: torso distance to each approach site, (batch, horizon).
        distance_to_left_approach = np.linalg.norm(torso_to_approach_left, axis=-1)
        distance_to_mid_approach = np.linalg.norm(torso_to_approach_mid, axis=-1)
        distance_to_right_approach = np.linalg.norm(torso_to_approach_right, axis=-1)
        # approach_distances: stack the three, (batch, horizon, 3); min_approach_distance: the nearest site.
        approach_distances = np.stack(
            [distance_to_left_approach, distance_to_mid_approach, distance_to_right_approach], axis=-1
        )
        min_approach_distance = np.min(approach_distances, axis=-1)
        # approach_site_proximity_reward: (batch,). Reward standing near the *closest* approach site.
        approach_site_proximity_reward = -self.config.w_approach_site_proximity * min_approach_distance.mean(-1)

        # --- Reach term: pull the gripper toward the handle grasp point ---
        # distance_to_grasp: gripper->handle distance, (batch, horizon).
        distance_to_grasp = np.linalg.norm(gripper_to_grasp, axis=-1)
        # gripper_to_grasp_proximity_reward: (batch,). The main "get the hand onto the handle" signal.
        gripper_to_grasp_proximity_reward = -self.config.w_gripper_to_grasp_proximity * distance_to_grasp.mean(-1)

        # --- Gripper orientation term: approach the handle with the gripper upright (z aligned with bucket z) ---
        # *_norm: unit-normalized axes (the +1e-8 avoids divide-by-zero). z_dot_product in [-1, 1].
        gripper_z_axis_norm = gripper_z_axis / (np.linalg.norm(gripper_z_axis, axis=-1, keepdims=True) + 1e-8)
        object_z_axis_norm = object_z_axis / (np.linalg.norm(object_z_axis, axis=-1, keepdims=True) + 1e-8)
        z_dot_product = np.sum(gripper_z_axis_norm * object_z_axis_norm, axis=-1)
        # z_alignment_error: 0 when the gripper z matches the bucket z. (batch, horizon).
        z_alignment_error = 1.0 - z_dot_product
        # gripper_orientation_reward: (batch,). Penalize a misaligned gripper roll.
        gripper_orientation_reward = -self.config.w_gripper_orientation * z_alignment_error.mean(axis=-1)

        # --- Grasp-based terms (use the resistance detection above) ---
        # false_grasp_penalty: (batch,). Penalty if the hand ever closed on empty air while commanded shut.
        false_grasp_penalty = -self.config.w_false_grasp_penalty * is_false_grasp.any(axis=-1)
        # grasp_quality: how hard the handle resists closing, squashed to [0, 1]. (batch, horizon).
        grasp_quality = np.clip(position_error / 0.5, 0, 1)
        # grasp_quality_reward: (batch,). Positive reward, only paid on timesteps that are actually grasping.
        grasp_quality_reward = self.config.w_grasp_quality * (is_grasping * grasp_quality).mean(axis=-1)

        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert object_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert object_orientation_reward.shape == (batch_size,)
        assert approach_site_proximity_reward.shape == (batch_size,)
        assert gripper_to_grasp_proximity_reward.shape == (batch_size,)
        assert gripper_orientation_reward.shape == (batch_size,)
        assert false_grasp_penalty.shape == (batch_size,)
        assert grasp_quality_reward.shape == (batch_size,)

        # Total reward = unweighted sum of the (already weighted) terms, shape (batch,).
        # Grouped by role: [constraints] keep robot/bucket safe and upright; [approach/reach/orient]
        # shape the contact; [grasp] reward holding the handle; [goal] is the task objective.
        return (
            spot_fence_reward  # constraint: stay inside the safe area
            + spot_fallen_reward  # constraint: don't let the robot fall
            + object_fallen_reward  # constraint: don't knock the bucket over/off
            + goal_reward  # objective: bucket close to the goal
            + object_orientation_reward  # shaping: keep the bucket upright
            + approach_site_proximity_reward  # shaping: stand next to the bucket
            + gripper_to_grasp_proximity_reward  # shaping: move the hand onto the handle
            + gripper_orientation_reward  # shaping: approach with the gripper upright
            + false_grasp_penalty  # grasp: punish closing on empty air
            + grasp_quality_reward  # grasp: reward actually holding the handle
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and bucket - bucket starts upright, resting on the ground."""
        reset_object_pose = np.array([*DEFAULT_OBJECT_POS, DEFAULT_OBJECT_HEIGHT, 1, 0, 0, 0])

        return np.array(
            [
                *DEFAULT_SPOT_POS,
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
        """Check if the bucket has been successfully dragged to the goal position."""
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        goal_position = np.array(self.config.goal_position)
        distance_to_goal = np.linalg.norm(object_pos - goal_position)
        return bool(distance_to_goal <= self.config.position_tolerance)
