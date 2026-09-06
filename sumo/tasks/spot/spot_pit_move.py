# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Spot pit-move task.

Walk to the water pitcher, grasp its handle, lift it, carry it
to the goal position, and set it down upright.

Planner/plant split (see spot_pit_base.py): ``SpotPitMove`` is the planner task
(no water balls; optional rigid-water mass augmentation). ``SpotPitMovePlant`` is the
same scene with N water balls appended -- the "real" simulation the episode runs on.
"""

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
from sumo.tasks.spot.spot_pit_base import (
    PIT_REST_Z,
    append_water_balls,
    apply_rigid_water_mass,
    build_handle,
    calibrate_gripper_actuator,
    chunky_ball_count,
    fill_ratio_to_ball_count,
    scale_pitcher,
    settled_ball_poses,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_pit.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])

USE_GRIPPER = True

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
DEFAULT_OBJECT_POS = np.array([1.5, 0.0])


@dataclass
class SpotPitMoveConfig(SpotBaseConfig):
    """Config for the Spot pit-move task (grasp handle, lift, carry to goal, set down)."""

    goal_position: np.ndarray = np_1d_field(
        np.array([-1.0, 1.8, PIT_REST_Z], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, -1.0],
        maxs=[5.0, 5.0, 2.0],
        steps=[0.1, 0.1, 0.1],
        vis_name="pit_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[-1.0, 1.8, 0.0],
    )
    # Bookkeeping only: the water model is fixed at task construction time
    # (fill_ratio/planner_water_model constructor args), not via this config.
    fill_ratio: float = 0.5
    w_fence: float = 1000.0
    w_goal: float = 500.0
    w_orientation: float = 100.0  # keep the pitcher upright (spill proxy)
    # Strong penalty on tilt BEYOND the hook's natural free-hang angle (~26 deg): the
    # excess is what actually pours water out. Threshold sits above the hang angle so
    # the reward does not fight the hook-carry itself.
    w_tilt_penalty: float = 800.0
    tilt_penalty_threshold_deg: float = 40.0
    w_object_fallen: float = 2500.0
    object_tipped_cos: float = 0.5  # z-axis alignment below this counts as tipped over
    w_approach_site_proximity: float = 100.0
    w_gripper_to_grasp_proximity: float = 250.0
    w_gripper_orientation: float = 100.0
    w_carry: float = 300.0
    carry_height: float = 0.25  # target pitcher-origin height while carrying
    hook_gate_dist: float = 0.05  # finger this close to the hook point counts as hooked
    place_radius: float = 0.4  # within this xy distance of the goal, lower to rest height
    w_object_velocity: float = 5.0
    w_controls: float = 2.0
    # Resistance-based grasp detection (same convention as spot_bucket_drag).
    resistance_threshold: float = 0.15
    fully_closed_tolerance: float = 0.05
    w_false_grasp_penalty: float = 1000.0
    w_grasp_quality: float = 100.0
    position_tolerance: float = 0.15
    upright_tolerance_cos: float = 0.9
    resting_height: float = 0.06
    resting_speed: float = 0.1


class SpotPitMove(SpotBase[SpotPitMoveConfig]):
    """Planner task: grasp the pitcher handle, lift, carry to the goal, set down."""

    name = "spot_pit_move"
    config_t = SpotPitMoveConfig
    config: SpotPitMoveConfig

    def __init__(
        self,
        config: SpotPitMoveConfig | None = None,
        fill_ratio: float = 0.5,
        planner_water_model: str = "empty",
        chunky_radius: float = 0.025,
        handle_style: str = "plate",
        pit_scale: float = 1.0,
    ) -> None:
        self._handle_style = handle_style
        if planner_water_model not in ("empty", "rigid_mass", "chunky"):
            raise ValueError(
                f"planner_water_model must be 'empty', 'rigid_mass' or 'chunky', got {planner_water_model!r}"
            )
        if not 0.2 <= pit_scale <= 1.0:
            raise ValueError(f"pit_scale must be in [0.2, 1.0], got {pit_scale}")
        self._pit_scale = pit_scale
        self._fill_ratio = fill_ratio
        self._planner_water_model = planner_water_model
        self._n_balls = fill_ratio_to_ball_count(fill_ratio, pit_scale)
        self._chunky_radius = chunky_radius
        # "chunky": the PLANNER model itself carries a few large balls of equal total
        # water mass, so rollouts can predict slosh/spill at affordable cost.
        self._n_chunky = (
            chunky_ball_count(fill_ratio, chunky_radius, pit_scale) if planner_water_model == "chunky" else 0
        )
        super().__init__(model_path=XML_PATH, use_arm=True, use_gripper=USE_GRIPPER, config=config)
        self.config.fill_ratio = fill_ratio

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("pit_joint")
        self.object_vel_idx = self.get_joint_velocity_start_index("pit_joint")
        self.object_x_axis_idx = self.get_sensor_start_index("object_x_axis")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")

        self.gripper_to_grasp_idx = self.get_sensor_start_index("sensor_gripper_to_grasp_handle")
        self.gripper_to_hook_idx = self.get_sensor_start_index("sensor_gripper_to_hook")
        self.gripper_x_axis_idx = self.get_sensor_start_index("sensor_gripper_x_axis")
        self.gripper_z_axis_idx = self.get_sensor_start_index("sensor_gripper_z_axis")

        self.torso_to_approach_left_idx = self.get_sensor_start_index("sensor_torso_to_approach_left")
        self.torso_to_approach_mid_idx = self.get_sensor_start_index("sensor_torso_to_approach_mid")
        self.torso_to_approach_right_idx = self.get_sensor_start_index("sensor_torso_to_approach_right")

        self.gripper_joint_idx = self.get_joint_position_start_index("arm_f1x")

    def _process_spec(self) -> None:
        super()._process_spec()
        calibrate_gripper_actuator(self.spec)
        scale_pitcher(self.spec, self._pit_scale)
        build_handle(self.spec, self._handle_style, scale=self._pit_scale)
        if self._planner_water_model == "rigid_mass":
            apply_rigid_water_mass(self.spec, self._n_balls, scale=self._pit_scale)
        elif self._planner_water_model == "chunky":
            append_water_balls(self.spec, self._n_chunky, radius=self._chunky_radius)

    # ------------------------------------------------------------------
    # Shared reward building blocks (also used by the pour task subclassing this).
    # ------------------------------------------------------------------
    def _common_terms(self, states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> dict[str, np.ndarray]:
        """Per-(batch, horizon) intermediate quantities shared by move/pour rewards."""
        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_linvel = states[..., self.object_vel_idx : self.object_vel_idx + 3]
        object_angvel = states[..., self.object_vel_idx + 3 : self.object_vel_idx + 6]

        object_x_axis = sensors[..., self.object_x_axis_idx : self.object_x_axis_idx + 3]
        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        gripper_to_grasp = sensors[..., self.gripper_to_grasp_idx : self.gripper_to_grasp_idx + 3]
        gripper_to_hook = sensors[..., self.gripper_to_hook_idx : self.gripper_to_hook_idx + 3]
        gripper_x_axis = sensors[..., self.gripper_x_axis_idx : self.gripper_x_axis_idx + 3]
        gripper_z_axis = sensors[..., self.gripper_z_axis_idx : self.gripper_z_axis_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]

        # z-axis alignment of the pitcher with world up, in [-1, 1].
        z_alignment = object_z_axis[..., 2]

        # Resistance-based grasp detection. Gripper convention (verified empirically via
        # tools/pit_grasp_probe.py): open = -1.54, closed = 0.0. When the handle bar
        # blocks the fingers, qpos stays BELOW the close command (e.g. -0.2 vs 0.0), so
        # resistance shows as a NEGATIVE position error.
        gripper_joint_pos = qpos[..., self.gripper_joint_idx]
        gripper_joint_cmd = controls[..., 9]
        position_error = gripper_joint_pos - gripper_joint_cmd
        has_resistance = position_error < -self.config.resistance_threshold
        is_fully_closed = np.abs(gripper_joint_pos - GRIPPER_CLOSED_POS) < self.config.fully_closed_tolerance
        is_grasping = has_resistance & ~is_fully_closed
        # Commanding (near-)closed while the fingers reach fully closed => grabbed air.
        is_false_grasp = is_fully_closed & (gripper_joint_cmd > -0.5)

        # Torso -> approach-site distances.
        approach = np.stack(
            [
                np.linalg.norm(sensors[..., idx : idx + 3], axis=-1)
                for idx in (
                    self.torso_to_approach_left_idx,
                    self.torso_to_approach_mid_idx,
                    self.torso_to_approach_right_idx,
                )
            ],
            axis=-1,
        )

        return {
            "body_height": body_height,
            "body_pos": body_pos,
            "object_pos": object_pos,
            "object_linvel": object_linvel,
            "object_angvel": object_angvel,
            "object_x_axis": object_x_axis,
            "z_alignment": z_alignment,
            "gripper_dist": np.linalg.norm(gripper_to_grasp, axis=-1),
            "hook_dist": np.linalg.norm(gripper_to_hook, axis=-1),
            "gripper_x_axis": gripper_x_axis,
            "object_z_axis": object_z_axis,
            "gripper_z_axis": gripper_z_axis,
            "object_y_axis": object_y_axis,
            "position_error": position_error,
            "is_grasping": is_grasping,
            "is_false_grasp": is_false_grasp,
            "min_approach_dist": approach.min(axis=-1),
        }

    def _constraint_rewards(self, t: dict[str, np.ndarray]) -> np.ndarray:
        """(batch,) fence + robot-fallen + pitcher-tipped penalties."""
        fence_x = (t["body_pos"][..., 0] < HARDWARE_FENCE_X[0]) | (t["body_pos"][..., 0] > HARDWARE_FENCE_X[1])
        fence_y = (t["body_pos"][..., 1] < HARDWARE_FENCE_Y[0]) | (t["body_pos"][..., 1] > HARDWARE_FENCE_Y[1])
        fence = -self.config.w_fence * (fence_x | fence_y).any(axis=-1)
        fallen = -self.config.fall_penalty * (t["body_height"] <= self.config.spot_fallen_threshold).any(axis=-1)
        tipped = -self.config.w_object_fallen * (t["z_alignment"] < self.config.object_tipped_cos).any(axis=-1)
        return fence + fallen + tipped

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward: thread a finger through the handle loop (HOOK), lift, carry, set down.

        The hook strategy is probe-validated (tools/pit_grasp_probe.py --grasp hook):
        an open finger through the loop hangs the pitcher at ~26 deg with ~0% spill,
        whereas any pinch-lift swings it to 72-109 deg and dumps the water. Pushing the
        pitcher along the ground remains a legal alternative strategy (no lift gate) --
        make it hard via the `pit_friction` knob in run_mismatch configs.
        """
        batch_size = states.shape[0]
        t = self._common_terms(states, sensors, controls)

        goal = np.asarray(self.config.goal_position)
        goal_dist_xy = np.linalg.norm(t["object_pos"][..., :2] - goal[None, None, :2], axis=-1)
        goal_reward = -self.config.w_goal * goal_dist_xy.mean(-1)

        # Reach: bring the finger (fngr site) to the loop-opening hook point.
        approach = -self.config.w_approach_site_proximity * t["min_approach_dist"].mean(-1)
        reach = -self.config.w_gripper_to_grasp_proximity * t["hook_dist"].mean(-1)
        # Thread the loop with the gripper level (z-axis horizontal).
        grip_orient = -self.config.w_gripper_orientation * np.abs(t["gripper_z_axis"][..., 2]).mean(-1)

        # Carry: once the finger is inside the loop (proximity gate -- hooking needs no
        # jaw closing), pull the pitcher toward carry height; near the goal the target
        # height drops to the rest height (set-down phase).
        hooked = t["hook_dist"] < self.config.hook_gate_dist
        target_z = np.where(goal_dist_xy < self.config.place_radius, PIT_REST_Z, self.config.carry_height)
        lift_error = np.abs(t["object_pos"][..., 2] - target_z)
        carry_reward = -self.config.w_carry * np.where(hooked, lift_error, self.config.carry_height).mean(-1)

        orientation_reward = -self.config.w_orientation * (1.0 - t["z_alignment"]).mean(-1)

        # Spill guard: heavily penalize tilt beyond the threshold (hook hang is ~26 deg;
        # beyond ~40 deg the free surface reaches the rim and water pours out).
        tilt = np.degrees(np.arccos(np.clip(t["z_alignment"], -1.0, 1.0)))
        excess_tilt = np.clip(tilt - self.config.tilt_penalty_threshold_deg, 0.0, None) / 90.0
        tilt_penalty = -self.config.w_tilt_penalty * excess_tilt.mean(-1)

        velocity_reward = -self.config.w_object_velocity * np.linalg.norm(t["object_linvel"], axis=-1).mean(-1)
        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        total = (
            self._constraint_rewards(t)
            + approach
            + reach
            + grip_orient
            + goal_reward
            + carry_reward
            + orientation_reward
            + tilt_penalty
            + velocity_reward
            + controls_reward
        )
        assert total.shape == (batch_size,)
        return total

    @property
    def reset_pose(self) -> np.ndarray:
        """Robot standing at DEFAULT_SPOT_POS; pitcher upright at DEFAULT_OBJECT_POS."""
        rest_z = PIT_REST_Z * self._pit_scale
        reset_object_pose = np.array([*DEFAULT_OBJECT_POS, rest_z, 1, 0, 0, 0])
        pose = np.array(
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
        if self._n_chunky > 0:
            pit_world = np.array([*DEFAULT_OBJECT_POS, rest_z])
            local = settled_ball_poses(self._n_chunky, radius=self._chunky_radius, scale=self._pit_scale)
            ball_qpos = np.concatenate([np.concatenate([pit_world + p, [1, 0, 0, 0]]) for p in local])
            pose = np.concatenate([pose, ball_qpos])
        return pose

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Pitcher at the goal (xy), upright, resting on the ground, and slow.

        Reads only the shared robot/pitcher state prefix, so it works on both the
        planner and plant models. Spill fraction is reported separately by the runner.
        """
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        goal = np.asarray(self.config.goal_position)
        if np.linalg.norm(object_pos[:2] - goal[:2]) > self.config.position_tolerance:
            return False
        pit_z_axis = data.xmat[model.body("pit").id].reshape(3, 3)[:, 2]
        if pit_z_axis[2] < self.config.upright_tolerance_cos:
            return False
        if object_pos[2] > self.config.resting_height:
            return False
        object_speed = np.linalg.norm(data.qvel[self.object_vel_idx : self.object_vel_idx + 3])
        return bool(object_speed <= self.config.resting_speed)


class SpotPitMovePlant(SpotPitMove):
    """Plant variant: the same scene with N water balls appended (the 'real' sim)."""

    name = "spot_pit_move_plant"

    def __init__(
        self,
        config: SpotPitMoveConfig | None = None,
        fill_ratio: float = 0.5,
        handle_style: str = "plate",
        pit_scale: float = 1.0,
    ) -> None:
        # The plant pitcher is always the true empty pitcher + real balls.
        super().__init__(
            config=config,
            fill_ratio=fill_ratio,
            planner_water_model="empty",
            handle_style=handle_style,
            pit_scale=pit_scale,
        )

    def _process_spec(self) -> None:
        super()._process_spec()
        append_water_balls(self.spec, self._n_balls)

    @property
    def reset_pose(self) -> np.ndarray:
        base = super().reset_pose
        if self._n_balls == 0:
            return base
        pit_world = np.array([*DEFAULT_OBJECT_POS, PIT_REST_Z * self._pit_scale])
        local = settled_ball_poses(self._n_balls, scale=self._pit_scale)
        ball_qpos = np.concatenate([np.concatenate([pit_world + p, [1, 0, 0, 0]]) for p in local])
        return np.concatenate([base, ball_qpos])
