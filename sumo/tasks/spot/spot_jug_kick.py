# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_jug_kick.xml")

# Object position is sampled in an annulus around the origin on reset.
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0

# Upright rest height of the jug body origin (mesh is centered; half of 0.483 m).
JUG_REST_HEIGHT = 0.2415


@dataclass
class SpotJugKickConfig(SpotBaseConfig):
    """Config for the Spot jug kick-over task.

    ``goal_pos`` keeps spot_navigate's name and semantics on purpose: the deployed
    click-to-go operator flow writes the clicked point into this field, and here the
    operator clicks the jug. Inherited from SpotBaseConfig: w_goal, w_controls,
    fall_penalty, spot_fallen_threshold.
    """

    w_tip: float = 150.0
    goal_distance_threshold: float = 0.5
    # Yaw-rate command floor (spot_base.yaw_command_floor): the same values as the rest of
    # the deployed nu=3 family, so a runtime switch into or out of kick keeps one mapping.
    yaw_rate_min: float = 0.4
    yaw_rate_deadzone: float = 0.1
    xy_speed_deadzone: float = 0.08   # see look_at.LookAtPointFields
    # Success when the jug's own z-axis has a world-z cosine below this (>60 deg over).
    tip_success_cos: float = 0.5
    goal_pos: np.ndarray = np_1d_field(
        np.array([2.0, 0.0, STANDING_HEIGHT]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


class SpotJugKick(SpotBase[SpotJugKickConfig]):
    """Task getting Spot to knock over a water jug by walking into it (no arm).

    Locomotion-only spot_navigate morphology (use_arm=False, base-velocity action
    space, nu=3) so the task deploys on the proven real-robot velocity stack. The
    operator points ``goal_pos`` at the jug; the reward drives the torso to the goal
    (spot_navigate's term) and pays ``w_tip`` for laying the jug's vertical axis
    toward the horizontal plane.

    The jug body frame equals the FoundationPose++ tracking-mesh frame (see
    objects/jug/jug.xml), so a perceived real-world jug pose maps 1:1 onto
    ``jug_joint`` via SceneLayout.
    """

    name = "spot_jug_kick"
    config_t: type[SpotJugKickConfig] = SpotJugKickConfig
    config: SpotJugKickConfig

    # Joints whose state comes from perception rather than from simulation, i.e. what a
    # real-robot deployment has to supply every tick. Consumed by
    # sumo_server.scene.SceneLayout.from_task.
    perceived_object_joints: tuple[str, ...] = ("jug_joint",)

    def __init__(self, config: SpotJugKickConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=False, config=config)

        self.body_pose_start = self.get_joint_position_start_index("base")
        self.object_pose_start = self.get_joint_position_start_index("jug_joint")
        # framezaxis of site_object (shared sensor.xml): the jug's z-axis in world frame.
        self.object_z_axis_start = self.get_sensor_start_index("object_z_axis")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward = walk to the goal (the jug) + tip the jug over - falls - control effort.

        Inputs (all batched over candidate rollouts):
            states:   (batch, horizon, nq + nv) -- concatenated [qpos | qvel] per timestep.
            sensors:  (batch, horizon, nsensordata) -- all <sensor> outputs per timestep.
            controls: (batch, horizon, nu) -- commanded base velocities per timestep.
        Each term is reduced to (batch,) -- one score per rollout.
        """
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_start + 2]
        body_pos = qpos[..., self.body_pose_start : self.body_pose_start + 3]

        # upright_cos: dot(jug z-axis, world Z) = the z-component of the unit z-axis
        # sensor, (batch, horizon). 1 = upright, 0 = lying flat, -1 = inverted.
        upright_cos = sensors[..., self.object_z_axis_start + 2]

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Same shape as spot_navigate's goal term: mean 3D torso->goal distance.
        goal_reward = -self.config.w_goal * np.linalg.norm(
            body_pos - np.asarray(self.config.goal_pos)[None, None], axis=-1
        ).mean(-1)

        # Tip term, SATURATED at horizontal: clip(1 - cos, 0, 1) is 0 while upright and
        # 1 once the jug is lying flat, and stays 1 past horizontal (inverted). The goal
        # is "vertical axis down to the horizontal plane" -- rolling further to inverted
        # adds no task value, so it must not out-score simply knocking the jug over
        # (otherwise the planner is paid ~w_tip more to keep rolling an already-tipped
        # jug, at the expense of the goal-distance term).
        tip_reward = self.config.w_tip * np.clip(1.0 - upright_cos, 0.0, 1.0).mean(-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert tip_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)
        return spot_fallen_reward + goal_reward + tip_reward + controls_reward

    def _jug_upright_cos(self, data: MjData) -> float:
        """dot(jug z-axis, world Z) computed from the free-joint quaternion."""
        quat = data.qpos[self.object_pose_start + 3 : self.object_pose_start + 7]
        w, x, y, z = quat / np.linalg.norm(quat)
        # R[2, 2] of the rotation matrix: world-z component of the body z-axis.
        return float(1.0 - 2.0 * (x * x + y * y))

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Success when the jug is knocked over and Spot is still standing."""
        tipped = self._jug_upright_cos(data) < self.config.tip_success_cos
        return bool(tipped and super().success(model, data, metadata))

    def failure(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if Spot has fallen."""
        body_height = data.qpos[self.body_pose_start + 2]
        return bool(body_height <= self.config.spot_fallen_threshold)

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose: jug upright AT the goal xy, robot an annulus away with random yaw.

        The jug spawns at ``config.goal_pos`` (with small jitter) rather than at an
        unrelated random point, so the navigate goal term always drives the torso
        toward the jug. This mirrors deployment, where the operator clicks the jug and
        that click IS ``goal_pos``; a headless/default reset that left the goal fixed
        while randomizing the jug would reward walking away from it (upright jugs give
        no tip reward, so the planner would just chase the stale goal).
        """
        goal_xy = np.asarray(self.config.goal_pos)[:2] + 0.1 * np.random.randn(2)
        object_pose = np.array([*goal_xy, JUG_REST_HEIGHT, 1, 0, 0, 0])

        # Robot starts a walk away (annulus around the jug) with random yaw, so it must
        # locomote to the jug to knock it over.
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        robot_pose_xy = goal_xy + np.array([radius * np.cos(theta), radius * np.sin(theta)])
        random_yaw_robot = np.random.uniform(0, 2 * np.pi)
        robot_pose_orientation = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])
        robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])
