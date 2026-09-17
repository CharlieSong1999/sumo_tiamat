# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.look_at import LookAtPointFields, point_look_term
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_barrel_perceive.xml")

# Upright rest height of the barrel body origin (mesh centered; half of 0.483 m).
BARREL_REST_HEIGHT = 0.2415


@dataclass
class SpotBarrelPerceiveConfig(LookAtPointFields, SpotBaseConfig):
    """Config for the barrel perception-replay task.

    Deliberately a plain navigate reward (walk toward a goal, stay standing). In the
    no-robot replay this task exists to (a) carry the barrel as a perceived free object
    so its measured pose can be injected into the scene and rendered, and (b) keep the
    planner producing sane plans. The robot's motion comes from the recording, so the
    reward is not doing manipulation -- do NOT copy a "kick" reward here.
    """

    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, STANDING_HEIGHT]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )
    # Flat-bottomed goal well: no goal gradient within goal_tolerance of goal_pos, so a
    # held position a few cm off (the goal anchors while the robot is still standing up;
    # odometry drifts) is not served with 0.1 m/s commands the real robot cannot execute
    # (2026-09-15: a 7 cm offset kept the legs shuffling). w_controls then makes zero the
    # optimum inside the well (judo's default is 0).
    goal_tolerance: float = 0.10
    w_controls: float = 10.0   # on (vx, vy) only; yaw is handled by the command deadbands


class SpotBarrelPerceive(SpotBase[SpotBarrelPerceiveConfig]):
    """Locomotion-only task that carries a perceived barrel, for the perception pipeline.

    Morphology matches the deployed spot_navigate (use_arm=False, base-velocity action
    space nu=3). The barrel is a free-joint object declared as perceived, so its pose is
    supplied every tick by the perception node (via objects_in) rather than simulated --
    the planner skips a tick whenever that pose is missing/stale, which is the intended
    behaviour. The barrel body frame equals the FoundationPose tracking-mesh frame
    (metric, z-up, centered; same mesh as objects/jug), so a perceived pose maps 1:1
    onto barrel_joint.
    """

    name = "spot_barrel_perceive"
    config_t: type[SpotBarrelPerceiveConfig] = SpotBarrelPerceiveConfig
    config: SpotBarrelPerceiveConfig

    # Supplied by perception every tick; consumed by sumo_server.scene.SceneLayout.from_task.
    perceived_object_joints: tuple[str, ...] = ("barrel_joint",)
    # The waiting task: no yaw floor, hold deadband instead (a look-at point set here still
    # steers, but only with commands the robot executes unaided). Use spot_navigate_look
    # for precise heading control.
    yaw_floor_enabled: bool = False
    # Action space: False = base velocity only (nu=3, the deployed spot_navigate morphology);
    # the `_arm` variants set True (nu=11, the arm tasks' family) and hold the arm by reward.
    use_arm: bool = False
    use_gripper: bool = False   # with the arm: the jug family's nu=11 = base 3 + arm 7 + gripper 1

    def __init__(self, config: SpotBarrelPerceiveConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=self.use_arm, use_gripper=self.use_gripper, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")

    def navigate_reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Plain navigate reward: torso toward goal, stay standing, cheap controls."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)
        goal_dist = np.linalg.norm(body_pos - np.asarray(self.config.goal_pos)[None, None], axis=-1)
        goal_reward = -self.config.w_goal * np.maximum(goal_dist - self.config.goal_tolerance, 0.0).mean(-1)
        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :2], axis=-1).mean(-1)
        if not self.yaw_floor_enabled and not self.config.look_at_enabled:
            controls_reward = controls_reward - self.config.w_yaw_hold * np.abs(controls[..., 2]).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        return spot_fallen_reward + goal_reward + controls_reward

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Navigate reward + the operator look-at point (off by default; yaw only)."""
        base = self.navigate_reward(states, sensors, controls, system_metadata)
        look_reward = point_look_term(states[..., : self.model.nq], self.body_pose_idx, self.config)
        assert look_reward.shape == base.shape
        return base + look_reward

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        return bool(super().success(model, data, metadata))

    def failure(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        body_height = data.qpos[self.body_pose_idx + 2]
        return bool(body_height <= self.config.spot_fallen_threshold)

    @property
    def reset_pose(self) -> np.ndarray:
        """Robot standing near origin; barrel upright a short distance ahead.

        In replay the barrel's pose is overwritten every tick by perception, so this
        placement only matters before the first measurement arrives.
        """
        object_pose = np.array([1.5, 0.0, BARREL_REST_HEIGHT, 1, 0, 0, 0])
        robot_pose = np.array([0.0, 0.0, STANDING_HEIGHT, 1, 0, 0, 0])
        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])
