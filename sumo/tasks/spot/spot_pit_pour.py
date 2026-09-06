# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Spot pit-pour task (v3).

Grasp the pitcher, LIFT it off the ground, and hold a
commanded POUR ATTITUDE in the air.

The pour attitude is defined by the SPOUT ANGLE: the angle between the spout
direction (the pitcher's local -x axis) and straight-down. 90 deg = upright
(spout horizontal), 0 deg = spout pointing straight down. Success requires, held
continuously for ``hold_time_s``:

  - gripper engaged at the selected grasp target (``grasp_mode``), AND
  - pitcher LIFTED (origin above ``lift_gate_z`` -- ground-assisted tipping does
    not count), AND
  - |spout angle - target_spout_angle_deg| <= spout_tolerance_deg.

Grasp modes (--grasp-mode): pinch the flat handle plate ("plate"), pinch the +y
wall top edge ("wall"), or insert the gripper into the mouth and OPEN it against
the inner walls ("inside", expansion grip -- the strongest static hold in the
scripted probes: ~1.5 cm slip, ~12 deg free-hang).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks.spot.spot_constants import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS
from mujoco import MjData, MjModel

from sumo.tasks.spot.spot_pit_base import append_water_balls, settled_ball_poses, spout_angle_deg
from sumo.tasks.spot.spot_pit_move import (
    DEFAULT_OBJECT_POS,
    PIT_REST_Z,
    SpotPitMove,
    SpotPitMoveConfig,
)

GRASP_MODE_SENSORS = {
    "plate": "sensor_gripper_to_grasp_handle",
    "wall": "sensor_gripper_to_wall",
    "inside": "sensor_gripper_to_inside",
}


@dataclass
class SpotPitPourConfig(SpotPitMoveConfig):
    """Config for the Spot pit-pour task (grasp, lift, hold a commanded pour attitude)."""

    target_spout_angle_deg: float = 45.0  # 90 = upright, 0 = spout straight down
    spout_tolerance_deg: float = 10.0
    hold_time_s: float = 1.0
    pour_height: float = 0.3  # target pitcher-origin height while pouring
    lift_gate_z: float = 0.15  # pitcher origin above this = airborne (no ground assist)
    engaged_dist: float = 0.08  # gripper within this distance of the grasp target = engaged
    w_wrong_way: float = 400.0  # spout tipping AWAY from the pour direction (always on)
    w_spout: float = 600.0
    w_object_angvel: float = 20.0  # smooth, controlled attitude
    w_grip_effort: float = 50.0  # shape the gripper command toward the mode's hold command
    # Method A (hold-agnostic): engage requires mech coupling + airborne + gripper near
    # the PITCHER BODY (any hold point); the straddle targets remain as guidance only.
    engage_any_hold: bool = False
    near_pit_dist: float = 0.35


class SpotPitPour(SpotPitMove):
    """Planner task: grasp the pitcher, lift it, hold the commanded spout angle in the air."""

    name = "spot_pit_pour"
    config_t = SpotPitPourConfig  # pyright: ignore[reportIncompatibleVariableOverride]
    config: SpotPitPourConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: SpotPitPourConfig | None = None,
        fill_ratio: float = 0.5,
        planner_water_model: str = "empty",
        chunky_radius: float = 0.025,
        grasp_mode: str = "plate",
        handle_style: str = "plate",
        pit_scale: float = 1.0,
    ) -> None:
        if grasp_mode not in GRASP_MODE_SENSORS:
            raise ValueError(f"grasp_mode must be one of {sorted(GRASP_MODE_SENSORS)}, got {grasp_mode!r}")
        self._grasp_mode = grasp_mode
        super().__init__(
            config=config,
            fill_ratio=fill_ratio,
            planner_water_model=planner_water_model,
            chunky_radius=chunky_radius,
            handle_style=handle_style,
            pit_scale=pit_scale,
        )
        self._hold_steps = 0
        self.pour_target_idx = self.get_sensor_start_index(GRASP_MODE_SENSORS[grasp_mode])
        # Straddle sensors: finger pad -> finger-side target, palm -> palm-side target.
        self.straddle_finger_idx = self.get_sensor_start_index(f"sensor_straddle_finger_{grasp_mode}")
        self.straddle_palm_idx = self.get_sensor_start_index(f"sensor_straddle_palm_{grasp_mode}")
        self.gripper_to_pit_idx = self.get_sensor_start_index("sensor_gripper_to_pit")
        # The hold command the reward shapes toward: pinches CLOSE, expansion OPENS.
        self._hold_grip_cmd = GRIPPER_OPEN_POS if grasp_mode == "inside" else GRIPPER_CLOSED_POS

    def _process_spec(self) -> None:
        """Add gripper jaw sites (finger pad + palm) and the straddle sensors.

        The judo arm model has no sites on the jaw bodies, so they are injected at the
        MjSpec level: the FINGER PAD on arm_link_fngr (between the two jaw capsules)
        and the PALM on arm_link_wr1 (top face of the bottom-jaw box). Sensors are
        appended at the END of the sensor table, so existing sensor addresses (used by
        the locomotion backend) are unaffected.
        """
        super()._process_spec()
        fngr_body = self.spec.body("arm_link_fngr")
        site = fngr_body.add_site()
        site.name = "site_jaw_finger"
        site.pos = [0.057, 0.0, -0.023]
        wr1_body = self.spec.body("arm_link_wr1")
        site = wr1_body.add_site()
        site.name = "site_jaw_palm"
        site.pos = [0.17, 0.0, -0.038]

        import mujoco as _mj

        pair_targets = {
            "plate": ("site_plate_finger", "site_plate_palm"),
            "wall": ("site_wall_finger", "site_wall_palm"),
            "inside": ("site_inside_finger", "site_inside_palm"),
        }
        for mode, (finger_target, palm_target) in pair_targets.items():
            for tag, jaw_site, target in (
                ("finger", "site_jaw_finger", finger_target),
                ("palm", "site_jaw_palm", palm_target),
            ):
                sensor = self.spec.add_sensor()
                sensor.name = f"sensor_straddle_{tag}_{mode}"
                sensor.type = _mj.mjtSensor.mjSENS_FRAMEPOS
                sensor.objtype = _mj.mjtObj.mjOBJ_SITE
                sensor.objname = jaw_site
                sensor.reftype = _mj.mjtObj.mjOBJ_SITE
                sensor.refname = target

        # Gripper wrist relative to the pitcher center (method A: hold-agnostic engage).
        sensor = self.spec.add_sensor()
        sensor.name = "sensor_gripper_to_pit"
        sensor.type = _mj.mjtSensor.mjSENS_FRAMEPOS
        sensor.objtype = _mj.mjtObj.mjOBJ_SITE
        sensor.objname = "site_arm_link_wr1"
        sensor.reftype = _mj.mjtObj.mjOBJ_SITE
        sensor.refname = "site_object"

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward: reach the grasp target -> engage -> lift -> hold the spout angle, smoothly."""
        batch_size = states.shape[0]
        t = self._common_terms(states, sensors, controls)
        cfg = self.config

        # Reach = STRADDLE: the finger pad and the palm each have their own target on
        # OPPOSITE sides of the material. Both can only be satisfied with the jaw
        # straddling it, which implicitly forces the collision-free entry direction
        # (e.g. a down-bite over the rim instead of ramming the wall sideways).
        finger_vec = sensors[..., self.straddle_finger_idx : self.straddle_finger_idx + 3]
        palm_vec = sensors[..., self.straddle_palm_idx : self.straddle_palm_idx + 3]
        finger_dist = np.linalg.norm(finger_vec, axis=-1)
        palm_dist = np.linalg.norm(palm_vec, axis=-1)
        target_dist = np.maximum(finger_dist, palm_dist)
        approach = -cfg.w_approach_site_proximity * t["min_approach_dist"].mean(-1)
        reach = -cfg.w_gripper_to_grasp_proximity * (0.5 * (finger_dist + palm_dist)).mean(-1)

        # PRE-GRASP POSE: per-mode gripper orientation so the material lands BETWEEN
        # the jaws (gripper x = jaw forward, gripper z = jaw opening direction):
        #   plate : bite down on the horizontal plate, jaw pointing inward (toward -x)
        #   wall  : vertical down-bite over the rim, jaw gap across the wall (obj y)
        #   inside: vertical insertion, expansion gap across the inner +-y walls
        g_x, g_z = t["gripper_x_axis"], t["gripper_z_axis"]
        if self._grasp_mode == "plate":
            # Vertical grip segment: jaw opens HORIZONTALLY (closing axis g_z parallel
            # to the pitcher's y axis), gripper pointing inward along -x.
            align_err = (1.0 - np.abs((g_z * t["object_y_axis"]).sum(-1))) + (
                1.0 + (g_x * t["object_x_axis"]).sum(-1)
            )
        elif self._grasp_mode == "wall":
            align_err = (1.0 - np.abs((g_z * t["object_y_axis"]).sum(-1))) + (1.0 + g_x[..., 2])
        else:  # inside
            align_err = (1.0 - np.abs((g_z * t["object_y_axis"]).sum(-1))) + (1.0 + g_x[..., 2])
        grasp_pose = -cfg.w_gripper_orientation * align_err.mean(-1)

        # Engaged = near the target AND mechanically coupled: pinch modes must have
        # the jaw BLOCKED while closing (holding material); the inside-expansion mode
        # must have the jaw blocked while OPENING (jammed against the inner walls).
        if cfg.engage_any_hold:
            # Method A: any mechanically-coupled hold near the pitcher body counts.
            pit_vec = sensors[..., self.gripper_to_pit_idx : self.gripper_to_pit_idx + 3]
            engaged_near = np.linalg.norm(pit_vec, axis=-1) < cfg.near_pit_dist
        else:
            engaged_near = target_dist < cfg.engaged_dist
        if self._grasp_mode == "inside":
            mech = t["position_error"] > 0.5
        else:
            mech = t["is_grasping"]
        engaged = engaged_near & mech
        lifted = t["object_pos"][..., 2] > cfg.lift_gate_z
        pouring = engaged & lifted

        # Engage effort: once near, shape the gripper command toward the mode's hold
        # command (close for pinches, OPEN for the inside expansion grip).
        grip_cmd = controls[..., 9]
        grip_effort = -cfg.w_grip_effort * (engaged_near * np.abs(grip_cmd - self._hold_grip_cmd)).mean(-1)

        # Lift: while engaged, pull the pitcher toward pour height (constant otherwise).
        lift_error = np.abs(t["object_pos"][..., 2] - cfg.pour_height)
        carry_reward = -cfg.w_carry * np.where(engaged, lift_error, cfg.pour_height).mean(-1)

        # Pour attitude: spout angle = angle(spout dir, straight-down); cos = +x axis z.
        spout = np.degrees(np.arccos(np.clip(t["object_x_axis"][..., 2], -1.0, 1.0)))
        target = cfg.target_spout_angle_deg
        spout_err_scale = np.abs(spout - target) / 90.0
        spout_reward = -cfg.w_spout * np.where(pouring, spout_err_scale, target / 90.0).mean(-1)

        # Keep the pitcher upright while NOT in the pour phase (no knocking it over,
        # no ground-tipping shortcuts -- tilt only pays when airborne and engaged).
        orientation_reward = -cfg.w_orientation * (~pouring * (1.0 - t["z_alignment"])).mean(-1)

        # ALWAYS-ON: punish tipping the spout the WRONG way (spout angle > 100 deg means
        # the spout points upward -- knocked backward / pawed over, never a pour).
        wrong_way = -cfg.w_wrong_way * np.clip((spout - 100.0) / 90.0, 0.0, None).mean(-1)

        angvel_reward = -cfg.w_object_angvel * (pouring * np.linalg.norm(t["object_angvel"], axis=-1)).mean(-1)
        velocity_reward = -cfg.w_object_velocity * np.linalg.norm(t["object_linvel"], axis=-1).mean(-1)
        controls_reward = -cfg.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        fence_x = (t["body_pos"][..., 0] < -2.0) | (t["body_pos"][..., 0] > 3.0)
        fence_y = (t["body_pos"][..., 1] < -3.0) | (t["body_pos"][..., 1] > 2.5)
        fence = -cfg.w_fence * (fence_x | fence_y).any(axis=-1)
        fallen = -cfg.fall_penalty * (t["body_height"] <= cfg.spot_fallen_threshold).any(axis=-1)

        total = (
            fence
            + fallen
            + approach
            + reach
            + grasp_pose
            + grip_effort
            + carry_reward
            + spout_reward
            + orientation_reward
            + wrong_way
            + angvel_reward
            + velocity_reward
            + controls_reward
        )
        assert total.shape == (batch_size,)
        return total

    def reset(self) -> None:
        super().reset()
        self._hold_steps = 0

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Engaged + airborne + spout angle in band, held for hold_time_s (continuous).

        Stateful: relies on being called once per sim step by the episode loop.
        """
        cfg = self.config
        if cfg.engage_any_hold:
            pit_vec = data.sensordata[self.gripper_to_pit_idx : self.gripper_to_pit_idx + 3]
            near = float(np.linalg.norm(pit_vec)) < cfg.near_pit_dist
        else:
            finger_vec = data.sensordata[self.straddle_finger_idx : self.straddle_finger_idx + 3]
            palm_vec = data.sensordata[self.straddle_palm_idx : self.straddle_palm_idx + 3]
            near = max(float(np.linalg.norm(finger_vec)), float(np.linalg.norm(palm_vec))) < cfg.engaged_dist
        # Mechanical coupling heuristic from the gripper joint alone: a pinch holding
        # material sits partially open; the inside jam cannot open (stays near closed).
        grip_q = float(data.qpos[self.gripper_joint_idx])
        mech = (grip_q > -0.5) if self._grasp_mode == "inside" else (-1.2 < grip_q < -0.06)
        engaged = near and mech
        lifted = data.qpos[self.object_pose_idx + 2] > cfg.lift_gate_z
        spout = spout_angle_deg(model, data)
        in_hold = engaged and lifted and (abs(spout - cfg.target_spout_angle_deg) <= cfg.spout_tolerance_deg)
        self._hold_steps = self._hold_steps + 1 if in_hold else 0
        if self._hold_steps in (1, 25, 50):  # instrument the hold window
            z = float(data.qpos[self.object_pose_idx + 2])
            print(
                f"    [hold {self._hold_steps}] t={data.time:.2f} z={z:.3f} spout={spout:.1f} "
                f"near={near} mech={mech} grip_q={float(data.qpos[self.gripper_joint_idx]):.2f}"
            )
        # One episode-loop step advances task.dt (a 50 Hz policy tick).
        return self._hold_steps * self.dt >= cfg.hold_time_s


class SpotPitPourPlant(SpotPitPour):
    """Plant variant: the same scene with N water balls appended (the 'real' sim)."""

    name = "spot_pit_pour_plant"

    def __init__(
        self,
        config: SpotPitPourConfig | None = None,
        fill_ratio: float = 0.5,
        grasp_mode: str = "plate",
        handle_style: str = "plate",
        pit_scale: float = 1.0,
    ) -> None:
        super().__init__(
            config=config,
            fill_ratio=fill_ratio,
            planner_water_model="empty",
            grasp_mode=grasp_mode,
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
