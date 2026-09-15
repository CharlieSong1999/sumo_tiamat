"""Jug manipulation using the production tracking mesh."""

from dataclasses import dataclass

import mujoco
import numpy as np
from judo.utils.fields import np_1d_field

from sumo.tasks.spot import jug_water
from sumo.tasks.spot.look_at import object_look_term
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT
from sumo.tasks.spot.spot_jug_kick import XML_PATH
from sumo.tasks.spot.spot_upright import ground_clearance_height


@dataclass
class SpotJugManipulationConfig(SpotBaseConfig):
    # Construction-time model parameters; rebuild the task to change these.
    water_fill_ratio: float = 0.0
    water_ball_radius: float = 0.025
    # Total jug mass, kg. The tracking mesh's XML declares 1.0 kg (a near-empty 18.9 L
    # jug); the real jug used on 2026-09-14 carries a handful of small stones instead of
    # water, estimated at ~0.5 kg, hence 1.5. Mass and inertia are scaled together, so
    # the XML's inertia shape is kept. `--set jug_mass=1.0` reproduces the sim rounds.
    jug_mass: float = 1.5
    # Rolling-friction coefficient (m) of the jug against the floor, construction-time.
    # The XML's 0.0001 is an ideal cylinder: once nudged it rolls across the room, and
    # the MPC (which plans on this model) sees nothing wrong with that. A plastic jug
    # with loose stones inside stops within a metre or two; 0.01 is a conservative
    # stand-in. `--set rolling_friction=0.0001` reproduces the earlier sim rounds.
    rolling_friction: float = 0.01
    # Base-velocity cap (m/s) for the pushing modes (roll/move): a slower push keeps the
    # jug's speed within what the 2 s horizon can still stop, and is what a 1 m demo in a
    # small room can afford. upright/lay_down keep the base task's +-0.7.
    max_base_speed: float = 0.4
    # Yaw-only "face the jug" term (look_at.object_look_term), faded out as the jug
    # comes under the robot (w = w_look_object * (1 - exp(-(d/look_ramp_dist)^2))), so
    # the heading target cannot flip around when the jug is between the feet.
    w_look_object: float = 10.0
    look_ramp_dist: float = 0.6
    w_orientation: float = 150.0
    w_approach: float = 12.0
    w_position: float = 15.0
    w_linear_velocity: float = 8.0
    w_angular_velocity: float = 2.0
    w_controls: float = 0.3
    w_height: float = 100.0
    position_tolerance: float = 0.35
    angle_tolerance_deg: float = 15.0
    linear_speed_tolerance: float = 0.15
    angular_speed_tolerance: float = 0.6
    start_pos: np.ndarray = np_1d_field(np.array([0.95, 0.0, 0.0]))
    # Hand-selected approach target, not a detected keypoint or a physical constraint.
    gripper_target_local: np.ndarray = np_1d_field(np.array([0.0, 0.0, 0.17]))
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.95, 0.0, 0.2415]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


@dataclass
class SpotJugUprightConfig(SpotJugManipulationConfig):
    pass


@dataclass
class SpotJugLayDownConfig(SpotJugManipulationConfig):
    w_approach: float = 30.0
    w_position: float = 80.0
    w_linear_velocity: float = 25.0
    w_angular_velocity: float = 5.0
    # A forward tip pivots around the far base rim: center travels r + h/2 ~= .38 m.
    goal_pos: np.ndarray = np_1d_field(
        np.array([1.33, 0.0, 0.14]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


@dataclass
class SpotJugRollConfig(SpotJugManipulationConfig):
    w_position: float = 100.0
    w_approach: float = 15.0
    w_orientation: float = 40.0
    w_roll: float = 25.0
    w_slip: float = 4.0
    radius: float = 0.14
    min_roll_distance: float = 0.5
    goal_pos: np.ndarray = np_1d_field(
        np.array([2.15, 0.0, 0.14]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


@dataclass
class SpotJugMoveConfig(SpotJugManipulationConfig):
    """Move the jug 3 m, with no object orientation or velocity objective."""

    w_position: float = 100.0
    w_approach: float = 15.0
    w_orientation: float = 0.0
    w_height: float = 0.0
    w_linear_velocity: float = 0.0
    w_angular_velocity: float = 0.0
    goal_pos: np.ndarray = np_1d_field(
        np.array([3.95, 0.0, 0.14]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


def disable_velocity_rewards(config):
    """Ablate all jug velocity terms, including rolling/slip terms derived from velocity.

    Success thresholds and the base-command regularizer are unchanged.
    """
    for name in ("w_linear_velocity", "w_angular_velocity", "w_roll", "w_slip"):
        if hasattr(config, name):
            setattr(config, name, 0.0)


class SpotJugManipulation(SpotBase):
    config_t = SpotJugManipulationConfig
    mode = "upright"
    perceived_object_joints = ("jug_joint",)

    def __init__(self, config=None):
        water_config = config if config is not None else self.config_t()
        # judo's base builds the model (_process_spec) BEFORE it stores the passed config,
        # so every construction-time parameter is read from `water_config` here.
        self.jug_mass = float(water_config.jug_mass)
        self.rolling_friction = float(water_config.rolling_friction)
        if not (np.isfinite(self.jug_mass) and self.jug_mass > 0):
            raise ValueError(f"jug_mass must be finite and positive, got {water_config.jug_mass!r}")
        if not (np.isfinite(self.rolling_friction) and self.rolling_friction >= 0):
            raise ValueError(f"rolling_friction must be finite and >= 0, got {water_config.rolling_friction!r}")
        self.water_radius = water_config.water_ball_radius
        self.water_count, self.water_mass = jug_water.water_parameters(water_config.water_fill_ratio, self.water_radius)
        super().__init__(
            model_path=XML_PATH, use_arm=self.mode == "upright", use_gripper=self.mode == "upright", config=config
        )
        self.body_pose_start = self.get_joint_position_start_index("base")
        self.object_pose_start = self.get_joint_position_start_index("jug_joint")
        self.object_vel_start = self.get_joint_velocity_start_index("jug_joint")
        self.object_z_axis_start = self.get_sensor_start_index("object_z_axis")
        self.gripper_start = self.get_sensor_start_index("sensor_arm_link_fngr")

    def _process_spec(self):
        super()._process_spec()
        jug = self.spec.body("jug")
        if jug.mass <= 0:
            raise ValueError("jug body declares no mass; cannot scale it to jug_mass")
        ratio = self.jug_mass / float(jug.mass)
        jug.mass = self.jug_mass
        jug.inertia = np.asarray(jug.inertia, dtype=float) * ratio
        # The jug geom has priority 6 (> ground's 5), so its friction triple wins the
        # contact; rolling friction only acts with condim 6.
        geom = self.spec.geom("jug_collision")
        friction = np.asarray(geom.friction, dtype=float)
        friction[2] = self.rolling_friction
        geom.friction = friction
        geom.condim = 6
        if self.water_count:
            jug_water.configure_water_solver(self.spec)
            jug_water.add_inner_shell(self.spec, self.spec.body("jug"))
            jug_water.add_balls(self.spec, self.water_count, self.water_radius, self.water_mass)
            # Ground can catch an escaped ball; solid exterior hull never touches water.
            self.spec.geom("ground").conaffinity |= 2

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Base task's bounds, with vx/vy capped to +-max_base_speed in the pushing modes."""
        limits = np.array(super().actuator_ctrlrange, dtype=float, copy=True)
        if self.mode in ("roll", "move"):
            cap = float(self.config.max_base_speed)
            limits[0:2, 0] = np.maximum(limits[0:2, 0], -cap)
            limits[0:2, 1] = np.minimum(limits[0:2, 1], cap)
        return limits

    @property
    def reset_pose(self):
        if self.mode == "upright":
            # Fully horizontal, neck toward Spot; no initial momentum.
            quat = np.array([np.sqrt(0.5), 0, -np.sqrt(0.5), 0])
        elif self.mode == "roll":
            direction = np.asarray(self.config.goal_pos)[:2] - np.asarray(self.config.start_pos)[:2]
            yaw = np.arctan2(direction[1], direction[0])
            quat = np.array([np.cos(yaw / 2), np.cos(yaw / 2), np.sin(yaw / 2), np.sin(yaw / 2)]) / np.sqrt(2)
        else:
            quat = np.array([1.0, 0, 0, 0])
        height = ground_clearance_height(self.model, "jug", quat)
        xy = np.asarray(self.config.start_pos)[:2]
        base = np.array(
            [0, 0, STANDING_HEIGHT, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *xy, height, *quat]
        )
        if not self.water_count:
            return base
        local = jug_water.settled_positions(self.water_count, self.water_radius, self.water_mass, tuple(quat))
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quat)
        positions = local @ rotation.reshape(3, 3).T + [*xy, height]
        particles = np.column_stack([positions, np.tile([1, 0, 0, 0], (self.water_count, 1))])
        return np.r_[base, particles.ravel()]

    def reset(self):
        self._last_metric_time = 0.0
        self.rolling_distance = 0.0
        self.axial_rotation = 0.0
        super().reset()
        self.data.time = 0.0

    def _quantities(self, states, sensors):
        o, v = self.object_pose_start, self.object_vel_start
        pos = states[..., o : o + 3]
        vel = states[..., v : v + 3]
        omega = states[..., v + 3 : v + 6]  # free-joint angular velocity: BODY frame
        axis = sensors[..., self.object_z_axis_start : self.object_z_axis_start + 3]
        cos = np.clip(axis[..., 2], -1.0, 1.0)
        return pos, vel, omega, axis, cos

    def _roll_speeds(self, pos, vel, omega, axis):
        direction = np.asarray(self.config.goal_pos)[:2] - np.asarray(self.config.start_pos)[:2]
        direction = direction / max(np.linalg.norm(direction), 1e-8)
        forward = np.sum(vel[..., :2] * direction, axis=-1)
        rolling_velocity = self.config.radius * omega[..., 2, None] * np.cross(axis, [0.0, 0.0, 1.0])
        spin_forward = np.sum(rolling_velocity[..., :2] * direction, axis=-1)
        grounded = (pos[..., 2] < 0.20) & (np.abs(axis[..., 2]) < 0.3)
        coupled = np.where(grounded, np.clip(np.minimum(forward, spin_forward), 0, 1), 0.0)
        slip = np.linalg.norm(vel[..., :2] - rolling_velocity[..., :2], axis=-1)
        return coupled, slip

    def reward_terms(self, states, sensors, controls, system_metadata=None):
        c = self.config
        pos, vel, omega, axis, cos = self._quantities(states, sensors)
        body = states[..., self.body_pose_start : self.body_pose_start + 3]
        distance = np.linalg.norm(pos[..., :2] - np.asarray(c.goal_pos)[:2], axis=-1)
        aligned = (
            cos > np.cos(np.deg2rad(c.angle_tolerance_deg))
            if self.mode == "upright"
            else np.abs(cos) < np.sin(np.deg2rad(c.angle_tolerance_deg))
        )
        near = np.exp(-np.square(distance / c.position_tolerance))
        settle = aligned * near
        if self.mode == "lay_down":
            # Penalize the incoming impact too, before the jug reaches horizontal.
            settle = np.ones_like(cos)
        if self.mode == "upright":
            orientation = 1.0 - cos
            gripper = sensors[..., self.gripper_start : self.gripper_start + 3]
            approach = np.linalg.norm(gripper - np.asarray(c.gripper_target_local), axis=-1)
            height_error = np.abs(pos[..., 2] - 0.2415) * np.clip(cos, 0, 1)
        else:
            orientation = np.abs(cos)
            if self.mode in ("roll", "move"):
                direction = np.asarray(c.goal_pos)[:2] - pos[..., :2]
                direction = direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 0.1)
                approach = np.linalg.norm(body[..., :2] - (pos[..., :2] - 0.65 * direction), axis=-1)
                # Near the goal the push direction is undefined (d -> 0 flips `direction`
                # with every jitter) and chasing the standoff point walked the robot around
                # the jug and into it (rehearsal 2026-09-16: jug parked at B, then kicked
                # 8 m away). Fade the standoff term out as the jug arrives; the settle terms
                # and the control cost take over.
                approach = approach * (1.0 - near)
            else:
                approach = np.linalg.norm(body[..., :2] - pos[..., :2], axis=-1) * np.abs(cos)
            height_error = np.maximum(pos[..., 2] - 0.17, 0)
        terms = {
            "orientation": -c.w_orientation * orientation.mean(-1),
            "position": -c.w_position * distance.mean(-1),
            "approach": -c.w_approach * approach.mean(-1),
            "height": -c.w_height * height_error.mean(-1),
            "linear_velocity": -c.w_linear_velocity * (settle * np.linalg.norm(vel, axis=-1)).mean(-1),
            "angular_velocity": -c.w_angular_velocity * (settle * np.linalg.norm(omega, axis=-1)).mean(-1),
            "controls": -c.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1),
            "look": object_look_term(states[..., : self.model.nq], self.body_pose_start,
                                     self.object_pose_start, c.w_look_object, c.look_ramp_dist),
            "fall": -c.fall_penalty * (body[..., 2] <= c.spot_fallen_threshold).any(-1),
        }
        if self.mode == "roll":
            coupled, slip = self._roll_speeds(pos, vel, omega, axis)
            terms["roll"] = c.w_roll * (coupled * (1 - near)).mean(-1)
            terms["slip"] = -c.w_slip * slip.mean(-1)
        return terms

    def reward(self, states, sensors, controls, system_metadata=None):
        total = np.zeros(states.shape[0])
        for term in self.reward_terms(states, sensors, controls, system_metadata).values():
            total += term
        return total

    def post_sim_step(self):
        # Idempotent: hierarchical backend also calls this before updating data.
        now = float(self.data.time)
        dt = now - self._last_metric_time
        if self.mode != "roll" or dt <= 0 or not hasattr(self, "object_pose_start"):
            return
        self._last_metric_time = now
        state = np.concatenate([self.data.qpos, self.data.qvel])[None, None]
        pos, vel, omega, axis, _ = self._quantities(state, self.data.sensordata[None, None])
        coupled, _ = self._roll_speeds(pos, vel, omega, axis)
        self.rolling_distance += float(coupled.item()) * dt
        self.axial_rotation += abs(float(omega[..., 2].item())) * dt

    def metrics(self, data):
        o, v = self.object_pose_start, self.object_vel_start - self.model.nq
        quat = data.qpos[o + 3 : o + 7]
        quat = quat / np.linalg.norm(quat)
        cos = float(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))
        result = {
            "tilt_deg": float(np.rad2deg(np.arccos(cos))),
            "goal_distance": float(np.linalg.norm(data.qpos[o : o + 2] - np.asarray(self.config.goal_pos)[:2])),
            "jug_height": float(data.qpos[o + 2]),
            "linear_speed": float(np.linalg.norm(data.qvel[v : v + 3])),
            "angular_speed": float(np.linalg.norm(data.qvel[v + 3 : v + 6])),
            "robot_height": float(data.qpos[self.body_pose_start + 2]),
            "rolling_distance": self.rolling_distance,
            "axial_rotation": self.axial_rotation,
        }
        if self.water_count:
            local = jug_water.ball_local_positions(self.model, data, self.water_count)
            result["water_retained_fraction"] = float(jug_water.contained_mask(local).mean())
            result["water_com_local"] = local.mean(0).tolist()
        return result

    def success(self, model, data, metadata=None):
        m, c = self.metrics(data), self.config
        aligned = (
            m["tilt_deg"] < c.angle_tolerance_deg
            if self.mode == "upright"
            else abs(m["tilt_deg"] - 90) < c.angle_tolerance_deg
        )
        grounded = abs(m["jug_height"] - 0.2415) < 0.04 if self.mode == "upright" else 0.08 < m["jug_height"] < 0.20
        if self.mode == "move":
            # Any final orientation is acceptable; support may be on the side or base.
            aligned = True
            grounded = 0.08 < m["jug_height"] < 0.32
        rolled = self.mode != "roll" or (m["rolling_distance"] >= c.min_roll_distance and m["axial_rotation"] >= np.pi)
        return bool(
            aligned
            and grounded
            and rolled
            and m["goal_distance"] < c.position_tolerance
            and m["linear_speed"] < c.linear_speed_tolerance
            and m["angular_speed"] < c.angular_speed_tolerance
            and m["robot_height"] > c.spot_fallen_threshold
        )

    def failure(self, model, data, metadata=None):
        return bool(data.qpos[self.body_pose_start + 2] <= self.config.spot_fallen_threshold)


class SpotJugUpright(SpotJugManipulation):
    name = "spot_jug_upright"
    config_t = SpotJugUprightConfig
    mode = "upright"


class SpotJugLayDown(SpotJugManipulation):
    name = "spot_jug_lay_down"
    config_t = SpotJugLayDownConfig
    mode = "lay_down"


class SpotJugRoll(SpotJugManipulation):
    name = "spot_jug_roll"
    config_t = SpotJugRollConfig
    mode = "roll"


class SpotJugMove(SpotJugManipulation):
    name = "spot_jug_move"
    config_t = SpotJugMoveConfig
    mode = "move"
