"""Jug manipulation using the production tracking mesh."""

from dataclasses import dataclass

import mujoco
import numpy as np
from judo.utils.fields import np_1d_field

from sumo.tasks.spot import jug_neck_grasp, jug_water
from sumo.tasks.spot.look_at import object_look_term
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import ARM_JOINT_NAMES, LEGS_STANDING_POS, STANDING_HEIGHT
from sumo.tasks.spot.spot_jug_kick import XML_PATH
from sumo.tasks.spot.spot_upright import ground_clearance_height


@dataclass
class SpotJugManipulationConfig(SpotBaseConfig):
    # Opt-in experiment only; existing tasks and deployment defaults are unchanged.
    neck_grasp: bool = False
    w_neck_reach: float = 60.0
    w_neck_alignment: float = 20.0
    w_neck_open: float = 5.0
    w_neck_grasp: float = 40.0
    w_neck_false: float = 15.0
    w_neck_straddle: float = 0.0
    w_neck_close: float = 0.0
    neck_grasp_fade_width: float = 1.0
    # Construction-time model parameters; rebuild the task to change these.
    # Water balls are opt-in (rebuild to change). The planner can place them inside the
    # perceived jug (synthesize_qpos), but five 2.5 cm balls (0.026 = ~0.5 kg, the stones
    # in the real jug) cost 93 ms per plan instead of 30 in the 2026-09-15 rehearsal --
    # the CG/sparse solver they need -- and the deployed loop has a 150 ms budget. Until
    # the budget allows it the stones stay lumped into jug_mass.
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
    max_base_speed: float = 0.3
    # Yaw-only "face the jug" term (look_at.object_look_term), faded out as the jug
    # comes under the robot (w = w_look_object * (1 - exp(-(d/look_ramp_dist)^2))), so
    # the heading target cannot flip around when the jug is between the feet.
    w_look_object: float = 10.0
    look_ramp_dist: float = 0.6
    # Yaw-rate command floor, see spot_base.yaw_command_floor and look_at.LookAtPointFields
    # (same values, same reason; not a --task-set field).
    yaw_rate_min: float = 0.4
    yaw_rate_deadzone: float = 0.1
    xy_speed_deadzone: float = 0.08   # see look_at.LookAtPointFields
    w_orientation: float = 150.0
    w_approach: float = 12.0
    w_position: float = 15.0
    w_linear_velocity: float = 8.0
    w_angular_velocity: float = 2.0
    w_controls: float = 0.3
    w_height: float = 100.0
    position_tolerance: float = 0.35
    # Arrival deadband for the pushing modes (roll/move): with the jug within
    # arrive_tolerance of goal_pos the position term is flat, approach/roll are off and a
    # standoff term keeps the body >= standoff_dist from the jug. Real robot 2026-09-15:
    # a hold (goal frozen ON the jug) plus 17 cm of perception drift became a "roll 17 cm
    # in a noise-defined direction" and the robot kicked the jug 1.1 m.
    arrive_tolerance: float = 0.25
    # The push terms come back gradually over arrive_ramp past the tolerance: a jug that
    # overshoots B by a few cm gets a weak pull, not a full re-approach from the far side
    # (rehearsal 2026-09-15: a 2 cm overshoot sent the robot around the jug to push it back).
    arrive_ramp: float = 0.3
    standoff_dist: float = 0.6
    w_standoff: float = 40.0
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
    # Construction-time opt-in: keep the deployed/default roll action space at nu=3.
    # Rebuild with True to expose the arm and gripper-selection controls (nu=11).
    roll_use_arm: bool = False
    # Opt-in costs on actual predicted velocities of the six arm joints (not
    # gripper closure, base motion, or jug velocity). The peak term catches brief
    # whips that a horizon-average cost can otherwise dilute. This is NOT a limit.
    w_arm_speed: float = 0.0
    w_arm_peak_speed: float = 0.0
    arm_speed_soft_limit: float = 1.0  # rad/s, per joint
    # Optional body-to-jug XY clearance to discourage replacing arm pushes with
    # leg kicks when arm motion has a cost. This proxy is not contact detection.
    w_arm_body_clearance: float = 0.0
    arm_body_clearance: float = 0.75  # m, between base and jug centers
    # Optional Cartesian hand guidance. Sensors have no mass/collision geometry.
    w_hand_speed: float = 0.0
    w_hand_peak_speed: float = 0.0
    hand_speed_soft_limit: float = 0.35  # world-frame m/s, not a hard limit
    w_hand_reach: float = 0.0
    w_arm_rest: float = 0.0
    hand_reach_backoff: float = 0.16  # hand point behind jug center, along A->B
    hand_reach_height: float = 0.06  # above jug center
    w_position: float = 100.0
    w_approach: float = 15.0
    w_orientation: float = 40.0
    w_roll: float = 25.0
    w_slip: float = 4.0
    # Rolling faster than this earns nothing, and jug speed above it costs w_overspeed
    # per m/s: a jug kicked to 0.75 m/s rolls 2 m past B on rolling friction 0.01
    # (rehearsal 2026-09-15; real robot the same day: one contact sent it 1.1 m).
    roll_speed_cap: float = 0.3
    w_overspeed: float = 40.0
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
    for name in ("w_linear_velocity", "w_angular_velocity", "w_roll", "w_slip", "w_overspeed"):
        if hasattr(config, name):
            setattr(config, name, 0.0)


class SpotJugManipulation(SpotBase):
    config_t = SpotJugManipulationConfig
    mode = "upright"
    perceived_object_joints = ("jug_joint",)

    def __init__(self, config=None):
        water_config = config if config is not None else self.config_t()
        self.neck_grasp = bool(water_config.neck_grasp)
        if self.neck_grasp and self.mode != "upright":
            raise ValueError("neck_grasp is an upright-only experiment")
        if self.neck_grasp and not (
            np.isfinite(water_config.neck_grasp_fade_width) and water_config.neck_grasp_fade_width > 0
        ):
            raise ValueError("neck_grasp_fade_width must be finite and positive")
        # judo's base builds the model (_process_spec) BEFORE it stores the passed config,
        # so every construction-time parameter is read from `water_config` here.
        self.jug_mass = float(water_config.jug_mass)
        self.rolling_friction = float(water_config.rolling_friction)
        if not (np.isfinite(self.jug_mass) and self.jug_mass > 0):
            raise ValueError(f"jug_mass must be finite and positive, got {water_config.jug_mass!r}")
        if not (np.isfinite(self.rolling_friction) and self.rolling_friction >= 0):
            raise ValueError(f"rolling_friction must be finite and >= 0, got {water_config.rolling_friction!r}")
        ramp = float(getattr(water_config, "arrive_ramp", 1.0))
        if not (np.isfinite(ramp) and ramp > 0):
            raise ValueError(f"arrive_ramp must be finite and > 0, got {water_config.arrive_ramp!r}")
        self.water_radius = water_config.water_ball_radius
        self.water_count, self.water_mass = jug_water.water_parameters(water_config.water_fill_ratio, self.water_radius)
        use_arm = self.mode in ("upright", "arm_idle") or (
            self.mode == "roll" and bool(getattr(water_config, "roll_use_arm", False))
        )
        if self.mode == "roll":
            for name in (
                "w_arm_speed", "w_arm_peak_speed", "arm_speed_soft_limit",
                "w_arm_body_clearance", "arm_body_clearance",
                "w_hand_speed", "w_hand_peak_speed", "hand_speed_soft_limit",
                "w_hand_reach", "w_arm_rest", "hand_reach_backoff", "hand_reach_height",
            ):
                value = getattr(water_config, name)
                if not (np.isfinite(value) and value >= 0):
                    raise ValueError(f"{name} must be finite and nonnegative")
        self.roll_hand_sensors = self.mode == "roll" and use_arm and any(
            getattr(water_config, name, 0) for name in ("w_hand_speed", "w_hand_peak_speed", "w_hand_reach")
        )
        super().__init__(
            model_path=XML_PATH,
            use_arm=use_arm,
            use_gripper=use_arm,
            config=config,
        )
        self.body_pose_start = self.get_joint_position_start_index("base")
        self.object_pose_start = self.get_joint_position_start_index("jug_joint")
        self.object_vel_start = self.get_joint_velocity_start_index("jug_joint")
        self.object_z_axis_start = self.get_sensor_start_index("object_z_axis")
        self.gripper_start = self.get_sensor_start_index("sensor_arm_link_fngr")
        self.arm_velocity_indices = np.array(
            [self.get_joint_velocity_start_index(name) for name in ARM_JOINT_NAMES if name != "arm_f1x"]
        )
        self.arm_position_indices = np.array(
            [self.get_joint_position_start_index(name) for name in ARM_JOINT_NAMES if name != "arm_f1x"]
        )
        if self.roll_hand_sensors:
            self.hand_position_start = self.get_sensor_start_index("jug_roll_hand_position")
            self.hand_velocity_start = self.get_sensor_start_index("jug_roll_hand_velocity")

    def _process_spec(self):
        super()._process_spec()
        if self.roll_hand_sensors:
            self.spec.body("arm_link_wr1").add_site(
                name="jug_roll_hand_point", pos=[0.185, 0, -0.008], size=[0.004, 0, 0], group=3,
            )
            for name, kind in (
                ("jug_roll_hand_position", mujoco.mjtSensor.mjSENS_FRAMEPOS),
                ("jug_roll_hand_velocity", mujoco.mjtSensor.mjSENS_FRAMELINVEL),
            ):
                sensor = self.spec.add_sensor(name=name)
                sensor.type = kind
                sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
                sensor.objname = "jug_roll_hand_point"  # no reference: WORLD frame
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
        if self.neck_grasp:
            jug_neck_grasp.build_neck_grasp(self.spec)
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

    @property
    def synthesized_joints(self) -> tuple[str, ...]:
        """Free joints the planner cannot observe and must fill from the jug's pose (water)."""
        return tuple(f"jug_water_{i}_joint" for i in range(self.water_count))

    def synthesize_qpos(self, qpos: np.ndarray) -> None:
        """Fill the water balls' qpos in place from the jug's free-joint qpos (sumo_server.scene).

        The balls are put where settled water would sit for the jug's current
        orientation (jug_water.pooled_local_positions); their quaternions are identity.
        """
        if not self.water_count:
            return
        o = self.object_pose_start
        jug_pos, jug_quat = qpos[o : o + 3], np.array(qpos[o + 3 : o + 7], dtype=float)
        norm = float(np.linalg.norm(jug_quat))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"jug quaternion {jug_quat.tolist()} is degenerate; cannot place the water")
        jug_quat = jug_quat / norm
        local = jug_water.pooled_local_positions(self.water_count, self.water_radius, jug_quat)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(jug_quat, dtype=float))
        world = local @ rot.reshape(3, 3).T + jug_pos
        for i, name in enumerate(self.synthesized_joints):
            adr = self.model.jnt_qposadr[self.model.joint(name).id]
            qpos[adr : adr + 3] = world[i]
            qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

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
        cap = float(getattr(self.config, "roll_speed_cap", 1.0))
        coupled = np.where(grounded, np.clip(np.minimum(forward, spin_forward), 0, cap), 0.0)
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
        pushing = self.mode in ("roll", "move")
        # Arrival deadband (pushing modes): flat position cost inside arrive_tolerance and
        # a per-step gate that switches the push terms off / the standoff on.
        if pushing:
            push = np.clip((distance - c.arrive_tolerance) / c.arrive_ramp, 0.0, 1.0)   # 0 arrived .. 1 pushing
            position_cost = np.maximum(distance - c.arrive_tolerance, 0.0)
        else:
            push, position_cost = np.ones_like(distance), distance
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
                approach = approach * (1.0 - near) * push
            else:
                approach = np.linalg.norm(body[..., :2] - pos[..., :2], axis=-1) * np.abs(cos)
            height_error = np.maximum(pos[..., 2] - 0.17, 0)
        terms = {
            "orientation": -c.w_orientation * orientation.mean(-1),
            "position": -c.w_position * position_cost.mean(-1),
            "approach": -c.w_approach * approach.mean(-1),
            "height": -c.w_height * height_error.mean(-1),
            "linear_velocity": -c.w_linear_velocity * (settle * np.linalg.norm(vel, axis=-1)).mean(-1),
            "angular_velocity": -c.w_angular_velocity * (settle * np.linalg.norm(omega, axis=-1)).mean(-1),
            "controls": -c.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1),
            "look": object_look_term(
                states[..., : self.model.nq],
                self.body_pose_start,
                self.object_pose_start,
                c.w_look_object,
                c.look_ramp_dist,
            ),
            "fall": -c.fall_penalty * (body[..., 2] <= c.spot_fallen_threshold).any(-1),
        }
        if pushing:
            body_dist = np.linalg.norm(body[..., :2] - pos[..., :2], axis=-1)
            standoff = np.maximum(c.standoff_dist - body_dist, 0.0) * (1.0 - push)
            terms["standoff"] = -c.w_standoff * standoff.mean(-1)
        if self.mode == "roll":
            coupled, slip = self._roll_speeds(pos, vel, omega, axis)
            terms["roll"] = c.w_roll * (coupled * (1 - near) * push).mean(-1)
            terms["slip"] = -c.w_slip * slip.mean(-1)
            overspeed = np.maximum(np.linalg.norm(vel[..., :2], axis=-1) - c.roll_speed_cap, 0.0)
            terms["overspeed"] = -c.w_overspeed * overspeed.mean(-1)
            if self.use_arm and (c.w_arm_speed or c.w_arm_peak_speed):
                arm_velocity = states[..., self.arm_velocity_indices]
                excess = np.maximum(np.abs(arm_velocity) - c.arm_speed_soft_limit, 0.0)
                terms["arm_speed"] = -c.w_arm_speed * np.square(arm_velocity).sum(-1).mean(-1)
                terms["arm_peak_speed"] = -c.w_arm_peak_speed * np.square(excess).max(axis=(-2, -1))
            if self.use_arm and c.w_arm_body_clearance:
                clearance_error = np.maximum(c.arm_body_clearance - body_dist, 0.0)
                terms["arm_body_clearance"] = -c.w_arm_body_clearance * clearance_error.max(-1)
            if self.roll_hand_sensors:
                h = sensors[..., self.hand_position_start : self.hand_position_start + 3]
                hv = sensors[..., self.hand_velocity_start : self.hand_velocity_start + 3]
                excess = np.maximum(np.linalg.norm(hv, axis=-1) - c.hand_speed_soft_limit, 0.0)
                terms["hand_speed"] = -c.w_hand_speed * np.square(excess).mean(-1)
                terms["hand_peak_speed"] = -c.w_hand_peak_speed * np.square(excess).max(-1)
                # This is a hand-selected approach point, not a prescribed motion
                # or a contact constraint. Its dense gradient helps slow approaches.
                route = np.asarray(c.goal_pos)[:2] - np.asarray(c.start_pos)[:2]
                route = route / max(np.linalg.norm(route), 1e-8)
                target = pos.copy()
                target[..., :2] -= c.hand_reach_backoff * route
                target[..., 2] += c.hand_reach_height
                reach = np.maximum(np.linalg.norm(h - target, axis=-1) - 0.04, 0.0)
                terms["hand_reach"] = -c.w_hand_reach * (push * reach).mean(-1)
            if self.use_arm and c.w_arm_rest:
                # Return softly to the existing neutral extended reset posture as
                # the original arrival deadband turns off pushing. No mode latch.
                error = states[..., self.arm_position_indices] - np.asarray(self.reset_arm_pos)[:6]
                terms["arm_rest"] = -c.w_arm_rest * ((1 - push) * np.square(error).sum(-1)).mean(-1)
        if self.neck_grasp:
            # Replace hinge-to-neck attraction with pinch-centre guidance.
            terms["approach"] = np.zeros_like(terms["approach"])
            terms.update(jug_neck_grasp.grasp_terms(self, states, sensors, controls))
        return terms

    def reward(self, states, sensors, controls, system_metadata=None):
        if self.mode == "arm_idle":
            return self._arm_idle_reward(states, controls)
        total = np.zeros(states.shape[0])
        for term in self.reward_terms(states, sensors, controls, system_metadata).values():
            total += term
        return total

    def _arm_idle_reward(self, states, controls):
        """Stand where the goal is, arm stowed, hands off the jug.

        The neutral member of the nu=11 family. The policy node is launched on it and the operator switches to
        spot_jug_upright / spot_jug_upright_grasp from the monitor (the policy honours
        switches only within one nu). Nothing here reads the jug.
        """
        c = self.config
        qpos = states[..., : self.model.nq]
        body = qpos[..., self.body_pose_start : self.body_pose_start + 3]
        arm = qpos[..., self.body_pose_start + 7 + 12 : self.body_pose_start + 7 + 19]
        goal = -c.w_goal * np.linalg.norm(body - np.asarray(c.goal_pos)[None, None], axis=-1).mean(-1)
        stow = -c.w_arm_stow * np.square(arm - np.asarray(self.reset_arm_pos)[None, None]).sum(-1).mean(-1)
        fall = -c.fall_penalty * (body[..., 2] <= c.spot_fallen_threshold).any(-1)
        ctrl = -c.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)
        return goal + stow + fall + ctrl

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
        if self.neck_grasp:
            result.update(jug_neck_grasp.grasp_metrics(self, data))
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


@dataclass
class SpotJugArmIdleConfig(SpotJugManipulationConfig):
    """Neutral nu=11 task: hold position, arm at its stowed pose."""

    w_arm_stow: float = 20.0
    w_controls: float = 0.3
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, STANDING_HEIGHT]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_pos",
        xyz_vis_indices=[0, 1, None],
    )


class SpotJugArmIdle(SpotJugManipulation):
    name = "spot_jug_arm_idle"
    config_t = SpotJugArmIdleConfig
    mode = "arm_idle"


@dataclass
class SpotJugUprightGraspConfig(SpotJugUprightConfig):
    """Upright with the explicit neck grasp (jug_neck_grasp).

    Corrected neck collision geometry and pinch-centre guidance. A separate task because
    the switch is construction-time and the deployed planner builds tasks from defaults.
    """

    neck_grasp: bool = True


class SpotJugUprightGrasp(SpotJugUpright):
    name = "spot_jug_upright_grasp"
    config_t = SpotJugUprightGraspConfig


class SpotJugLayDown(SpotJugManipulation):
    name = "spot_jug_lay_down"
    config_t = SpotJugLayDownConfig
    mode = "lay_down"


class SpotJugRoll(SpotJugManipulation):
    name = "spot_jug_roll"
    config_t = SpotJugRollConfig
    mode = "roll"


@dataclass
class SpotJugRollArmGentleConfig(SpotJugRollConfig):
    """Frozen deployment profile: gentle arm roll v4 with the 4 cm hand-guidance sphere.

    The values are the `initial_config` of the authoritative 32x1 / 1.5 s run
    (out/jug_roll_horizon_tradeoff_20260917/n32_h1p5/front/seed0.json; handoff in
    auto_sumo/docs/GE_test/20260917_gentle_arm_roll_point_deployment_handoff.md §3.1),
    pinned by tests/test_jug_roll_arm_gentle.py. A separate registered task because
    `roll_use_arm` and `water_fill_ratio` are construction-time (nu 3 -> 11, hand
    sensors, 19 water balls) and the deployed planner/policy build tasks from defaults.
    `start_pos` / `goal_pos` keep the base defaults: on the robot A is the first jug
    observation and B the operator's click (run_planner), never these offline coordinates.
    """

    roll_use_arm: bool = True
    water_fill_ratio: float = 0.1
    # Jug velocity costs off: the arm/hand costs below shape the push instead.
    w_linear_velocity: float = 0.0
    w_angular_velocity: float = 0.0
    w_roll: float = 0.0
    w_slip: float = 0.0
    w_overspeed: float = 0.0
    w_arm_speed: float = 0.5
    w_arm_peak_speed: float = 2.0
    w_arm_body_clearance: float = 400.0
    w_hand_speed: float = 2.0
    w_hand_peak_speed: float = 10.0
    hand_speed_soft_limit: float = 0.5
    w_hand_reach: float = 15.0
    w_arm_rest: float = 10.0


class SpotJugRollArmGentle(SpotJugRoll):
    name = "spot_jug_roll_arm_gentle"
    config_t = SpotJugRollArmGentleConfig


@dataclass
class SpotJugRollArmGentleDryConfig(SpotJugRollArmGentleConfig):
    """The gentle arm-roll profile without explicit water.

    Same reward and action space; the 3.4 kg total load lumped into the jug body (1.5 kg
    jug + 1.9 kg of the wet profile's 19 balls). Registered separately because on the deployed chain
    (2026-09-17 rehearsals) the wet profile plans at 12 Hz (75-94 ms/plan) and never
    approaches the jug, while this one plans at 20 Hz and pushes it to B. It is a
    different model, not a tuning of the frozen one; sloshing is not represented.
    """

    water_fill_ratio: float = 0.0
    jug_mass: float = 3.4


class SpotJugRollArmGentleDry(SpotJugRollArmGentle):
    name = "spot_jug_roll_arm_gentle_dry"
    config_t = SpotJugRollArmGentleDryConfig


class SpotJugMove(SpotJugManipulation):
    name = "spot_jug_move"
    config_t = SpotJugMoveConfig
    mode = "move"
