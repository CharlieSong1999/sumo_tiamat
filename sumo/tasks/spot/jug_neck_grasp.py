"""Opt-in neck-grasp experiment: geometry, observations and no-velocity shaping."""

import mujoco
import numpy as np

from sumo import MODEL_PATH

JAW_GEOMS = ("left_jaw_collision", "right_jaw_collision", "bottom_jaw_collision", "front_jaw_collision")
TARGET = np.array([0.0, 0.0, 0.205])


def build_neck_grasp(spec):
    # A single convex hull fills the narrow neck (r~6.5 cm at z=.20 instead
    # of mesh r~2.5 cm). Keep the visual and body frame; split collision there.
    path = MODEL_PATH / "meshes/jug/jug_meshy_20k.obj"
    with path.open() as stream:
        vertices = np.array([[float(x) for x in line.split()[1:4]] for line in stream if line.startswith("v ")])
    spec.add_mesh(name="jug_grasp_body", uservert=vertices[vertices[:, 2] <= 0.17].ravel())
    spec.geom("jug_collision").meshname = "jug_grasp_body"
    spec.body("jug").add_geom(
        name="jug_neck_collision",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.205],
        size=[0.029, 0.0365, 0],
        contype=1,
        conaffinity=1,
        condim=3,
        priority=6,
        friction=[0.6, 0.005, 0.0001],
        group=3,
        mass=0,
    )
    # Original site_arm_link_fngr is the finger hinge, NOT the pinch centre.
    spec.body("arm_link_wr1").add_site(name="jug_pinch_center", pos=[0.185, 0, -0.008], size=[0.004, 0, 0], group=3)
    sensor = spec.add_sensor(name="jug_pinch_local")
    sensor.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
    sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
    sensor.objname = "jug_pinch_center"
    sensor.reftype = mujoco.mjtObj.mjOBJ_SITE
    sensor.refname = "site_object"
    sensor = spec.add_sensor(name="jug_pinch_forward")
    sensor.type = mujoco.mjtSensor.mjSENS_FRAMEXAXIS
    sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
    sensor.objname = "jug_pinch_center"
    for name in JAW_GEOMS:
        sensor = spec.add_sensor(name=f"jug_neck_distance_{name}")
        sensor.type = mujoco.mjtSensor.mjSENS_GEOMDIST
        sensor.objtype = sensor.reftype = mujoco.mjtObj.mjOBJ_GEOM
        sensor.objname = name
        sensor.refname = "jug_neck_collision"
        sensor.cutoff = 0.3


def grasp_quantities(task, states, sensors, controls):
    idx = task.get_sensor_start_index("jug_pinch_local")
    local = sensors[..., idx : idx + 3]
    distance = np.linalg.norm(local - TARGET, axis=-1)
    idx = task.get_sensor_start_index("jug_pinch_forward")
    forward = sensors[..., idx : idx + 3]
    axis = sensors[..., task.object_z_axis_start : task.object_z_axis_start + 3]
    # Approach the cap axially, pointing from the mouth towards the barrel body.
    alignment = 1 + np.clip(np.sum(forward * axis, axis=-1), -1, 1)
    q = states[..., task.get_joint_position_start_index("arm_f1x")]
    # The selection channel overrides the raw finger command when negative.
    command = np.where(controls[..., 10] < 0, 0, controls[..., 9])
    resistance = np.maximum(command - q, 0)
    gaps = np.stack([sensors[..., task.get_sensor_start_index(f"jug_neck_distance_{name}")] for name in JAW_GEOMS], -1)
    upper = np.min(gaps[..., :2], axis=-1)
    lower = np.min(gaps[..., 2:], axis=-1)
    bilateral = (upper < 0.002) & (lower < 0.002)
    closing = command > -0.25
    grasp = (distance < 0.065) & closing & (resistance > 0.12) & (q < -0.05) & bilateral
    return distance, alignment, q, closing, resistance, bilateral, grasp


def grasp_terms(task, states, sensors, controls):
    distance, alignment, q, closing, resistance, bilateral, grasp = grasp_quantities(task, states, sensors, controls)
    near = np.exp(-np.square(distance / 0.10))
    # Smoothly stop paying for grasp acquisition once the jug is upright.
    axis = sensors[..., task.object_z_axis_start : task.object_z_axis_start + 3]
    needs_lift = np.clip((1 - axis[..., 2]) / 0.25, 0, 1)
    upright = np.clip((axis[..., 2] - 0.85) / 0.13, 0, 1)
    local_idx = task.get_sensor_start_index("jug_pinch_local")
    target = np.broadcast_to(TARGET, sensors[..., local_idx : local_idx + 3].shape).copy()
    target[..., 2] += 0.10 * upright  # Withdraw above the cap after setting down.
    reach_distance = np.linalg.norm(sensors[..., local_idx : local_idx + 3] - target, axis=-1)
    gaps = np.stack([sensors[..., task.get_sensor_start_index(f"jug_neck_distance_{name}")] for name in JAW_GEOMS], -1)
    straddle_error = np.maximum(np.min(gaps[..., :2], axis=-1), 0) + np.maximum(np.min(gaps[..., 2:], axis=-1), 0)
    command = np.where(controls[..., 10] < 0, 0, controls[..., 9])
    c = task.config
    return {
        "neck_reach": -c.w_neck_reach * reach_distance.mean(-1),
        "neck_alignment": -c.w_neck_alignment * (alignment * np.exp(-np.square(distance / 0.4))).mean(-1),
        "neck_open": -c.w_neck_open * ((1 - near * (1 - upright)) * np.square(q + 1.1)).mean(-1),
        "neck_close": -c.w_neck_close * (near * (1 - upright) * np.abs(command)).mean(-1),
        "neck_straddle": -c.w_neck_straddle * (near * straddle_error * (1 - upright)).mean(-1),
        # Fade over the whole lift. Fading a 40-point bonus over only .25 in
        # cosine costs 160 points per unit cosine, defeating the 150-point
        # upright term and creating a partly-upright holding local optimum.
        "neck_grasp": c.w_neck_grasp
        * (grasp * np.clip(resistance / 0.5, 0, 1) * np.clip((1 - axis[..., 2]) / c.neck_grasp_fade_width, 0, 1)).mean(
            -1
        ),
        "neck_empty_close": -c.w_neck_false * (near * closing * (q > -0.05) * ~bilateral * needs_lift).mean(-1),
    }


def grasp_metrics(task, data, applied_control=None):
    # The hierarchical backend copies qpos/qvel but not data.ctrl. Obtain the
    # actual previous high-level command from the runner, never from stale ctrl.
    command = np.zeros((1, 1, task.nu))
    if applied_control is not None:
        command[:] = applied_control
    values = grasp_quantities(task, np.r_[data.qpos, data.qvel][None, None], data.sensordata[None, None], command)
    distance, alignment, q, closing, resistance, bilateral, grasp = values
    result = {
        "pinch_distance": float(distance.item()),
        "pinch_alignment_error": float(alignment.item()),
        "finger_angle": float(q.item()),
        "neck_bilateral_contact": bool(bilateral.item()),
    }
    if applied_control is not None:
        result["neck_grasp_detected"] = bool(grasp.item())
        result["finger_closing_resistance"] = float(resistance.item())
    return result
