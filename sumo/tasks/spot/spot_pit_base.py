# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Shared machinery for the Spot pit-pitcher water tasks (move / pour).

The pit tasks use a planner/plant model split:

- The PLANNER task model contains Spot + the (empty or mass-augmented) pitcher. It is
  what the controller rolls out and what ``reward()`` is written against.
- The PLANT task model is the same XML with N water balls appended at the END of the
  spec, so the shared robot/pitcher state prefix is index-identical and the plant
  state maps onto the planner state by slicing (see :func:`project_state`).

Water amount is parameterized by ``fill_ratio`` (fraction of the pitcher cavity
filled), mapped to a ball count via the settled-packing calibration from
``tools/water_ball_test.py`` and docs/water_ball_approximation.md.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from sumo import MODEL_PATH

PIT_XML_PATH = MODEL_PATH / "xml" / "objects" / "pit" / "pit.xml"

# Interior cavity of the pit body frame (must match pit.xml collision geometry).
CAVITY_HX, CAVITY_HY, CAVITY_TOP = 0.105, 0.066, 0.38
PIT_REST_Z = 0.02  # pit free-joint origin height when resting on flat ground

BALL_RADIUS = 0.015
BALL_DENSITY = 1560.0  # -> bulk density ~1000 kg/m^3 at random close packing ~0.64
BALL_MASS = BALL_DENSITY * (4.0 / 3.0) * np.pi * BALL_RADIUS**3  # ~0.022 kg

# Settled-packing calibration: ~24 balls per layer, ~0.028 m per settled layer ->
# a full cavity (0.38 m) holds ~326 balls.
BALLS_PER_FULL_CAVITY = 326

CACHE_DIR = Path(__file__).resolve().parents[3] / "out" / "pit_water_cache"

# Gripper servo stiffness calibration. The stock model drives arm_f1x with a SOFT
# position servo (kp=16 vs 120 for the arm joints), so the sustained clamp torque is
# kp * position_error: pinching the 12 mm handle plate (error ~0.26 rad) yields only
# ~4 N*m (~35 N at the fingertip) -- far below the actuator's own force limit of
# 15.3 N*m (~130 N fingertip, matching the real Spot gripper spec). A real gripper's
# closed-loop controller sustains clamp forces near its spec, so we stiffen the servo;
# the forcerange cap (the physical limit) is left untouched.
GRIPPER_KP_CALIBRATED = 50.0
GRIPPER_KV_CALIBRATED = 1.0


def scale_pitcher(spec: mujoco.MjSpec, scale: float) -> None:
    """Uniformly scale the pit.xml pitcher by ``scale`` (geometry, sites, inertial).

    Mass scales with volume (s^3), inertia with s^5, all positions/sizes with s.
    Must run BEFORE :func:`build_handle`: the handle is added afterwards with its
    grip cross-section kept at FULL size (the gripper is fixed-size hardware, so the
    grasp interface stays identical across scales -- only the load changes).
    """
    if scale == 1.0:
        return
    for mesh in spec.meshes:
        if mesh.name == "pit_mesh":
            mesh.scale = np.array(mesh.scale) * scale
    pit = spec.body("pit")
    for g in pit.geoms:
        g.pos = np.array(g.pos) * scale
        g.size = np.array(g.size) * scale
    for s in pit.sites:
        s.pos = np.array(s.pos) * scale
    pit.mass = float(pit.mass) * scale**3
    pit.ipos = np.array(pit.ipos) * scale
    pit.inertia = np.array(pit.inertia) * scale**5


def _set_pit_site(spec: mujoco.MjSpec, name: str, pos: list[float]) -> None:
    for s in spec.body("pit").sites:
        if s.name == name:
            s.pos = pos
            return


def build_handle(spec: mujoco.MjSpec, style: str, scale: float = 1.0) -> None:
    """Build the C-shaped handle on the pit body: top arm + vertical grip + bottom arm.

    style "plate": the vertical segment is a flat bar (40 mm wide along x, 14 mm thick
    along y -> planar pinch faces). style "cylinder": a 30 mm diameter capsule (same
    diameter as the jaw pads). At scale 1 the grip hangs at x=0.20, spanning z 0.16..0.36.

    Scaling policy (scale != 1): attach points and vertical extent follow the pitcher,
    but the grip CROSS-SECTION and its 75 mm standoff from the wall stay full-size --
    the jaw is fixed hardware, so the contact geometry of the grasp is unchanged and
    the scale experiment isolates the weight/inertia effect. The handle-related sites
    (grasp/straddle/hook) are repositioned to match (scale_pitcher scaled them along
    with everything else).
    """
    if style not in ("plate", "cylinder"):
        raise ValueError(f"handle_style must be 'plate' or 'cylinder', got {style!r}")
    pit = spec.body("pit")

    attach_x = 0.125 * scale  # anchored just inside the (scaled) +x wall
    grip_x = attach_x + 0.075  # standoff kept full-size: jaw clearance is hardware
    top_z, bot_z = 0.36 * scale, 0.16 * scale
    grip_mid_z = 0.5 * (top_z + bot_z)

    def _grip_geom(g):
        g.condim = 6
        g.priority = 8
        g.friction = [1.2, 0.05, 0.005]
        g.group = 3

    for name, fromto in (
        ("handle_arm_top", [attach_x, 0, top_z, grip_x, 0, top_z]),
        ("handle_arm_bottom", [attach_x, 0, bot_z, grip_x, 0, bot_z]),
    ):
        g = pit.add_geom(name=name)
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size[0] = 0.010
        g.fromto = fromto
        _grip_geom(g)
        v = pit.add_geom(name=f"{name}_visual")
        v.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        v.size[0] = 0.010
        v.fromto = fromto
        v.contype = 0
        v.conaffinity = 0
        v.group = 2
        v.rgba = [0.95, 0.55, 0.1, 1.0]

    grip_fromto = [grip_x, 0, bot_z + 0.01, grip_x, 0, top_z - 0.01]
    for name, is_visual in (("handle_grip", False), ("handle_grip_visual", True)):
        g = pit.add_geom(name=name)
        if style == "plate":
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.pos = [grip_x, 0, grip_mid_z]
            g.size = [0.020, 0.007, 0.5 * (top_z - bot_z)]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.size[0] = 0.015
            g.fromto = grip_fromto
        if is_visual:
            g.contype = 0
            g.conaffinity = 0
            g.group = 2
            g.rgba = [0.95, 0.55, 0.1, 1.0]
        else:
            _grip_geom(g)

    # Handle-related sites track the actual (scale-adjusted) handle geometry. The
    # straddle y-offsets stay full-size: they mirror the fixed jaw pad positions.
    _set_pit_site(spec, "site_grasp_handle", [grip_x, 0, grip_mid_z])
    _set_pit_site(spec, "trace_grasp_handle", [grip_x, 0, grip_mid_z])
    _set_pit_site(spec, "site_plate_finger", [grip_x, -0.035, grip_mid_z])
    _set_pit_site(spec, "site_plate_palm", [grip_x, 0.035, grip_mid_z])
    _set_pit_site(spec, "site_hook", [0.5 * (attach_x + grip_x), 0, top_z - 0.05 * scale])


def calibrate_gripper_actuator(spec: mujoco.MjSpec) -> None:
    """Stiffen the arm_f1x position servo (see GRIPPER_KP_CALIBRATED note)."""
    for act in spec.actuators:
        if act.target == "arm_f1x":
            act.gainprm[0] = GRIPPER_KP_CALIBRATED
            act.biasprm[1] = -GRIPPER_KP_CALIBRATED
            act.biasprm[2] = -GRIPPER_KV_CALIBRATED
            return
    raise ValueError("arm_f1x actuator not found in spec")


def fill_ratio_to_ball_count(fill_ratio: float, scale: float = 1.0) -> int:
    """Map a cavity fill ratio (0..1) to the number of water balls.

    ``scale`` is the pitcher scale factor: cavity volume (and thus ball count at
    fixed ball radius) shrinks with scale^3.
    """
    if not 0.0 <= fill_ratio <= 1.0:
        raise ValueError(f"fill_ratio must be in [0, 1], got {fill_ratio}")
    return int(round(fill_ratio * BALLS_PER_FULL_CAVITY * scale**3))


def ball_mass(radius: float = BALL_RADIUS) -> float:
    """Mass of one water ball of the given radius at BALL_DENSITY."""
    return BALL_DENSITY * (4.0 / 3.0) * np.pi * radius**3


def chunky_ball_count(fill_ratio: float, radius: float, scale: float = 1.0) -> int:
    """Number of radius-r balls carrying the same water mass as the fine-ball fill."""
    n_fine = fill_ratio_to_ball_count(fill_ratio, scale)
    return max(1, int(round(n_fine * (BALL_RADIUS / radius) ** 3)))


def ball_grid_positions(
    n_balls: int, radius: float = BALL_RADIUS, drop_z: float = 0.05, scale: float = 1.0
) -> np.ndarray:
    """Jittered grid of ball drop positions over the cavity opening, in the pit frame."""
    hx, hy = CAVITY_HX * scale, CAVITY_HY * scale
    margin = radius + 0.004
    xs = np.arange(-hx + margin, hx - margin + 1e-9, 2 * radius + 0.003)
    ys = np.arange(-hy + margin, hy - margin + 1e-9, 2 * radius + 0.003)
    # A small scaled cavity can be narrower than 2*margin: fall back to a center column.
    if xs.size == 0:
        xs = np.array([0.0])
    if ys.size == 0:
        ys = np.array([0.0])
    rng = np.random.default_rng(0)
    positions = []
    z = drop_z
    while len(positions) < n_balls:
        for x in xs:
            for y in ys:
                if len(positions) >= n_balls:
                    break
                jx, jy = rng.uniform(-1e-3, 1e-3, 2)
                positions.append([x + jx, y + jy, z])
            if len(positions) >= n_balls:
                break
        z += 2 * radius + 0.004
    return np.asarray(positions)


def settled_ball_poses(n_balls: int, radius: float = BALL_RADIUS, scale: float = 1.0) -> np.ndarray:
    """Settled ball positions in the pit frame, shape (n_balls, 3). Cached on disk.

    Computed once per (count, radius, scale) by dropping a grid of balls into a
    standalone pit-only scene (Newton solver, task timestep) and simulating 3 s.
    """
    suffix = "" if radius == BALL_RADIUS else f"_r{round(radius * 1000)}"
    if scale != 1.0:
        suffix += f"_s{round(scale * 100)}"
    cache_path = CACHE_DIR / f"settled_{n_balls}{suffix}.npz"
    if cache_path.exists():
        return np.load(cache_path)["positions"]

    ball_xml = "\n".join(
        f'<body name="ball_{i}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}">'
        f'<freejoint name="ball_{i}_joint"/><geom name="ball_{i}_geom" class="water_ball" size="{radius}"/></body>'
        for i, p in enumerate(ball_grid_positions(n_balls, radius=radius, scale=scale))
    )
    scene = f"""
<mujoco model="pit_settle">
  <option timestep="0.01" solver="Newton" integrator="implicitfast" density="1"/>
  <include file="{PIT_XML_PATH.as_posix()}"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.01" friction="0.7" priority="5"/>
    {ball_xml}
  </worldbody>
</mujoco>
"""
    if scale == 1.0:
        model = mujoco.MjModel.from_xml_string(scene)
    else:
        spec = mujoco.MjSpec.from_string(scene)
        scale_pitcher(spec, scale)
        model = spec.compile()
    data = mujoco.MjData(model)
    pit_adr = model.joint("pit_joint").qposadr[0]
    data.qpos[pit_adr + 2] = PIT_REST_Z * scale
    data.qpos[pit_adr + 3 : pit_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    for _ in range(300):  # 3 s
        mujoco.mj_step(model, data)

    pit_pos = data.xpos[model.body("pit").id].copy()
    ball_ids = [model.body(f"ball_{i}").id for i in range(n_balls)]
    local = data.xpos[ball_ids] - pit_pos  # pit stays upright during settling
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, positions=local)
    return local


def append_water_balls(spec: mujoco.MjSpec, n_balls: int, radius: float = BALL_RADIUS) -> None:
    """Append N water-ball bodies to the END of the spec's worldbody.

    Geom parameters mirror the `water_ball` defaults class in pit_defs.xml (set
    explicitly here because MjSpec-added geoms don't resolve XML defaults classes).
    Appending last guarantees ball DOFs sit at the tail of qpos/qvel.
    """
    for i in range(n_balls):
        body = spec.worldbody.add_body(name=f"water_ball_{i}", pos=[0, 0, -1.0 - 0.05 * i])
        body.add_freejoint(name=f"water_ball_{i}_joint")
        geom = body.add_geom(name=f"water_ball_{i}_geom")
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size[0] = radius
        geom.density = BALL_DENSITY
        geom.condim = 1
        geom.priority = 7
        geom.rgba = [0.25, 0.55, 0.95, 0.85]
        geom.group = 2


def apply_rigid_water_mass(spec: mujoco.MjSpec, n_balls: int, scale: float = 1.0) -> None:
    """Augment the planner pitcher's inertial with the water as a RIGID lump.

    This is the "knows the weight, not the sloshing" planner variant: total ball
    mass is added at the settled-water COM (half the fill height above the cavity
    floor), with the inertia of a solid water block of the fill dimensions.
    Deliberately NOT modeled: COM shift and slosh during motion.
    """
    if n_balls <= 0:
        return
    water_mass = n_balls * BALL_MASS
    fill_height = (n_balls / (BALLS_PER_FULL_CAVITY * scale**3)) * CAVITY_TOP * scale
    water_com_z = fill_height / 2.0

    pit_body = spec.body("pit")
    m0 = pit_body.mass
    com0 = np.array(pit_body.ipos)
    i0 = np.array(pit_body.inertia)  # diagonal, principal frame ~ body frame here

    # Water block inertia about its own COM (solid box 2*HX x 2*HY x fill_height).
    lx, ly, lz = 2 * CAVITY_HX * scale, 2 * CAVITY_HY * scale, fill_height
    iw = (water_mass / 12.0) * np.array([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
    com_w = np.array([0.0, 0.0, water_com_z])

    m = m0 + water_mass
    com = (m0 * com0 + water_mass * com_w) / m
    # Parallel-axis both inertias to the combined COM (diagonal terms only; offsets are ~z-only).
    d0 = com0 - com
    dw = com_w - com
    i_total = (
        i0
        + m0 * np.array([d0[1] ** 2 + d0[2] ** 2, d0[0] ** 2 + d0[2] ** 2, d0[0] ** 2 + d0[1] ** 2])
        + iw
        + water_mass * np.array([dw[1] ** 2 + dw[2] ** 2, dw[0] ** 2 + dw[2] ** 2, dw[0] ** 2 + dw[1] ** 2])
    )
    pit_body.mass = m
    pit_body.ipos = com
    pit_body.inertia = i_total


def project_state(
    qpos: np.ndarray, qvel: np.ndarray, nq_planner: int, nv_planner: int
) -> tuple[np.ndarray, np.ndarray]:
    """Plant state -> planner state: drop the trailing water-ball DOFs."""
    return np.array(qpos[:nq_planner]), np.array(qvel[:nv_planner])


# Empty-pitcher inertial baseline (must match the <inertial> in pit.xml).
EMPTY_PIT_MASS = 1.0
EMPTY_PIT_COM = np.array([0.0, 0.0, 0.15])
EMPTY_PIT_INERTIA = np.array([0.0181, 0.0152, 0.0067])


def water_in_pitcher(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_balls: int,
    radius: float = BALL_RADIUS,
    slack: float = 0.02,
    scale: float = 1.0,
) -> tuple[float, np.ndarray]:
    """(mass, COM in the pit body frame) of the water currently INSIDE the cavity.

    The feedback signal for the adaptive rigid-water planner: as water pours out
    or sloshes, both the retained mass and its COM (including x/y offsets) change.
    Returns (0.0, cavity-center) when no balls remain inside.
    """
    if n_balls == 0:
        return 0.0, np.array([0.0, 0.0, 0.05 * scale])
    pit_id = model.body("pit").id
    pit_pos = data.xpos[pit_id]
    pit_rot = data.xmat[pit_id].reshape(3, 3)
    ball_ids = [model.body(f"water_ball_{i}").id for i in range(n_balls)]
    local = (data.xpos[ball_ids] - pit_pos) @ pit_rot
    inside = (
        (np.abs(local[:, 0]) < CAVITY_HX * scale + slack)
        & (np.abs(local[:, 1]) < CAVITY_HY * scale + slack)
        & (local[:, 2] > -slack)
        & (local[:, 2] < CAVITY_TOP * scale + 6 * radius)
    )
    if not inside.any():
        return 0.0, np.array([0.0, 0.0, 0.05 * scale])
    return float(inside.sum() * ball_mass(radius)), local[inside].mean(axis=0)


def set_rigid_water_on_model(
    model: mujoco.MjModel, water_mass: float, water_com: np.ndarray, scale: float = 1.0
) -> None:
    """Write empty-pitcher + rigid-water inertial into a COMPILED planner model.

    Runtime counterpart of :func:`apply_rigid_water_mass` (which edits the MjSpec at
    build time). NOTE: the C++ rollout systems hold model COPIES, so after calling
    this the controller's rollout backend must be RECREATED for the change to reach
    the rollouts (verified empirically; recreation of 24 systems costs ~1.6 s).
    """
    empty_mass = EMPTY_PIT_MASS * scale**3
    empty_com = EMPTY_PIT_COM * scale
    empty_inertia = EMPTY_PIT_INERTIA * scale**5
    pit_id = model.body("pit").id
    if water_mass <= 0.0:
        model.body_mass[pit_id] = empty_mass
        model.body_ipos[pit_id] = empty_com
        model.body_inertia[pit_id] = empty_inertia
        return

    # Water block: footprint of the cavity, height from bulk-water volume.
    fill_height = max(water_mass / (1000.0 * 4 * CAVITY_HX * CAVITY_HY * scale**2), 0.01)
    lx, ly, lz = 2 * CAVITY_HX * scale, 2 * CAVITY_HY * scale, fill_height
    iw = (water_mass / 12.0) * np.array([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])

    m = empty_mass + water_mass
    com = (empty_mass * empty_com + water_mass * water_com) / m
    d0 = empty_com - com
    dw = np.asarray(water_com) - com
    inertia = (
        empty_inertia
        + empty_mass * np.array([d0[1] ** 2 + d0[2] ** 2, d0[0] ** 2 + d0[2] ** 2, d0[0] ** 2 + d0[1] ** 2])
        + iw
        + water_mass * np.array([dw[1] ** 2 + dw[2] ** 2, dw[0] ** 2 + dw[2] ** 2, dw[0] ** 2 + dw[1] ** 2])
    )
    model.body_mass[pit_id] = m
    model.body_ipos[pit_id] = com
    model.body_inertia[pit_id] = inertia


def spill_fraction(
    model: mujoco.MjModel, data: mujoco.MjData, n_balls: int, slack: float = 0.02, scale: float = 1.0
) -> float:
    """Fraction of water balls OUTSIDE the pitcher cavity, measured in the pit frame."""
    if n_balls == 0:
        return 0.0
    pit_id = model.body("pit").id
    pit_pos = data.xpos[pit_id]
    pit_rot = data.xmat[pit_id].reshape(3, 3)
    ball_ids = [model.body(f"water_ball_{i}").id for i in range(n_balls)]
    local = (data.xpos[ball_ids] - pit_pos) @ pit_rot
    inside = (
        (np.abs(local[:, 0]) < CAVITY_HX * scale + slack)
        & (np.abs(local[:, 1]) < CAVITY_HY * scale + slack)
        & (local[:, 2] > -slack)
        & (local[:, 2] < CAVITY_TOP * scale + 6 * BALL_RADIUS)
    )
    return float(1.0 - inside.mean())


def pitcher_tilt_deg(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Tilt of the pitcher away from upright, in degrees (0 = upright)."""
    pit_id = model.body("pit").id
    z_axis = data.xmat[pit_id].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))


def spout_angle_deg(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Angle between the SPOUT direction and straight-down, in degrees.

    The spout points along the pitcher's local -x axis. Convention:
    90 deg = pitcher upright (spout horizontal); 0 deg = spout pointing straight
    down (pitcher lying mouth-down). Pouring at "spout angle a" means holding the
    spout a degrees away from vertical -- the pour-attitude definition of the task.
    cos(angle) equals the world-z component of the pitcher's +x axis.
    """
    pit_id = model.body("pit").id
    x_axis = data.xmat[pit_id].reshape(3, 3)[:, 0]
    return float(np.degrees(np.arccos(np.clip(x_axis[2], -1.0, 1.0))))
