"""Coarse water particles in a sealed, hollow proxy of the tracking jug.

The exterior tracking mesh remains unchanged. Interior segments collide only
with particles, because the original exterior collision mesh is a solid hull.
This is a mass/slosh approximation, not a fluid solver.
"""

from functools import lru_cache

import mujoco
import numpy as np

# Approximate inner profile measured against jug_meshy_20k.obj, in metres.
PROFILE = ((-0.224, 0.127), (0.125, 0.127), (0.175, 0.022), (0.232, 0.022))
SEGMENTS = 16
# Profile radii are polygon apothems. Frustum integration gives cavity volume.
CAVITY_VOLUME = sum(
    SEGMENTS * np.tan(np.pi / SEGMENTS) * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3
    for (z0, r0), (z1, r1) in zip(PROFILE[:-1], PROFILE[1:], strict=True)
)


# Largest ball that still sits inside the 0.127 m cylindrical part of the cavity with
# margin (2026-09-17: 0.045 -> 3 balls, 0.06 -> 1 ball at 10 % fill).
MAX_BALL_RADIUS = 0.06
# From this many balls up the contact island is cheaper under CG/sparse than Newton.
# Measured on the deployed planner (32 x 1.5 s arm-roll profile, robot pushing the jug,
# Xeon w5-3435X, ms/plan p50): Newton 3 balls 32-44, 7: 42, 9: 52, 11: 57 (CG: 57);
# 19 balls Newton 134 (Ryzen) vs CG(50) 75-94. Newton wins or ties up to 11, CG wins at
# 19; the crossover lies in 12..18, untested. Budget-wise (50 ms p95) the workstation
# holds 7 balls under Newton; see auto_sumo/data/20260917_ball_sweep/.
CG_MIN_BALLS = 12


def water_parameters(fill_ratio, radius):
    if not 0 <= fill_ratio <= 0.25:
        raise ValueError("Jug water fill must be in [0, 0.25]; higher fills are not calibrated")
    if not 0.01 <= radius <= MAX_BALL_RADIUS:
        raise ValueError(f"Water ball radius must be in [0.01, {MAX_BALL_RADIUS}] metres")
    mass = 1000 * CAVITY_VOLUME * fill_ratio
    count = max(1, round(mass / (1560 * 4 * np.pi * radius**3 / 3))) if fill_ratio else 0
    return count, mass


def add_inner_shell(spec, body):
    """Closed polygonal cavity; independent collision bit 2, no extra jug mass."""
    for layer, ((z0, r0), (z1, r1)) in enumerate(zip(PROFILE[:-1], PROFILE[1:], strict=True)):
        for i in range(SEGMENTS):
            angles = (2 * np.pi * i / SEGMENTS, 2 * np.pi * (i + 1) / SEGMENTS)
            vertices = [
                [r / np.cos(np.pi / SEGMENTS) * np.cos(a), r / np.cos(np.pi / SEGMENTS) * np.sin(a), z]
                for z, inner in ((z0, r0), (z1, r1))
                for r in (inner, inner + 0.008)
                for a in angles
            ]
            name = f"jug_inner_{layer}_{i}"
            spec.add_mesh(name=name, uservert=np.asarray(vertices).ravel())
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                contype=2,
                conaffinity=2,
                group=3,
                mass=0,
                condim=1,
                priority=7,
            )
    for name, z, r in (
        ("floor", PROFILE[0][0] - 0.004, PROFILE[0][1]),
        ("cap", PROFILE[-1][0] + 0.004, PROFILE[-1][1]),
    ):
        body.add_geom(
            name=f"jug_inner_{name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0, z],
            size=[r + 0.012, 0.004, 0],
            contype=2,
            conaffinity=2,
            group=3,
            mass=0,
            condim=1,
            priority=7,
        )


def add_balls(spec, count, radius, total_mass):
    for i in range(count):
        body = spec.worldbody.add_body(name=f"jug_water_{i}")
        body.add_freejoint(name=f"jug_water_{i}_joint")
        body.add_geom(
            name=f"jug_water_{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            mass=total_mass / count,
            contype=2,
            conaffinity=2,
            condim=1,
            priority=7,
            rgba=[0.05, 0.45, 1.0, 1.0],
            group=2,
        )


def configure_water_solver(spec):
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    spec.option.iterations = 50


@lru_cache(maxsize=16)
def settled_positions(count, radius, mass, quat):
    """Settle water with a fixed jug, before episode start; deterministic/resettable."""
    spec = mujoco.MjSpec()
    configure_water_solver(spec)
    spec.option.timestep = 0.002
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat))
    rotation = rotation.reshape(3, 3)
    spec.option.gravity = rotation.T @ [0, 0, -9.81]
    add_inner_shell(spec, spec.worldbody.add_body(name="container"))
    add_balls(spec, count, radius, mass)
    model = spec.compile()
    data = mujoco.MjData(model)
    spacing = 2 * radius + 0.002

    def grid(xy):
        return np.array(
            [
                [x, y, z]
                for z in np.arange(PROFILE[0][0] + radius + 0.004, PROFILE[1][0] - radius, spacing)
                for x in xy
                for y in xy
                if np.hypot(x, y) < PROFILE[0][1] - radius - 0.003
            ]
        )

    # The original grid first: it is what every pinned reference run started from.
    candidates = grid(np.arange(-PROFILE[0][1] + radius + 0.004, PROFILE[0][1] - radius, spacing))
    if len(candidates) < count:
        # Big balls (r >= 0.045) can miss it entirely; a grid symmetric about the axis
        # always has the (0, 0) slot.
        half = np.arange(0.0, PROFILE[0][1] - radius, spacing)
        candidates = grid(np.unique(np.concatenate([-half, half])))
    if len(candidates) < count:
        raise ValueError("Requested particles exceed the validated initialization grid")
    order = np.argsort(-(candidates @ spec.option.gravity), kind="stable")
    data.qpos.reshape(-1, 7)[:, :3] = candidates[order[:count]]
    data.qpos.reshape(-1, 7)[:, 3:] = [1, 0, 0, 0]
    for _ in range(1500):
        mujoco.mj_step(model, data)
    return data.qpos.reshape(-1, 7)[:, :3].copy()


def pooled_local_positions(count, radius, quat_wxyz):
    """Where settled water sits for a jug at this orientation, in the jug's local frame.

    Analytic stand-in for `settled_positions` when the jug pose comes from perception at
    20 Hz and a physics settle per tick is not affordable: the balls line up on the
    cavity wall at the lowest point under gravity, in a row perpendicular to gravity.
    Lying jug: a row along the axis on the bottom wall. Upright jug: a row across the
    bottom. Good enough to put the mass where it is; the rollouts slosh it from there.
    """
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat_wxyz, dtype=float))
    rot = rot.reshape(3, 3)
    u = rot.T @ np.array([0.0, 0.0, -1.0])            # gravity in the jug frame
    u = u / max(np.linalg.norm(u), 1e-9)
    (z_bot, wall), (z_top, _), *_ = PROFILE            # cylindrical part of the cavity
    z_mid = 0.5 * (z_bot + z_top)
    radial = np.hypot(u[0], u[1])
    t = wall / radial if radial > 1e-6 else np.inf     # distance to the side wall along u
    if u[2] < -1e-6:
        t = min(t, (z_mid - z_bot) / -u[2])            # ... or to the bottom
    elif u[2] > 1e-6:
        t = min(t, (z_top - z_mid) / u[2])             # ... or to the top
    center = np.array([0.0, 0.0, z_mid]) + u * (t - radius)
    d = np.array([0.0, 0.0, 1.0]) - u[2] * u          # row direction: perpendicular to gravity
    if np.linalg.norm(d) < 1e-6:                       # upright: any radial direction
        d = np.array([1.0, 0.0, 0.0])
    d = d / np.linalg.norm(d)
    spacing = 2.0 * radius
    # Few, big balls: a placement that keeps them APART by construction, because the
    # planner re-synthesizes this every tick and an overlap is a 1 m/s transient in every
    # rollout (codex review 2026-09-17). This <=3-ball rule is what packed_local_positions
    # (5/7 balls) delegates to and preserves; do not change one without the other.
    # Tilted or lying: a line along the wall's generatrix (straight, so spacing survives),
    # piled against the lower end cap. Near upright: a flat cluster on the bottom.
    if count <= 3 and (count - 1) * spacing + 2 * radius <= z_top - z_bot:
        if radial > 0.5:                                    # more than ~30 deg from upright
            rdir = np.array([u[0], u[1], 0.0]) / radial * (wall - radius)
            if u[2] < -0.05:                                # bottom end is lower: pile there
                z0 = z_bot + radius
            elif u[2] > 0.05:                               # top end is lower
                z0 = z_top - radius - (count - 1) * spacing
            else:                                           # level: centred on the wall
                z0 = z_mid - (count - 1) * spacing / 2.0
            return np.array([[rdir[0], rdir[1], z0 + k * spacing] for k in range(count)])
        z_floor = z_bot + radius if u[2] <= 0 else z_top - radius
        if count == 1:
            xy = np.zeros((1, 2))
        elif count == 2:
            xy = np.array([[-radius, 0.0], [radius, 0.0]])
        else:                                               # equilateral triangle, side 2r
            circ = spacing / np.sqrt(3.0)
            xy = circ * np.array([[np.cos(a), np.sin(a)] for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3)])
        return np.column_stack([xy, np.full(len(xy), z_floor)])
    offsets = (np.arange(count) - (count - 1) / 2.0) * spacing
    pts = center[None, :] + offsets[:, None] * d[None, :]
    # Clamp into the cavity (side wall and both ends): at intermediate tilts, or with many
    # balls, a straight row leaves it (codex review 2026-09-15: 7.8 cm through the wall at
    # 45 deg). A clamped ball may overlap a neighbour; MuJoCo separates those in the first
    # steps, a ball outside the shell it never can.
    rad = np.hypot(pts[:, 0], pts[:, 1])
    over = rad > wall - radius
    pts[over, :2] *= ((wall - radius) / rad[over])[:, None]
    pts[:, 2] = np.clip(pts[:, 2], z_bot + radius, z_top - radius)
    return pts


# Radii of the validated coarse-water study (tools/jug_roll_water_reconstruction.py).
PACKED_RADII = {3: 0.045, 5: 0.04, 7: 0.035}


def packed_local_positions(count, radius, quat):
    """Nonoverlapping coarse proxy, NOT a static equilibrium or a fluid estimator.

    Authoritative copy of the placement developed in tools/jug_roll_water_reconstruction.py
    (2026-09-17); that tool imports this. Preserve the deployed <=3-ball rule exactly.
    For 5/7, select low gravitational potential sites from a translated seven-column
    hexagonal lattice. All columns are separated by >=2r and inside an inscribed
    cylinder; all axial levels are separated by >=2r. Selecting a subset cannot
    introduce overlaps. Ties minimize horizontal centre-of-mass bias. No target/robot/
    seed enters the placement.
    """
    import itertools

    quat = np.asarray(quat, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("Invalid jug quaternion")
    quat = quat / np.linalg.norm(quat)
    if count <= 3:
        return pooled_local_positions(count, radius, quat)
    if count not in (5, 7) or not np.isclose(radius, PACKED_RADII[count], rtol=0, atol=1e-12):
        raise ValueError("Packing is validated only for the 3/5/7-ball study")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    down = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    radial = np.linalg.norm(down[:2])
    axis = down[:2] / radial if radial > 1e-10 else np.array([1.0, 0.0])
    lateral = np.array([-axis[1], axis[0]])
    (bottom, wall), (top, _), *_ = PROFILE
    margin = 0.0005
    spacing = 2 * radius + 0.001
    room = wall - radius - margin
    if spacing > room:
        raise ValueError("Hexagonal lattice does not fit")
    angles = np.arange(6) * np.pi / 3
    ring = spacing * (np.cos(angles)[:, None] * axis + np.sin(angles)[:, None] * lateral)
    xy = np.vstack([np.zeros(2), ring])
    xy += (room - spacing) * axis * min(radial / 0.5, 1.0)
    levels = int(np.floor((top - bottom - 2 * radius - 2 * margin) / spacing)) + 1
    z = (np.arange(levels) - (levels - 1) / 2) * spacing + (bottom + top) / 2
    if down[2] < -0.05:
        z += bottom + radius + margin - z.min()
    elif down[2] > 0.05:
        z += top - radius - margin - z.max()
    candidates = np.array([[*p, zz] for p in xy for zz in z])
    potential = -(candidates @ down)
    order = np.argsort(potential, kind="stable")
    cutoff = potential[order[count - 1]]
    selected = list(np.flatnonzero(potential < cutoff - 1e-10))
    tied = np.flatnonzero(np.abs(potential - cutoff) <= 1e-10)
    needed = count - len(selected)
    centre = np.array([0.0, 0.0, (bottom + top) / 2])

    def tie_score(indices):
        points = candidates[selected + list(indices)]
        com = points.mean(0) - centre
        horizontal = com - np.dot(com, down) * down
        return float(np.dot(horizontal, horizontal)), float(np.square(points - centre).sum())

    selected.extend(min(itertools.combinations(tied.tolist(), needed), key=tie_score))
    points = candidates[selected]
    assert np.all(np.linalg.norm(points[:, :2], axis=1) + radius <= wall + 1e-10)
    assert np.all(points[:, 2] - radius >= bottom - 1e-10)
    assert np.all(points[:, 2] + radius <= top + 1e-10)
    return points


def ball_local_positions(model, data, count):
    jug = model.body("jug").id
    ids = [model.body(f"jug_water_{i}").id for i in range(count)]
    return (data.xpos[ids] - data.xpos[jug]) @ data.xmat[jug].reshape(3, 3)


def contained_mask(positions):
    """Centre containment with 5 mm contact tolerance, not a spill reward."""
    z = positions[:, 2]
    radius = np.interp(z, [p[0] for p in PROFILE], [p[1] for p in PROFILE])
    return (
        (z >= PROFILE[0][0] - 0.005)
        & (z <= PROFILE[-1][0] + 0.005)
        & (np.linalg.norm(positions[:, :2], axis=1) <= radius / np.cos(np.pi / SEGMENTS) + 0.005)
    )
