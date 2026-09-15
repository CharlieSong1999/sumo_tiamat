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


def water_parameters(fill_ratio, radius):
    if not 0 <= fill_ratio <= 0.25:
        raise ValueError("Jug water fill must be in [0, 0.25]; higher fills are not calibrated")
    if not 0.01 <= radius <= 0.03:
        raise ValueError("Water ball radius must be in [0.01, 0.03] metres")
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
    xy = np.arange(-PROFILE[0][1] + radius + 0.004, PROFILE[0][1] - radius, spacing)
    candidates = np.array(
        [
            [x, y, z]
            for z in np.arange(PROFILE[0][0] + radius + 0.004, PROFILE[1][0] - radius, spacing)
            for x in xy
            for y in xy
            if np.hypot(x, y) < PROFILE[0][1] - radius - 0.003
        ]
    )
    if len(candidates) < count:
        raise ValueError("Requested particles exceed the validated initialization grid")
    order = np.argsort(-(candidates @ spec.option.gravity), kind="stable")
    data.qpos.reshape(-1, 7)[:, :3] = candidates[order[:count]]
    data.qpos.reshape(-1, 7)[:, 3:] = [1, 0, 0, 0]
    for _ in range(1500):
        mujoco.mj_step(model, data)
    return data.qpos.reshape(-1, 7)[:, :3].copy()


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
