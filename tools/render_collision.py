# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Headless collision-geometry inspector for sumo tasks (SSH-friendly).

Judo's viser GUI hides any geom whose name contains "collision" (see
judo/visualizers/visualizer.py: geom_exclude_substring="collision"), and it has no
collision-group toggle. This script renders a task's object offscreen with the visual
mesh made translucent and the collision geoms tinted, so you can verify that collision
primitives line up with the visual mesh — then save PNGs you can open in VS Code.

Usage:
    MUJOCO_GL=egl pixi run python -m tools.render_collision spot_bucket_drag
    MUJOCO_GL=egl pixi run python -m tools.render_collision spot_bucket_drag --object-joint bucket_joint

Notes:
- Requires a GL backend: set MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU/software).
- Default offscreen framebuffer is 640x480; keep --size <= 480 unless you add
  <visual><global offwidth=... offheight=.../></visual> to the model.
"""

import argparse
import importlib
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import sumo.tasks  # noqa: F401 -- registers all tasks
from judo.tasks import get_registered_tasks

# Tint cycle for collision geoms so adjacent pieces are distinguishable.
_TINTS = [
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.6, 1.0, 1.0],
    [1.0, 0.6, 0.0, 1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", help="registered task name, e.g. spot_bucket_drag")
    ap.add_argument("--object-joint", default=None, help="free-joint name to center on (auto-detected if omitted)")
    ap.add_argument("--size", type=int, default=480, help="image size (<=480 for default framebuffer)")
    ap.add_argument("--distance", type=float, default=1.0, help="camera distance from the object")
    ap.add_argument("--out", default="out/collision_render", help="output directory")
    args = ap.parse_args()

    reg = get_registered_tasks()
    if args.task not in reg:
        raise SystemExit(f"Unknown task '{args.task}'. Registered: {sorted(reg)}")
    task = reg[args.task].task_type()
    m, d = task.model, task.data

    # Reset to the task's nominal pose, then recenter the object at the origin for framing.
    d.qpos[:] = task.reset_pose
    d.qvel[:] = 0
    obj_start = getattr(task, "object_pose_idx", getattr(task, "object_pose_start", None))
    if obj_start is not None:
        d.qpos[obj_start : obj_start + 3] = [0, 0, float(d.qpos[obj_start + 2])]
        d.qpos[obj_start + 3 : obj_start + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    look_z = float(d.qpos[obj_start + 2]) if obj_start is not None else 0.2

    # Make the visual mesh translucent and tint collision geoms (group 3 by convention).
    tint_i = 0
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_group[g] == 2:  # visual
            m.geom_rgba[g] = [0.85, 0.85, 0.85, 0.2]
        elif m.geom_group[g] == 3 and ("bucket" in name or "barrel" in name or "handle" in name or name == ""):
            m.geom_rgba[g] = _TINTS[tint_i % len(_TINTS)]
            tint_i += 1

    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1
    opt.geomgroup[3] = 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(m, args.size, args.size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, look_z]
    cam.distance = args.distance
    views = [("front", 90, -8), ("side", 0, -8), ("3q", 45, -12), ("top", 90, -89)]
    for tag, az, el in views:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam, opt)
        path = outdir / f"{args.task}_collision_{tag}.png"
        Image.fromarray(r.render()).save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
