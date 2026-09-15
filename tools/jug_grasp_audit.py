"""Read-only contact audit of a neck-grasp NPZ; no dynamics or object pose edits in the saved run.

Reconstructs geometric contacts, not contact forces or a proof of force closure.
The saved high-level command is used for the closing-resistance proxy.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from sumo.tasks.spot.jug_neck_grasp import JAW_GEOMS, grasp_metrics
from sumo.tasks.spot.spot_jug_manipulation import SpotJugUpright, SpotJugUprightConfig


def audit(path):
    with np.load(path, allow_pickle=False) as saved:
        result = json.loads(str(saved["result"]))
        if result["task"] != "spot_jug_upright" or not result["config"].get("neck_grasp"):
            raise ValueError("Expected a neck_grasp upright trajectory")
        config = SpotJugUprightConfig()
        for key, value in result["config"].items():
            setattr(config, key, np.asarray(value) if isinstance(value, list) else value)
        task = SpotJugUpright(config)
        model = task.model
        model.geom_condim[model.geom("jug_collision").id] = result["jug_contact_dim"]
        data = mujoco.MjData(model)
        neck = model.geom("jug_neck_collision").id
        moving = {model.geom(name).id for name in JAW_GEOMS[:2]}
        fixed = {model.geom(name).id for name in JAW_GEOMS[2:]}
        frames = []
        best_dot = None
        for pose, velocity, action, metric in zip(
            saved["qpos"], saved["qvel"], saved["actions"], json.loads(str(saved["metrics"])), strict=True
        ):
            data.qpos[:] = pose
            data.qvel[:] = velocity
            mujoco.mj_forward(model, data)
            proxy = grasp_metrics(task, data, action)["neck_grasp_detected"]
            if not proxy or metric["tilt_deg"] < 15:
                continue
            normals = [[], []]
            for contact in data.contact:
                if contact.dist > 0 or neck not in (contact.geom1, contact.geom2):
                    continue
                jaw = contact.geom2 if contact.geom1 == neck else contact.geom1
                # All normals point from the jaw towards the neck.
                normal = contact.frame[:3] * (-1 if contact.geom1 == neck else 1)
                if jaw in moving:
                    normals[0].append(normal)
                elif jaw in fixed:
                    normals[1].append(normal)
            dots = [float(np.dot(a, b)) for a in normals[0] for b in normals[1]]
            if dots and min(dots) < -0.5:
                frames.append(metric["time"])
                best_dot = min(dots) if best_dot is None else min(best_dot, *dots)
        longest = current = 0
        previous = None
        for stamp in frames:
            current = current + 1 if previous is not None and np.isclose(stamp - previous, task.dt) else 1
            longest = max(longest, current)
            previous = stamp
        return {
            "trajectory": str(Path(path).resolve()),
            "sample_dt": task.dt,
            "proxy_seconds_all_orientations": result["neck_grasp_seconds"],
            "opposed_touching_frames_before_upright": len(frames),
            "sampled_opposed_touching_seconds_before_upright": len(frames) * task.dt,
            "longest_sampled_opposed_touching_seconds": longest * task.dt,
            "first_opposed_touching_time": frames[0] if frames else None,
            "minimum_normal_dot": best_dot,
            "qualifier": "Geometric contacts plus closing resistance; not a force-closure or sustained lift proof",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    print(json.dumps(audit(parser.parse_args().trajectory), indent=2))
