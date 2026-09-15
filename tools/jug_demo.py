"""Reproducible, closed-loop SUMO jug demos; no forces or object pose edits."""

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from judo.app.structs import MujocoState
from judo.optimizers.cem import CrossEntropyMethod, CrossEntropyMethodConfig
from judo.simulation.hierarchical_mj_simulation import HierarchicalMJSimulation
from judo.tasks import get_registered_tasks
from judo.utils.hierarchical_mj_rollout_backend import HierarchicalMJRolloutBackend
from PIL import Image, ImageDraw, ImageFont

from sumo.controller import Controller, ControllerConfig
from sumo.run_mpc.run_mpc import _create_sim
from sumo.tasks.spot.jug_water import CAVITY_VOLUME
from sumo.tasks.spot.spot_jug_manipulation import (
    SpotJugManipulation,
    SpotJugManipulationConfig,
    disable_velocity_rewards,
)


def render(task, trajectory, metrics, filename, horizon):
    model = task.model
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    # Ground shares collision group 3, while visual robot meshes use group 2.
    # Retain the visible floor when hiding duplicate collision geometry.
    model.geom_group[model.geom("ground").id] = 0
    if task.water_count:
        # Visualization only: expose the physical particles through a translucent shell.
        model.mat_rgba[model.material("jug_material").id, 3] = 0.22
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.0, 0.0, 0.22]
    camera.distance = 3.4
    camera.azimuth = 125
    camera.elevation = -25
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, height=640, width=960)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    no_velocity = all(
        getattr(task.config, key, 0) == 0 for key in ("w_linear_velocity", "w_angular_velocity", "w_roll", "w_slip")
    )
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            "960x640",
            "-pix_fmt",
            "rgb24",
            "-r",
            "25",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(filename),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for idx in range(0, len(trajectory), 2):
            data.qpos[:] = trajectory[idx]
            mujoco.mj_forward(model, data)
            camera.lookat[:2] = (
                data.qpos[task.body_pose_start : task.body_pose_start + 2]
                + data.qpos[task.object_pose_start : task.object_pose_start + 2]
            ) / 2
            renderer.update_scene(data, camera, option)
            # Goal and A markers are visual only, added to the renderer scene.
            markers = [(task.config.goal_pos, [0.2, 0.95, 0.4, 0.35], "B / goal", task.config.position_tolerance)]
            if task.mode in ("roll", "move"):
                markers.append((task.config.start_pos, [1, 0.7, 0.1, 0.4], "A", 0.10))
            for position, color, label, radius in markers:
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    [radius, 0.002, 0],
                    [*position[:2], 0.006],
                    np.eye(3).ravel(),
                    color,
                )
                geom.label = label
                renderer.scene.ngeom += 1
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            m = metrics[idx]
            draw.rectangle((0, 0, 960, 101 if task.water_count else 76), fill=(20, 25, 35))
            draw.text((15, 8), f"{task.name} | CEM MPC H={horizon:g}s | t={m['time']:.2f}s", fill="white", font=font)
            draw.text(
                (15, 34),
                f"tilt={m['tilt_deg']:.1f}deg   goal error={m['goal_distance']:.3f}m   "
                f"speed={m['linear_speed']:.3f}m/s   roll={m['rolling_distance']:.3f}m",
                fill="white",
                font=font,
            )
            draw.text(
                (15, 55),
                ("SUCCESS" if m["success"] else "RUNNING") + (" | jug velocity rewards OFF" if no_velocity else ""),
                fill=(80, 255, 120) if m["success"] else (255, 210, 80),
                font=font,
            )
            if task.water_count:
                draw.text(
                    (15, 78),
                    f"water={task.config.water_fill_ratio:.0%} | {task.water_count} balls | "
                    f"{task.water_mass:.3f} kg | retained={m['water_retained_fraction']:.0%}",
                    fill=(90, 185, 255),
                    font=font,
                )
            process.stdin.write(np.asarray(frame).tobytes())
            if idx in (0, len(trajectory) - 1, len(trajectory) - 2):
                frame.save(filename.with_name(filename.stem + ("_start.png" if idx == 0 else "_end.png")))
    finally:
        process.stdin.close()
        renderer.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["spot_jug_upright", "spot_jug_lay_down", "spot_jug_roll", "spot_jug_move"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--rollouts", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--water-fill", type=float, default=0.0, help="Construction-time water volume fraction")
    parser.add_argument("--out", type=Path, default=Path("out/jug_tasks"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--replay", type=Path, help="Render an existing NPZ without rerunning physics")
    parser.add_argument("--set", action="append", default=[], help="Task config field=JSON value")
    parser.add_argument(
        "--no-velocity-reward",
        action="store_true",
        help="Zero all jug velocity rewards, including rolling and slip; keep success checks",
    )
    args = parser.parse_args()
    if args.replay:
        saved = np.load(args.replay, allow_pickle=False)
        result = json.loads(str(saved["result"]))
        if result["task"] != args.task:
            raise ValueError("Replay task does not match trajectory")
        config = get_registered_tasks()[args.task].task_config_type()
        for key, value in result["config"].items():
            setattr(config, key, np.asarray(value) if isinstance(value, list) else value)
        task_type = cast(type[SpotJugManipulation], get_registered_tasks()[args.task].task_type)
        task = task_type(config=config)
        args.out.mkdir(parents=True, exist_ok=True)
        stem = args.out / args.replay.stem
        render(
            task,
            saved["qpos"],
            json.loads(str(saved["metrics"])),
            stem.with_suffix(".mp4"),
            result["controller"]["horizon"],
        )
        if stem.with_suffix(".npz").resolve() != args.replay.resolve():
            shutil.copy2(args.replay, stem.with_suffix(".npz"))
        result["video"] = str(stem.with_suffix(".mp4").resolve())
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["video"])
        return
    np.random.seed(args.seed)
    registration = get_registered_tasks()[args.task]
    config = cast(SpotJugManipulationConfig, registration.task_config_type())
    config.water_fill_ratio = args.water_fill
    for value in args.set:
        key, raw = value.split("=", 1)
        if not hasattr(config, key):
            raise ValueError(key)
        parsed = json.loads(raw)
        setattr(config, key, np.asarray(parsed) if isinstance(parsed, list) else parsed)
    if args.no_velocity_reward:
        disable_velocity_rewards(config)
    task = cast(type[SpotJugManipulation], registration.task_type)(config=config)
    sim = cast(HierarchicalMJSimulation, _create_sim(args.task))
    sim.task = task
    assert registration.locomotion_policy_path is not None
    sim._init_cpp_systems(registration.locomotion_policy_path)
    optcfg = CrossEntropyMethodConfig(
        num_rollouts=args.rollouts,
        num_nodes=args.nodes,
        num_elites=4,
        sigma_min=0.12,
        sigma_max=1.0,
        use_noise_ramp=True,
        noise_ramp=2.0,
    )
    ctrlcfg = ControllerConfig(horizon=args.horizon, control_freq=25.0, max_opt_iters=1)
    controller = Controller(ctrlcfg, task, CrossEntropyMethod(optcfg, task.nu), rollout_backend="mujoco_hierarchical")
    if task.water_count:
        # Offline experiments must integrate the whole 2 s physical horizon, not
        # freeze its tail when the production 125 ms compute budget expires.
        backend = cast(HierarchicalMJRolloutBackend, controller.rollout_backend)
        native_rollout = backend._threaded_rollout

        def full_horizon_rollout(*native_args):
            return native_rollout(*native_args[:-1], float("inf"))

        backend._threaded_rollout = full_horizon_rollout
    sim.reset_policy_state()
    qpos, qvel, actions, metrics = [], [], [], []
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.task}_seed{args.seed}"
    started = time.monotonic()
    dwell = 0.0
    passed = False
    frozen_prediction_tails = 0
    action = task.optimizer_warm_start()
    for step in range(round(args.seconds / task.dt) + 1):
        task.post_sim_step()
        good = task.success(task.model, task.data)
        m = dict(time=float(task.data.time), success=good, **task.metrics(task.data))
        qpos.append(task.data.qpos.copy())
        qvel.append(task.data.qvel.copy())
        actions.append(action.copy())
        metrics.append(m)
        dwell = dwell + task.dt if good else 0.0
        if step % 50 == 0:
            print(json.dumps(m), flush=True)
            with stem.with_suffix(".progress.jsonl").open("a") as stream:
                stream.write(json.dumps(m) + "\n")
        if dwell >= 0.75:
            passed = True
            break
        if task.failure(task.model, task.data):
            break
        if step % 2 == 0:
            controller.update_states(
                MujocoState(
                    task.data.time,
                    task.data.qpos.copy(),
                    task.data.qvel.copy(),
                    task.data.mocap_pos.copy(),
                    task.data.mocap_quat.copy(),
                    task.get_sim_metadata(),
                )
            )
            for _ in range(args.iterations):
                # Each candidate starts with the real low-level recurrent state.
                controller._last_policy_output = np.tile(sim.last_policy_output, (args.rollouts, 1))
                controller.update_action()
                frozen_prediction_tails += int(
                    np.all(controller.states[:, -1] == controller.states[:, -2], axis=-1).sum()
                )
        action = controller.action(task.data.time)
        sim.step(action)
    config = asdict(task.config)
    config = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in config.items()}
    result = {
        "task": args.task,
        "seed": args.seed,
        "success": passed,
        "success_dwell_seconds": dwell,
        "wall_seconds": time.monotonic() - started,
        "final": metrics[-1],
        "config": config,
        "controller": asdict(ctrlcfg),
        "optimizer": asdict(optcfg),
        "iterations": args.iterations,
        "frozen_prediction_tails": frozen_prediction_tails,
        "no_velocity_reward": args.no_velocity_reward,
        "water": {
            "count": task.water_count,
            "radius_m": task.water_radius,
            "mass_kg": task.water_mass,
            "cavity_litres": CAVITY_VOLUME * 1000,
            "min_retained_fraction": min(m.get("water_retained_fraction", 1.0) for m in metrics),
            "sealed": True,
            "planner": "same_particles_as_plant",
            "compute_cutoff_disabled": bool(task.water_count),
        },
        "video": str(stem.with_suffix(".mp4").resolve()) if args.render else None,
    }
    np.savez_compressed(
        stem.with_suffix(".npz"),
        qpos=qpos,
        qvel=qvel,
        actions=actions,
        metrics=json.dumps(metrics),
        result=json.dumps(result),
    )
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if args.render:
        render(task, qpos, metrics, stem.with_suffix(".mp4"), ctrlcfg.horizon)


if __name__ == "__main__":
    main()
