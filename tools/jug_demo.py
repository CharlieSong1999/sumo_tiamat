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
from sumo.tasks.spot import jug_neck_grasp
from sumo.tasks.spot.jug_water import CAVITY_VOLUME
from sumo.tasks.spot.spot_jug_manipulation import (
    SpotJugManipulation,
    SpotJugManipulationConfig,
    disable_velocity_rewards,
)


class PaperVarianceCEM(CrossEntropyMethod):
    """Literal IV-A noise schedule: fixed variance .02 -> .6 over the knots.

    Elite means are updated as in CEM. Sampling resets covariance every time,
    rather than using Judo's fitted-sigma multiplier ramp. The paper does not
    specify how its schedule interacts with fitted covariance; this is an
    explicit interpretation, not an author-verified implementation.
    """

    def sample_control_knots(self, nominal_knots):
        sigma = np.sqrt(np.linspace(0.02, 0.6, self.num_nodes))[:, None]
        noise = np.random.randn(self.num_rollouts - 1, self.num_nodes, self.nu)
        return np.concatenate([nominal_knots[None], nominal_knots + sigma[None] * noise])


def planning_due(step, dt, frequency):
    """Quantize planning deadlines upward onto the unchanged low-level clock.

    At 50 Hz low-level / 20 Hz planning this alternates .06/.04 s intervals.
    """
    return step == 0 or int(step * dt * frequency + 1e-9) > int((step - 1) * dt * frequency + 1e-9)


def render(task, trajectory, metrics, filename, horizon, planning_label="", view="overview"):
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
    camera.distance = 1.05 if view == "neck" else 3.4
    camera.azimuth = 90 if view == "neck" else 125
    camera.elevation = -15 if view == "neck" else -25
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
            if view == "neck":
                jug = data.body("jug")
                camera.lookat[:] = jug.xpos + jug.xmat.reshape(3, 3) @ jug_neck_grasp.TARGET
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
            draw.text(
                (15, 8),
                f"{task.name} | MPC H={horizon:g}s | {planning_label} | t={m['time']:.2f}s",
                fill="white",
                font=font,
            )
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
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument(
        "--jug-contact-dim",
        type=int,
        choices=[1, 3, 4, 6],
        help="Experiment-only contact override; 3 reproduces the original water demos",
    )
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument(
        "--control-freq", type=float, default=25.0, help="Target MPC update rate on the 50 Hz plant clock"
    )
    parser.add_argument("--noise-profile", choices=["adaptive", "paper-variance"], default="adaptive")
    parser.add_argument("--water-fill", type=float, default=0.0, help="Construction-time water volume fraction")
    parser.add_argument("--out", type=Path, default=Path("out/jug_tasks"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--view", choices=["overview", "neck"], default="overview", help="Rendering camera only")
    parser.add_argument("--replay", type=Path, help="Render an existing NPZ without rerunning physics")
    parser.add_argument("--set", action="append", default=[], help="Task config field=JSON value")
    parser.add_argument(
        "--no-velocity-reward",
        action="store_true",
        help="Zero all jug velocity rewards, including rolling and slip; keep success checks",
    )
    args = parser.parse_args()
    if args.iterations < 1 or not 1 <= args.elites <= args.rollouts:
        parser.error("Require iterations >= 1 and 1 <= elites <= rollouts")
    if not np.isfinite(args.control_freq) or not 0 < args.control_freq <= 50:
        parser.error("Require finite control-freq in (0, 50] Hz")
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
        if "jug_contact_dim" in result:
            task.model.geom_condim[task.model.geom("jug_collision").id] = result["jug_contact_dim"]
        args.out.mkdir(parents=True, exist_ok=True)
        stem = args.out / args.replay.stem
        render(
            task,
            saved["qpos"],
            json.loads(str(saved["metrics"])),
            stem.with_suffix(".mp4"),
            result["controller"]["horizon"],
            f"{result['optimizer']['num_rollouts']}x{result['iterations']} CEM "
            f"@{result['controller']['control_freq']:g}Hz",
            args.view,
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
    if args.jug_contact_dim is not None:
        task.model.geom_condim[task.model.geom("jug_collision").id] = args.jug_contact_dim
    sim = cast(HierarchicalMJSimulation, _create_sim(args.task))
    sim.task = task
    assert registration.locomotion_policy_path is not None
    sim._init_cpp_systems(registration.locomotion_policy_path)
    optcfg = CrossEntropyMethodConfig(
        num_rollouts=args.rollouts,
        num_nodes=args.nodes,
        num_elites=args.elites,
        sigma_min=float(np.sqrt(0.02)) if args.noise_profile == "paper-variance" else 0.12,
        sigma_max=float(np.sqrt(0.6)) if args.noise_profile == "paper-variance" else 1.0,
        use_noise_ramp=args.noise_profile == "adaptive",
        noise_ramp=2.0,
    )
    ctrlcfg = ControllerConfig(horizon=args.horizon, control_freq=args.control_freq, max_opt_iters=1)
    optimizer_type = PaperVarianceCEM if args.noise_profile == "paper-variance" else CrossEntropyMethod
    controller = Controller(ctrlcfg, task, optimizer_type(optcfg, task.nu), rollout_backend="mujoco_hierarchical")
    if task.water_count:
        # Offline experiments must integrate the whole requested physical horizon, not
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
    planning_times = []
    planning_sim_times = []
    iteration_times = []
    action = task.optimizer_warm_start()
    for step in range(round(args.seconds / task.dt) + 1):
        task.post_sim_step()
        good = task.success(task.model, task.data)
        m = dict(time=float(task.data.time), success=good, **task.metrics(task.data))
        if task.neck_grasp:
            m.update(jug_neck_grasp.grasp_metrics(task, task.data, action))
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
        if planning_due(step, task.dt, args.control_freq):
            planning_sim_times.append(float(task.data.time))
            planning_started = time.perf_counter()
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
                iteration_started = time.perf_counter()
                controller.update_action()
                iteration_times.append(time.perf_counter() - iteration_started)
                frozen_prediction_tails += int(
                    np.all(controller.states[:, -1] == controller.states[:, -2], axis=-1).sum()
                )
            planning_times.append(time.perf_counter() - planning_started)
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
        "noise": {
            "profile": args.noise_profile,
            "variance_at_knots": np.linspace(0.02, 0.6, args.nodes).tolist()
            if args.noise_profile == "paper-variance"
            else None,
            "covariance_update": "fixed_each_sample" if args.noise_profile == "paper-variance" else "elite_fitted",
        },
        "jug_contact_dim": int(task.model.geom_condim[task.model.geom("jug_collision").id]),
        "planning": {
            "updates": len(planning_times),
            "candidates_per_update": args.rollouts * args.iterations,
            "physics_steps_per_candidate": int(controller.num_timesteps * task.physics_substeps),
            "mean_ms": float(np.mean(planning_times) * 1000) if planning_times else None,
            "p50_ms": float(np.percentile(planning_times, 50) * 1000) if planning_times else None,
            "p95_ms": float(np.percentile(planning_times, 95) * 1000) if planning_times else None,
            "max_ms": float(np.max(planning_times) * 1000) if planning_times else None,
            "fraction_within_40ms": float(np.mean(np.asarray(planning_times) <= 0.04)) if planning_times else None,
            "target_period_ms": 1000 / args.control_freq,
            "fraction_within_target_period": float(np.mean(np.asarray(planning_times) <= 1 / args.control_freq))
            if planning_times
            else None,
            "plant_tick_ms": task.dt * 1000,
            "actual_intervals_ms": sorted(set(np.round(np.diff(planning_sim_times) * 1000, 6).tolist())),
        },
        "frozen_prediction_tails": frozen_prediction_tails,
        "neck_grasp_seconds": sum(bool(m.get("neck_grasp_detected", False)) for m in metrics) * task.dt,
        "neck_bilateral_seconds": sum(bool(m.get("neck_bilateral_contact", False)) for m in metrics) * task.dt,
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
        planning_times=planning_times,
        planning_sim_times=planning_sim_times,
        iteration_times=iteration_times,
        metrics=json.dumps(metrics),
        result=json.dumps(result),
    )
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if args.render:
        render(
            task,
            qpos,
            metrics,
            stem.with_suffix(".mp4"),
            ctrlcfg.horizon,
            f"{args.rollouts}x{args.iterations} CEM @{args.control_freq:g}Hz",
            args.view,
        )


if __name__ == "__main__":
    main()
