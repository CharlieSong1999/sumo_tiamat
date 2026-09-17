"""Fixed-state v4 roll: passive pause diagnostics and target generalization tests."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from judo.app.structs import MujocoState
from judo.optimizers.cem import CrossEntropyMethod, CrossEntropyMethodConfig

from sumo.controller import Controller, ControllerConfig
from sumo.tasks.spot.spot_jug_manipulation import SpotJugRoll, SpotJugRollConfig
from tools.jug_budget_study import OfflineBackend, OfflinePlant, digest, jsonable
from tools.jug_demo import planning_due, render
from tools.jug_roll_arm_experiment import contact_groups, motion_sample, planning_profile, speed_summary, touching

REFERENCE = Path("out/jug_roll_24x1_20260917/guided_v4")
SCENARIOS = {
    "front": [[2.15, 0, 0.14]],
    "left": [[0.95, 1.2, 0.14]],
    "right": [[0.95, -1.2, 0.14]],
    "behind": [[-0.25, 0, 0.14]],
    "behind_far": [[-0.75, 0, 0.14]],
    "sequence_corner": [[2.15, 0, 0.14], [2.15, 1.2, 0.14]],
    "sequence_forward": [[2.15, 0, 0.14], [3.35, 0, 0.14]],
}


# Fields the task reads at construction (model/sensors); a post-construction set would only LOOK applied.
CONSTRUCTION_TIME_FIELDS = frozenset({"water_fill_ratio", "water_ball_radius", "jug_mass", "rolling_friction",
                                      "roll_use_arm", "neck_grasp", "ground_friction",
                                      "w_hand_speed", "w_hand_peak_speed", "w_hand_reach"})


def make_system(num_rollouts=24, horizon=2.0, hand_reach_weight=None, hand_reach_halfwidth=0.0, task_name=None, water_ball_radius=None,
                ground_friction=None, reward_sets=None):
    if num_rollouts is not None and (not isinstance(num_rollouts, int) or num_rollouts < 3):
        raise ValueError("At least three paths are required for the frozen three-elite CEM")
    if horizon is not None and (not np.isfinite(horizon) or horizon <= 0 or not np.isclose(horizon / 0.02, round(horizon / 0.02))):
        raise ValueError("Horizon must be positive and an integer number of 20 ms steps")
    if hand_reach_weight is not None and (not np.isfinite(hand_reach_weight) or hand_reach_weight < 0):
        raise ValueError("Hand reach weight must be finite and nonnegative")
    if not np.isfinite(hand_reach_halfwidth) or hand_reach_halfwidth < 0:
        raise ValueError("Hand reach halfwidth must be finite and nonnegative")
    reference = json.loads((REFERENCE / "gentle_seed0.json").read_text())
    config = dict(reference["config"])
    if hand_reach_weight is not None:
        config["w_hand_reach"] = float(hand_reach_weight)
    if task_name and (hand_reach_weight is not None or hand_reach_halfwidth):
        raise ValueError("--task runs a registered profile as-is; reward overrides do not apply to it")
    if task_name:
        # A REGISTERED deployment profile (its defaults, as the planner builds it), with
        # the reference run's A/B geometry. Its state layout may differ from the
        # reference (water balls), so the initial-state pins below do not apply.
        from judo.tasks import get_task_registration

        import sumo.tasks  # noqa: F401

        registration = get_task_registration(task_name)
        # Studies on the registered profile with ONE construction-time field changed: the ball
        # size (10 % fill fixed, so the count follows; see jug_water.water_parameters) or the
        # floor's sliding friction (feet only; the jug keeps its own).
        overrides = {k: v for k, v in (("water_ball_radius", water_ball_radius), ("ground_friction", ground_friction))
                     if v is not None}
        task = registration.task_type(registration.task_config_type(**overrides)) if overrides else registration.task_type()
        for key, value in (reward_sets or {}).items():
            if key in CONSTRUCTION_TIME_FIELDS:
                raise ValueError(f"--set {key}: construction-time field; use the dedicated option or a registered profile")
            if not hasattr(task.config, key):
                raise ValueError(f"--set {key}: {task_name} has no such config field")
            setattr(task.config, key, type(getattr(task.config, key))(value))
        for key in ("start_pos", "goal_pos"):
            setattr(task.config, key, np.asarray(config[key], dtype=float))
    elif hand_reach_halfwidth:
        from tools.jug_roll_region_task import RegionRoll, RegionRollConfig

        task = RegionRoll(RegionRollConfig(**config, hand_reach_halfwidth=hand_reach_halfwidth))
    else:
        task = SpotJugRoll(SpotJugRollConfig(**config))
    settings = planning_profile("hardware_roll")
    assert settings == reference["planning_settings"], "Deployment profile changed; do not mix protocols"
    if task_name:
        # The registered profile's MPC shape (what tiamat's planner asserts at startup),
        # unless the caller overrides rollouts/horizon explicitly.
        from sumo.controller.optimizer_overrides import set_default_spot_optimizer_overrides
        from sumo.controller.overrides import set_default_spot_overrides

        set_default_spot_overrides()
        set_default_spot_optimizer_overrides()
        cem_cfg, ctrl_cfg = CrossEntropyMethodConfig(), ControllerConfig()
        cem_cfg.set_override(task_name)
        ctrl_cfg.set_override(task_name)
        num_rollouts = num_rollouts if num_rollouts is not None else cem_cfg.num_rollouts
        horizon = horizon if horizon is not None else ctrl_cfg.horizon
        cem_cfg.num_rollouts, ctrl_cfg.horizon = num_rollouts, horizon
    else:
        num_rollouts = 24 if num_rollouts is None else num_rollouts
        horizon = 2.0 if horizon is None else horizon
        cem_cfg = CrossEntropyMethodConfig(**dict(reference["optimizer"], num_rollouts=num_rollouts))
        ctrl_cfg = ControllerConfig(**dict(reference["controller"], horizon=horizon))
    settings = dict(settings, num_rollouts=num_rollouts, horizon=horizon)
    optimizer = CrossEntropyMethod(cem_cfg, task.nu)
    controller = Controller(
        ctrl_cfg,
        task,
        optimizer,
        rollout_backend="mujoco_hierarchical",
        rollout_backend_registry={"mujoco_hierarchical": OfflineBackend},
    )
    controller.update_traces = lambda: None
    plant = OfflinePlant(task)
    initial = dict(
        qpos=task.data.qpos.copy(),
        qvel=task.data.qvel.copy(),
        policy_output=plant.last_policy_output.copy(),
        time=float(task.data.time),
    )
    if not task_name:
        for key in ("qpos", "qvel", "policy_output"):
            np.testing.assert_array_equal(initial[key], reference["initial"][key])
        assert digest(initial) == reference["physical_initial_hash"]
    return task, controller, plant, initial, settings


def set_target(task, goal, segment_start=None):
    """Change goal AFTER Controller.reset(), which otherwise rotates the initial jug."""
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("Goal must be finite xyz")
    before = np.r_[task.data.qpos.copy(), task.data.qvel.copy(), task.data.time]
    task.config.goal_pos = goal.copy()
    if segment_start is not None:
        task.config.start_pos = np.asarray(segment_start, dtype=float).copy()
        task.rolling_distance = task.axial_rotation = 0.0
        task._last_metric_time = float(task.data.time)
    np.testing.assert_array_equal(before, np.r_[task.data.qpos, task.data.qvel, task.data.time])


def gates(task, distance):
    c = task.config
    push = float(np.clip((distance - c.arrive_tolerance) / c.arrive_ramp, 0, 1))
    return dict(
        push_gate=push,
        effective_approach_weight=c.w_approach * (1 - np.exp(-((distance / c.position_tolerance) ** 2))) * push,
        effective_reach_weight=c.w_hand_reach * push,
        effective_rest_weight=c.w_arm_rest * (1 - push),
    )


def candidate_diagnostics(task, controller, snapshot=False):
    states, sensors = controller.states, controller.sensors
    terms = task.reward_terms(states, sensors, controller.rollout_controls, controller.system_metadata)
    rewards = sum(terms.values())
    np.testing.assert_allclose(rewards, controller.rewards, rtol=0, atol=1e-10)
    current = task.metrics(task.data)["goal_distance"]
    end = np.linalg.norm(
        states[:, -1, task.object_pose_start : task.object_pose_start + 2] - task.config.goal_pos[:2], axis=-1
    )
    progress = current - end
    safe = (states[..., task.body_pose_start + 2] > task.config.spot_fallen_threshold).all(-1)
    advances = (progress >= 0.05) & safe
    elite = np.argsort(rewards)[::-1][: controller.optimizer.num_elites]
    without_rest = rewards - terms.get("arm_rest", 0)
    result = dict(
        progressing_candidates=int(advances.sum()),
        any_progress_candidate=float(advances.any()),
        best_progress_m=float(progress[safe].max()) if safe.any() else 0,
        winner_progress_m=float(progress[elite[0]]),
        winner_progresses=float(advances[elite[0]]),
        elite_progress_count=int(advances[elite].sum()),
        nominal_progress_m=float(progress[0]),
        winner_changes_without_rest=float(np.argmax(without_rest) != elite[0]),
        winner_without_rest_progress_m=float(progress[np.argmax(without_rest)]),
        sampled_reward_mean=float(rewards.mean()),
        sampled_reward_best=float(rewards.max()),
    )
    if snapshot:
        result["snapshot"] = dict(
            elites=elite,
            reward=rewards,
            progress_m=progress,
            no_fall=safe,
            terms=terms,
            winner_without_rest=int(np.argmax(without_rest)),
            interpretation=f"Rescoring the SAME {len(rewards)} predictions only; no extra rollout or control change",
        )
    return result


def aggregate_rows(rows):
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("time", "snapshot", "goal_index")]
    return {k: speed_summary([r[k] for r in rows]) for k in keys}


def intervals(rows, key, dt=0.02):
    found, active = [], None
    for row in rows:
        if row[key] and active is None:
            active = max(0, row["time"] - dt)
        if not row[key] and active is not None:
            found.append([active, row["time"] - dt])
            active = None
    if active is not None:
        found.append([active, rows[-1]["time"]])
    return found


def _torso_tilt_stats(task, poses):
    """Body roll/pitch over the run (degrees): mean |.|, rms, max -- the 'does it look steady' numbers."""
    from sumo.tasks.spot.spot_jug_manipulation import torso_roll_pitch
    q = np.asarray(poses)[:, task.body_pose_start + 3 : task.body_pose_start + 7]
    roll, pitch = torso_roll_pitch(q)
    out = {}
    for name, a in (("roll", np.degrees(roll)), ("pitch", np.degrees(pitch))):
        out[name] = {"mean_abs": float(np.mean(np.abs(a))), "rms": float(np.sqrt(np.mean(a * a))), "max_abs": float(np.max(np.abs(a)))}
    return out


def run(
    out,
    scenario,
    seed,
    render_video=True,
    num_rollouts=24,
    horizon=2.0,
    hand_reach_weight=None,
    hand_reach_halfwidth=0.0,
    task_name=None,
    water_ball_radius=None,
    ground_friction=None,
    reward_sets=None,
):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed{seed}.json"
    if path.exists() or path.with_suffix(".mp4").exists():
        raise FileExistsError(path)
    goals = SCENARIOS[scenario]
    task, controller, plant, initial, settings = make_system(
        num_rollouts, horizon, hand_reach_weight, hand_reach_halfwidth, task_name, water_ball_radius, ground_friction,
        reward_sets,
    )
    num_rollouts, horizon = settings["num_rollouts"], settings["horizon"]   # resolved (registered defaults)
    if render_video:
        # Fail before a long simulation if the headless GL context is unavailable.
        with mujoco.Renderer(task.model, height=64, width=64):
            pass
    set_target(task, goals[0])
    initial_config = asdict(task.config)
    groups = contact_groups(task)
    np.random.seed(seed)
    source_paths = [Path(__file__), Path("tools/jug_demo.py"), Path("sumo/tasks/spot/spot_jug_manipulation.py")]
    if hand_reach_halfwidth:
        source_paths.append(Path("tools/jug_roll_region_task.py"))
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    rows, plans, snapshots, latencies, motions = [], [], [], [], []
    poses, frame_metrics = [task.data.qpos.copy()], []
    stage, stage_start, dwell, position_dwell = 0, 0.0, 0, 0
    stages, first_success, first_position, completed = [], None, None, False
    stage_contacts = dict(arm=0, leg=0, body=0)
    totals = dict(arm=0, leg=0, body=0)
    ever_fallen, retained = False, 1.0
    reward_sum = {}
    initial_m = task.metrics(task.data)
    frame_metrics.append(
        dict(
            initial_m,
            time=0.0,
            success=False,
            goal_index=stage,
            goal_pos=task.config.goal_pos.copy(),
            segment_start=task.config.start_pos.copy(),
        )
    )
    start_wall = time.monotonic()

    def stage_record(status):
        return dict(
            index=stage,
            start_time=stage_start,
            end_time=float(task.data.time),
            elapsed=float(task.data.time) - stage_start,
            goal=goals[stage],
            segment_start=task.config.start_pos.copy(),
            status=status,
            first_success=first_success,
            first_position_settled=first_position,
            end_success=dwell * task.dt >= 0.75,
            end_position_settled=position_dwell * task.dt >= 0.75,
            contact_seconds={k: v * task.dt for k, v in stage_contacts.items()},
            final=task.metrics(task.data),
        )

    for step in range(round(30 * len(goals) / task.dt)):
        # Switch only at the next normal planning tick: no extra plan, no reset.
        if len(goals) > 1 and dwell * task.dt >= 0.75 and stage < len(goals) - 1 and planning_due(step, task.dt, 20):
            stages.append(stage_record("success"))
            stage += 1
            stage_start = float(task.data.time)
            set_target(task, goals[stage], task.data.qpos[task.object_pose_start : task.object_pose_start + 3])
            dwell = position_dwell = 0
            first_success = first_position = None
            stage_contacts = dict(arm=0, leg=0, body=0)
            # Frame at this exact timestamp belongs to the newly selected goal.
            frame_metrics[-1].update(
                goal_index=stage,
                goal_pos=task.config.goal_pos.copy(),
                segment_start=task.config.start_pos.copy(),
                success=False,
                **task.metrics(task.data),
            )
        if task.data.time - stage_start >= 30 - 1e-8:
            break
        if planning_due(step, task.dt, 20):
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
            controller._last_policy_output = np.tile(plant.last_policy_output, (num_rollouts, 1))
            t0 = time.perf_counter()
            controller.update_action()
            latencies.append(time.perf_counter() - t0)
            snap = scenario == "front" and any(abs(task.data.time - t) < 1e-6 for t in (10, 15, 20, 23))
            diag = candidate_diagnostics(task, controller, snap)
            if "snapshot" in diag:
                snapshots.append(dict(time=float(task.data.time), **diag.pop("snapshot")))
            plans.append(dict(time=float(task.data.time), goal_index=stage, **diag))
        action = controller.action(task.data.time)
        mapped = task.task_to_sim_ctrl(action).reshape(-1)
        plant.step(action)
        task.post_sim_step()
        if not np.isfinite(np.r_[task.data.qpos, task.data.qvel]).all():
            raise FloatingPointError("Invalid executed state; not a task failure")
        m = task.metrics(task.data)
        raw_success = task.success(task.model, task.data)
        positional = (
            m["goal_distance"] < task.config.position_tolerance
            and abs(m["tilt_deg"] - 90) < 15
            and 0.08 < m["jug_height"] < 0.20
            and m["linear_speed"] < 0.15
            and m["angular_speed"] < 0.6
            and m["robot_height"] > task.config.spot_fallen_threshold
        )
        dwell = dwell + 1 if raw_success else 0
        position_dwell = position_dwell + 1 if positional else 0
        if first_success is None and dwell * task.dt >= 0.75:
            first_success = float(task.data.time)
        if first_position is None and position_dwell * task.dt >= 0.75:
            first_position = float(task.data.time)
        contact = touching(task, groups)
        for k in totals:
            totals[k] += int(contact[k])
            stage_contacts[k] += int(contact[k])
        motion = motion_sample(task)
        motions.append(motion)
        terms = task.reward_terms(
            np.r_[task.data.qpos, task.data.qvel][None, None], task.data.sensordata[None, None], action[None, None]
        )
        for k, v in terms.items():
            reward_sum[k] = reward_sum.get(k, 0) + float(v.item())
        gate = gates(task, m["goal_distance"])
        hand = task.data.sensordata[task.hand_position_start : task.hand_position_start + 3]
        obj = task.data.qpos[task.object_pose_start : task.object_pose_start + 3]
        body = task.data.qpos[task.body_pose_start : task.body_pose_start + 3]
        row = dict(
            time=float(task.data.time),
            goal_index=stage,
            distance=m["goal_distance"],
            jug_speed=m["linear_speed"],
            arm_contact=float(contact["arm"]),
            stationary_unfinished=float(m["linear_speed"] < 0.02 and m["goal_distance"] >= 0.35 and not contact["arm"]),
            body_jug_distance=float(np.linalg.norm(body[:2] - obj[:2])),
            hand_jug_distance=float(np.linalg.norm(hand - obj)),
            raw_xy_command=float(np.linalg.norm(action[:2])),
            mapped_xy_command=float(np.linalg.norm(mapped[:2])),
            raw_abs_yaw=abs(float(action[2])),
            mapped_abs_yaw=abs(float(mapped[2])),
            yaw_floor_active=float(0.1 <= abs(action[2]) < 0.4),
            xy_command_zeroed=float(0 < np.linalg.norm(action[:2]) < 0.08),
            **gate,
            **{f"reward_{k}": float(v.item()) for k, v in terms.items()},
        )
        if hand_reach_halfwidth:
            reach_distance = float(task.reach_distance(hand, obj))
            row.update(
                guide_distance=reach_distance,
                inside_guide_unfinished_no_contact=float(
                    reach_distance <= 1e-10 and m["goal_distance"] >= 0.35 and not contact["arm"]
                ),
                inside_guide_stationary_unfinished=float(reach_distance <= 1e-10 and row["stationary_unfinished"]),
            )
        rows.append(row)
        poses.append(task.data.qpos.copy())
        frame_metrics.append(
            dict(
                m,
                time=float(task.data.time),
                success=dwell * task.dt >= 0.75,
                goal_index=stage,
                goal_pos=task.config.goal_pos.copy(),
                segment_start=task.config.start_pos.copy(),
            )
        )
        retained = min(retained, m.get("water_retained_fraction", 1.0))   # no-water profiles report none
        ever_fallen |= task.failure(task.model, task.data)
        if (step + 1) % 50 == 0:
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        scenario=scenario,
                        time=task.data.time,
                        stage=stage,
                        error=m["goal_distance"],
                        first_success=first_success,
                        contact_seconds={k: v * task.dt for k, v in totals.items()},
                    )
                ),
                flush=True,
            )
        if len(goals) > 1 and stage == len(goals) - 1 and dwell * task.dt >= 0.75:
            completed = True
            break
    if len(goals) == 1:
        completed = dwell * task.dt >= 0.75
    stages.append(stage_record("success" if dwell * task.dt >= 0.75 else "timeout"))
    bins = []
    for index in range(int(np.ceil(task.data.time / 2))):
        a, b = 2 * index, 2 * (index + 1)
        executed = [r for r in rows if a < r["time"] <= b + 1e-8]
        predicted = [r for r in plans if a - 1e-8 <= r["time"] < b - 1e-8]
        bins.append(
            dict(
                start=a,
                end=min(b, float(task.data.time)),
                executed=aggregate_rows(executed),
                candidates=aggregate_rows(predicted),
                plan_count=len(predicted),
            )
        )
    result = dict(
        scenario=scenario,
        seed=seed,
        goals=goals,
        initial=initial,
        physical_initial_hash=digest(initial),
        initial_config=initial_config,
        planning_settings=settings,
        controller=asdict(controller.controller_cfg),
        optimizer=asdict(controller.optimizer.config),
        duration=float(task.data.time),
        steps=len(rows),
        planning_updates=len(plans),
        candidate_evaluations=num_rollouts * len(plans),
        predicted_policy_steps_per_plan=num_rollouts * round(horizon / task.dt),
        completed_all=completed,
        stages=stages,
        final=task.metrics(task.data),
        ever_fallen=ever_fallen,
        min_water_retained=retained,
        torso_tilt_deg=_torso_tilt_stats(task, poses),
        contact_seconds={k: v * task.dt for k, v in totals.items()},
        mean_reward_terms={k: v / len(rows) for k, v in reward_sum.items()},
        motion={k: speed_summary([m[k] for m in motions]) for k in motions[0]},
        diagnostic_bins=bins,
        candidate_snapshots=snapshots,
        arm_contact_intervals=intervals(rows, "arm_contact"),
        stationary_unfinished_intervals=[x for x in intervals(rows, "stationary_unfinished") if x[1] - x[0] >= 2],
        planning_ms=speed_summary(np.array(latencies) * 1000),
        wall_seconds=time.monotonic() - start_wall,
        source_sha256_at_start=sources,
        storage="Settings, aggregate diagnostics, scalar candidate score snapshots and video only. No state/action/rollout archives.",
        protocol="Frozen physical initial state; target changed after initialization. No added control rollouts. 30s per goal; switch at ordinary planning tick after >=0.75s success. Keep CEM warm start, reset only segment metrics/direction.",
    )
    if hand_reach_halfwidth:
        result["region_diagnostics"] = dict(
            inside_unfinished_no_contact_seconds=task.dt * sum(r["inside_guide_unfinished_no_contact"] for r in rows),
            inside_stationary_unfinished_seconds=task.dt * sum(r["inside_guide_stationary_unfinished"] for r in rows),
            long_inside_stationary_intervals=[
                x for x in intervals(rows, "inside_guide_stationary_unfinished") if x[1] - x[0] >= 2
            ],
        )
    if (
        not task_name
        and scenario == "front"
        and num_rollouts == 24
        and horizon == 2.0
        and seed in (0, 1, 2)
        and initial_config["w_hand_reach"] == 15
        and hand_reach_halfwidth == 0
    ):
        old = json.loads((REFERENCE / f"gentle_seed{seed}.json").read_text())
        np.testing.assert_allclose(first_success, old["first_success"], rtol=0, atol=1e-10)
        for key, value in old["final"].items():
            if key in result["final"]:
                np.testing.assert_allclose(result["final"][key], value, rtol=0, atol=1e-10)
        result["exact_replay_verified"] = True
    if render_video:
        video = path.with_suffix(".mp4")
        render(
            task,
            poses[:-1],
            frame_metrics[:-1],
            video,
            horizon,
            f"{num_rollouts}x1 H{horizon:g} | {scenario} | seed {seed}",
            view="route",
        )
        result["video"] = str(video.resolve())
    path.write_text(json.dumps(result, default=jsonable, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(scenario=scenario, seed=seed, completed_all=completed, duration=result["duration"], stages=stages),
            default=jsonable,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--num-rollouts", type=int, default=None, help="default 24, or the registered profile's with --task")
    p.add_argument("--horizon", type=float, default=None, help="default 2.0 s, or the registered profile's with --task")
    p.add_argument(
        "--hand-reach-halfwidth",
        type=float,
        default=0.0,
        help="Experiment-only lateral segment halfwidth (metres); zero preserves original reward",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Run a REGISTERED task (e.g. spot_jug_roll_arm_gentle_coarse) from its defaults "
             "instead of the frozen reference config; the initial-state pins are skipped",
    )
    p.add_argument(
        "--water-ball-radius",
        type=float,
        default=None,
        help="With --task: override only the water ball radius (ball count follows the fixed fill)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="With --task: set a REWARD-ONLY config field after construction (e.g. w_torso_roll=200); "
             "construction-time fields are refused",
    )
    p.add_argument(
        "--ground-friction",
        type=float,
        default=None,
        help="With --task: override the floor's sliding friction (feet only; the jug keeps its own)",
    )
    p.add_argument(
        "--hand-reach-weight",
        type=float,
        default=None,
        help="Override only the hand approach reward; omitted preserves the frozen reference",
    )
    args = p.parse_args()
    run(
        args.out,
        args.scenario,
        args.seed,
        not args.no_render,
        args.num_rollouts,
        args.horizon,
        args.hand_reach_weight,
        args.hand_reach_halfwidth,
        args.task,
        args.water_ball_radius,
        args.ground_friction,
        dict(kv.split("=", 1) for kv in args.set),
    )


if __name__ == "__main__":
    main()
