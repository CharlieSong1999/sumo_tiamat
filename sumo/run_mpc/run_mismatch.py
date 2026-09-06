# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
r"""Headless planner/plant model-mismatch runner for the Spot pit water tasks.

The CONTROLLER plans on the registered planner task (no water balls; optionally the
rigid-water mass variant), while the PLANT simulation steps the `_plant` task variant
whose model has N water balls appended. Plant state maps onto planner state by
slicing off the trailing ball DOFs (see sumo.tasks.spot.spot_pit_base.project_state).

Both models use the stock Newton solver. Everything else (spline controller, CEM,
C++ ONNX locomotion rollouts) is unchanged judo/sumo machinery.

Usage:
    MUJOCO_GL=egl pixi run python -m sumo.run_mpc.run_mismatch \\
        --task spot_pit_move --fill-ratio 0.5 --planner-water-model rigid_mass \\
        --num-episodes 1 --video
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
from judo.app.structs import MujocoState
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_task_registration
from judo.tasks.spot.spot_constants import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME, POLICY_OUTPUT_DIM

import sumo.controller  # noqa: F401 -- register controller/optimizer overrides
import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.controller import Controller, ControllerConfig
from sumo.tasks.spot.spot_pit_base import (
    pitcher_tilt_deg,
    set_rigid_water_on_model,
    settled_ball_poses,
    spill_fraction,
    spout_angle_deg,
    water_in_pitcher,
)
from sumo.tasks.spot.spot_pit_move import SpotPitMove, SpotPitMovePlant
from sumo.tasks.spot.spot_pit_pour import SpotPitPour, SpotPitPourPlant

try:
    from mujoco_extensions.policy_rollout import create_systems_vector, threaded_rollout
except ImportError as e:  # pragma: no cover
    raise ImportError("mujoco_extensions is not built; run `pixi run build` first.") from e

TASK_CLASSES = {
    "spot_pit_move": (SpotPitMove, SpotPitMovePlant),
    "spot_pit_pour": (SpotPitPour, SpotPitPourPlant),
}


@dataclass
class MismatchConfig:
    """Config for a planner/plant mismatch run."""

    task: str = "spot_pit_move"
    fill_ratio: float = 0.5
    # empty       : planner pitcher is the bare 1 kg container
    # rigid_mass  : + water mass/inertia as a fixed rigid lump (build-time)
    # chunky      : planner model carries a few LARGE balls of equal water mass, so
    #               rollouts predict slosh/spill; state re-seeded from the plant each replan
    # adaptive_rigid : rigid lump whose mass/COM is re-estimated from the plant during
    #               the episode; the rollout backend is rebuilt on each update (~1.6 s)
    planner_water_model: str = "empty"
    chunky_radius: float = 0.025  # chunky only: 25 mm -> ~35 balls at fill 0.5
    # Uniform pitcher scale: geometry/mass/water shrink together (mass ~ scale^3), but
    # the handle grip cross-section stays full-size so the grasp interface is unchanged
    # -- isolates the "lighter load" effect on controllability.
    pit_scale: float = 1.0
    planner_solver: str = "newton"  # newton | cg  (planner/rollout model only)
    adaptive_update_period_s: float = 2.0  # adaptive_rigid only
    adaptive_mass_deadband_kg: float = 0.15  # skip updates smaller than this
    target_tilt_deg: float = 90.0  # LEGACY (pour v2); ignored by pour v3
    target_spout_angle_deg: float = 45.0  # pour v3: 90 = upright, 0 = spout straight down
    grasp_mode: str = "plate"  # pour v3: plate | wall | inside
    handle_style: str = "plate"  # C-handle vertical grip: plate | cylinder
    engage_any_hold: bool = False  # method A: any mech-coupled airborne hold counts
    start_grasped: bool = False  # method B: episode starts from a scripted grasped state
    # Slide friction of the pitcher's body/floor/wall collision geoms (applied to BOTH
    # models -- a known property, not a mismatch subject). Pit geoms have priority 6 >
    # ground's 5, so pit-ground contacts use THIS value; raising it (e.g. 1.5) makes the
    # push-along-the-ground strategy expensive without touching the robot's foot-ground
    # friction. Handle geoms (priority 8) are unaffected.
    pit_friction: float = 0.6
    optimizer: str = "cem"
    num_episodes: int = 1
    episode_length_s: float = 30.0
    seed_base: int = 0
    video: bool = True
    video_fps: int = 20
    output_dir: str = "out/water_sim"
    run_name: str = ""  # defaults to a name derived from the settings
    viz_dt: float = 0.05  # metric/video sampling period


class PlantSim:
    """Minimal hierarchical (WBC-in-the-loop) simulation of a plant task instance.

    Mirrors judo's HierarchicalMJSimulation.step() but takes a task INSTANCE (so the
    plant task can be constructed with a non-default fill_ratio) instead of a task name.
    """

    def __init__(self, task, policy_path: str) -> None:
        self.task = task
        self._systems = create_systems_vector(task.model, str(policy_path), 1)
        self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)

    def reset(self) -> None:
        self.task.reset()
        self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)

    def step(self, command: np.ndarray) -> None:
        command = np.asarray(self.task.task_to_sim_ctrl(command), dtype=np.float64).flatten()
        state = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.task.pre_sim_step()
        out_states, _, policy_outputs = threaded_rollout(
            self._systems,
            np.array([state], dtype=np.float64),
            np.array([[command]], dtype=np.float64),
            np.array([self._last_policy_output], dtype=np.float64),
            1,
            self.task.physics_substeps,
            DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
        )
        self.task.post_sim_step()
        final_state = np.array(out_states[0][-1])
        nq = self.task.model.nq
        self.task.data.qpos[:] = final_state[:nq]
        self.task.data.qvel[:] = final_state[nq:]
        self.task.data.time += self.task.dt
        mujoco.mj_forward(self.task.model, self.task.data)
        self._last_policy_output = np.array(policy_outputs[0])


class EpisodeRenderer:
    """EGL offscreen renderer tracking the midpoint between robot and pitcher.

    Optionally draws a translucent blue goal marker (sphere + post) at ``goal``.
    """

    def __init__(
        self, model: mujoco.MjModel, width: int = 800, height: int = 480, goal: np.ndarray | None = None
    ) -> None:
        # Grow the offscreen framebuffer if the compiled default (640x480) is too small.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        # Brighten the headlight so the untextured ground reads as a floor, not a void.
        model.vis.headlight.ambient = [0.35, 0.35, 0.35]
        model.vis.headlight.diffuse = [0.6, 0.6, 0.6]
        # The ground geom sits in group 3 (collision), which the offscreen renderer
        # hides by default -- move it to group 0 and give it a plain gray look (the
        # blue_grid texture also renders near-black offscreen). Group is viz-only.
        ground = model.geom("ground")
        ground.group = 0
        ground.matid = -1
        ground.rgba = [0.72, 0.73, 0.76, 1.0]
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 3.2
        self.cam.azimuth = 145
        self.cam.elevation = -22
        self._body_id = model.body("body").id
        self._pit_id = model.body("pit").id
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64)

    def _add_goal_marker(self) -> None:
        scene = self.renderer.scene
        eye = np.eye(3).flatten()
        for size, pos, rgba in (
            ([0.12, 0, 0], self._goal + [0, 0, 0.12], [0.1, 0.45, 1.0, 0.35]),  # sphere at goal
            ([0.01, 0.01, 0.5], self._goal + [0, 0, 0.5], [0.1, 0.45, 1.0, 0.55]),  # tall post
        ):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE if size[1] == 0 else mujoco.mjtGeom.mjGEOM_BOX,
                np.array(size, dtype=np.float64),
                np.asarray(pos, dtype=np.float64),
                eye,
                np.array(rgba, dtype=np.float32),
            )
            scene.ngeom += 1

    def frame(self, data: mujoco.MjData) -> np.ndarray:
        mid = 0.5 * (data.xpos[self._body_id] + data.xpos[self._pit_id])
        self.cam.lookat = [mid[0], mid[1], 0.3]
        self.renderer.update_scene(data, self.cam)
        if self._goal is not None:
            self._add_goal_marker()
        return self.renderer.render()


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> Path:
    """Encode frames to H.264 mp4 via the system ffmpeg; fall back to GIF."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        h, w = frames[0].shape[:2]
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        if proc.wait() == 0:
            return path

    import PIL.Image

    gif = path.with_suffix(".gif")
    imgs = [PIL.Image.fromarray(f) for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    return gif


GRASP_INIT_YAW = {"plate": np.pi, "wall": -np.pi / 2, "inside": np.pi}


def initialize_grasped(plant_sim: PlantSim, grasp_mode: str) -> dict:
    """Method B: put the plant into a grasped state before handing control to MPC.

    For the handle ("plate") mode the grasp pose is specified EXPLICITLY per the task
    definition: gripper level, pointing at the vertical grip segment, wrist ROLLED 90
    degrees so the jaw opens HORIZONTALLY -- finger and palm land on the +-y flanks of
    the bar. The arm grid search matches the jaw center to the bar plus those
    orientation constraints; then the base is teleported and the jaw closed.
    """
    from judo.tasks.spot.spot_constants import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS

    task = plant_sim.task
    model, data = task.model, task.data
    arm_joints = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]
    adrs = [model.joint(j).qposadr[0] for j in arm_joints]
    base_adr = model.joint("base").qposadr[0]
    grip_adr = model.joint("arm_f1x").qposadr[0]
    fngr = model.site("site_arm_link_fngr").id
    wr1 = model.site("site_arm_link_wr1").id
    wr1_site = model.site("site_arm_link_wr1").id

    anchor_site = {"plate": "site_grasp_handle", "wall": "site_grasp_wall", "inside": "site_grasp_inside"}[grasp_mode]
    approach_grip = GRIPPER_CLOSED_POS if grasp_mode == "inside" else GRIPPER_OPEN_POS
    hold_grip = GRIPPER_OPEN_POS if grasp_mode == "inside" else GRIPPER_CLOSED_POS

    mujoco.mj_forward(model, data)
    anchor = data.site_xpos[model.site(anchor_site).id].copy()
    if grasp_mode == "plate":
        anchor[2] += 0.06  # aim at the bar's upper half: reachable by a LEVEL gripper

    yaw = GRASP_INIT_YAW[grasp_mode]
    base_q = np.array([0.0, 0.0, data.qpos[base_adr + 2], np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    qpos0 = data.qpos.copy()
    data.qpos[base_adr : base_adr + 7] = base_q
    data.qpos[grip_adr] = approach_grip
    best, best_score = None, np.inf
    for sh1 in np.arange(-1.6, 0.01, 0.1):
        for el0 in np.arange(0.6, 2.21, 0.1):
            for wr0 in np.arange(-1.2, 1.81, 0.2):
                for wr1_roll in (1.57, -1.57) if grasp_mode == "plate" else (0.0,):
                    arm = [0.0, sh1, el0, 0.0, wr0, wr1_roll]
                    for adr, v in zip(adrs, arm, strict=True):
                        data.qpos[adr] = v
                    mujoco.mj_forward(model, data)
                    fp, wp = data.site_xpos[fngr], data.site_xpos[wr1]
                    jaw = fp + 0.35 * (wp - fp)
                    site_mat = data.site_xmat[wr1_site].reshape(3, 3)
                    g_x, g_z = site_mat[:, 0], site_mat[:, 2]
                    if grasp_mode == "plate":
                        # Level gripper pointing at the bar (-x world); jaw opens
                        # horizontally (closing axis g_z parallel to world y).
                        orient = (1.0 + g_x @ np.array([1.0, 0, 0])) + abs(g_x[2]) + abs(g_z[2])
                    elif grasp_mode == "wall":
                        orient = (1.0 + g_x[2]) + abs(abs(g_z[1]) - 1.0)
                    else:
                        orient = 1.0 + g_x[2]
                    reach = np.linalg.norm(jaw[:2] - base_q[:2])
                    score = 6.0 * abs(jaw[2] - anchor[2]) + 2.0 * abs(reach - 0.62) + 2.0 * orient
                    if score < best_score:
                        best_score = score
                        best = (arm, jaw.copy())
    data.qpos[:] = qpos0
    assert best is not None
    arm, jaw = best

    data.qpos[base_adr : base_adr + 7] = base_q
    data.qpos[base_adr : base_adr + 2] += (anchor - jaw)[:2]
    for adr, v in zip(adrs, arm, strict=True):
        data.qpos[adr] = v
    data.qpos[grip_adr] = approach_grip
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    fp, wp = data.site_xpos[fngr], data.site_xpos[wr1]
    jaw_now = fp + 0.35 * (wp - fp)
    data.qpos[base_adr : base_adr + 2] += (anchor - jaw_now)[:2]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    def action(grip):
        return np.array([0.0, 0.0, 0.0, *arm, grip, 0.0])

    for _ in range(int(round(0.6 / task.dt))):
        plant_sim.step(action(approach_grip))
    for _ in range(int(round(1.5 / task.dt))):
        plant_sim.step(action(hold_grip))

    grip_q = float(data.qpos[grip_adr])
    mech_ok = (grip_q > -0.5) if grasp_mode == "inside" else (-1.2 < grip_q < -0.06)
    fp = data.site_xpos[fngr]
    residual = float(np.linalg.norm((fp + 0.35 * (data.site_xpos[wr1] - fp)) - data.site_xpos[model.site(anchor_site).id]))
    info = {"score": round(best_score, 3), "grip_q": round(grip_q, 3), "mech_ok": mech_ok, "residual": round(residual, 3)}
    print(f"  [start_grasped:{grasp_mode}] {info}")
    return info


def build(config: MismatchConfig):
    """Construct (planner_task, plant_task, controller, plant_sim)."""
    planner_cls, plant_cls = TASK_CLASSES[config.task]

    # adaptive_rigid starts from the rigid_mass model (best initial guess: all water
    # inside); the episode loop then re-estimates mass/COM and rebuilds the backend.
    build_water_model = "rigid_mass" if config.planner_water_model == "adaptive_rigid" else config.planner_water_model
    if config.task == "spot_pit_pour":
        planner_task = planner_cls(
            fill_ratio=config.fill_ratio,
            planner_water_model=build_water_model,
            chunky_radius=config.chunky_radius,
            grasp_mode=config.grasp_mode,
            handle_style=config.handle_style,
            pit_scale=config.pit_scale,
        )
        plant_task = plant_cls(
            fill_ratio=config.fill_ratio,
            grasp_mode=config.grasp_mode,
            handle_style=config.handle_style,
            pit_scale=config.pit_scale,
        )
        planner_task.config.target_spout_angle_deg = config.target_spout_angle_deg
        plant_task.config.target_spout_angle_deg = config.target_spout_angle_deg
        planner_task.config.engage_any_hold = config.engage_any_hold
        plant_task.config.engage_any_hold = config.engage_any_hold
    else:
        planner_task = planner_cls(
            fill_ratio=config.fill_ratio,
            planner_water_model=build_water_model,
            chunky_radius=config.chunky_radius,
            pit_scale=config.pit_scale,
        )
        plant_task = plant_cls(fill_ratio=config.fill_ratio, pit_scale=config.pit_scale)

    # Optional CG solver for the PLANNER model only (runtime-switchable; must be set
    # before the controller snapshots the model into its C++ rollout systems).
    if config.planner_solver == "cg":
        planner_task.model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        planner_task.model.opt.iterations = 50
        planner_task.model.opt.ls_iterations = 20
        planner_task.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    elif config.planner_solver != "newton":
        raise ValueError(f"planner_solver must be 'newton' or 'cg', got {config.planner_solver!r}")

    # Apply the pitcher friction knob to both models BEFORE the controller/plant sim
    # copy them into their C++ rollout systems.
    for model in (planner_task.model, plant_task.model):
        for geom_name in ("pit_floor_collision", "pit_wall_yneg", "pit_wall_ypos", "pit_wall_xneg", "pit_wall_xpos"):
            model.geom(geom_name).friction[0] = config.pit_friction

    optimizer_cls, optimizer_config_cls = get_registered_optimizers()[config.optimizer]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override(config.task)
    optimizer = optimizer_cls(optimizer_config, planner_task.nu)

    controller_config = ControllerConfig()
    controller_config.set_override(config.task)
    controller = Controller(controller_config, planner_task, optimizer, rollout_backend="mujoco_hierarchical")

    policy_path = get_task_registration(config.task).locomotion_policy_path
    plant_sim = PlantSim(plant_task, policy_path)
    return planner_task, plant_task, controller, plant_sim


def run_episode(
    config: MismatchConfig, planner_task, plant_task, controller, plant_sim, episode_idx: int, out_dir: Path
) -> dict:
    """One mismatch episode: plan on the planner model, step the plant model."""
    # One PlantSim.step advances task.dt (= model timestep x physics_substeps, i.e. one
    # 50 Hz locomotion-policy tick). NOTE: upstream run_mpc.py counts steps with the raw
    # model timestep, which doubles the actual episode duration; we count with task.dt.
    step_dt = plant_task.dt
    plan_dt = 1.0 / controller.controller_cfg.control_freq
    num_steps = int(config.episode_length_s / step_dt) + 1
    steps_per_plan = max(1, int(round(plan_dt / step_dt)))
    steps_per_viz = max(1, int(round(config.viz_dt / step_dt)))

    # Shared robot+pitcher state prefix (identical indices in both models).
    nq_shared = planner_task.model.nq - 7 * planner_task._n_chunky
    nv_shared = planner_task.model.nv - 6 * planner_task._n_chunky
    n_balls = plant_task._n_balls
    is_chunky = planner_task._n_chunky > 0
    pit_scale = planner_task._pit_scale
    chunky_template = (
        settled_ball_poses(planner_task._n_chunky, radius=planner_task._chunky_radius, scale=pit_scale)
        if is_chunky
        else np.empty((0, 3))
    )
    pit_adr = plant_task.object_pose_idx
    # object_vel_idx indexes the CONCATENATED [qpos|qvel] state (offset by nq);
    # subtract nq to index data.qvel directly.
    pit_vel_adr = plant_task.object_vel_idx - plant_task.model.nq

    is_adaptive = config.planner_water_model == "adaptive_rigid"
    adaptive_updates: list[dict] = []
    last_applied_mass = None
    steps_per_adaptive = max(1, int(round(config.adaptive_update_period_s / step_dt)))

    np.random.seed(config.seed_base + episode_idx)
    plant_sim.reset()
    planner_task.reset()
    controller.reset()
    plant_task.data.time = 0.0
    grasp_init_info = None
    if config.start_grasped and config.task == "spot_pit_pour":
        grasp_init_info = initialize_grasped(plant_sim, config.grasp_mode)

    goal = np.asarray(planner_task.config.goal_position) if config.task == "spot_pit_move" else None
    renderer = EpisodeRenderer(plant_task.model, goal=goal) if config.video else None
    frames: list[np.ndarray] = []
    metrics = {
        "time": [],
        "spill_frac": [],
        "tilt_deg": [],
        "spout_deg": [],
        "pit_pos": [],
        "grasp_dist": [],
        "reward": [],
    }
    success = False
    success_time = None
    current_action = None

    # Engage distance: for pour tasks measure to the ACTIVE grasp-mode target site,
    # otherwise to the handle plate.
    if hasattr(plant_task, "pour_target_idx"):
        grasp_sensor_adr = plant_task.pour_target_idx
    else:
        grasp_sensor_adr = plant_task.model.sensor("sensor_gripper_to_grasp_handle").adr[0]

    def planner_state() -> tuple[np.ndarray, np.ndarray]:
        """Build the planner state from the plant state.

        Shared prefix = slice. For the chunky planner, the large balls are re-seeded
        every replan at their settled template poses in the CURRENT pitcher frame
        (i.e. "water is settled now"); the rollouts then simulate their slosh/spill.
        """
        qpos_p = np.array(plant_task.data.qpos[:nq_shared])
        qvel_p = np.array(plant_task.data.qvel[:nv_shared])
        if not is_chunky:
            return qpos_p, qvel_p
        pit_pos = plant_task.data.qpos[pit_adr : pit_adr + 3]
        pit_quat = plant_task.data.qpos[pit_adr + 3 : pit_adr + 7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, pit_quat)
        rot = rot.reshape(3, 3)
        pit_linvel = plant_task.data.qvel[pit_vel_adr : pit_vel_adr + 3]
        ball_qpos = np.concatenate([np.concatenate([pit_pos + rot @ p, pit_quat]) for p in chunky_template])
        ball_qvel = np.concatenate([np.concatenate([pit_linvel, np.zeros(3)]) for _ in chunky_template])
        return np.concatenate([qpos_p, ball_qpos]), np.concatenate([qvel_p, ball_qvel])

    for step in range(num_steps):
        curr_time = step * step_dt

        # adaptive_rigid: periodically re-estimate the retained water (mass + COM in
        # the pitcher frame) from the plant and rebuild the rollout backend so the
        # planner's rigid lump tracks reality (pouring out, sloshing sideways).
        if is_adaptive and step > 0 and step % steps_per_adaptive == 0:
            water_mass, water_com = water_in_pitcher(plant_task.model, plant_task.data, n_balls, scale=pit_scale)
            if last_applied_mass is None or abs(water_mass - last_applied_mass) > config.adaptive_mass_deadband_kg:
                set_rigid_water_on_model(planner_task.model, water_mass, water_com, scale=pit_scale)
                t0 = time.perf_counter()
                controller.rollout_backend = controller._make_rollout_backend("mujoco_hierarchical")
                rebuild_s = time.perf_counter() - t0
                last_applied_mass = water_mass
                adaptive_updates.append(
                    {
                        "t": round(curr_time, 2),
                        "water_mass_kg": round(water_mass, 3),
                        "water_com": [round(float(v), 3) for v in water_com],
                        "backend_rebuild_s": round(rebuild_s, 2),
                    }
                )
                print(
                    f"  [adaptive] t={curr_time:.1f}s water={water_mass:.2f}kg com={np.round(water_com, 3)} (rebuild {rebuild_s:.1f}s)"
                )

        if step % steps_per_plan == 0:
            qpos_p, qvel_p = planner_state()
            controller.update_states(
                MujocoState(
                    time=curr_time,
                    qpos=qpos_p,
                    qvel=qvel_p,
                    mocap_pos=np.array(planner_task.data.mocap_pos),
                    mocap_quat=np.array(planner_task.data.mocap_quat),
                    sim_metadata={},
                )
            )
            controller.update_action()

        current_action = controller.action(curr_time)
        plant_sim.step(current_action)

        if not success and plant_task.success(plant_task.model, plant_task.data):
            success = True
            success_time = curr_time

        if step % steps_per_viz == 0:
            metrics["time"].append(round(curr_time, 3))
            metrics["spill_frac"].append(
                round(spill_fraction(plant_task.model, plant_task.data, n_balls, scale=pit_scale), 4)
            )
            metrics["tilt_deg"].append(round(pitcher_tilt_deg(plant_task.model, plant_task.data), 2))
            metrics["spout_deg"].append(round(spout_angle_deg(plant_task.model, plant_task.data), 2))
            pit_pos = plant_task.data.qpos[plant_task.object_pose_idx : plant_task.object_pose_idx + 3]
            metrics["pit_pos"].append([round(float(v), 3) for v in pit_pos])
            metrics["grasp_dist"].append(
                round(float(np.linalg.norm(plant_task.data.sensordata[grasp_sensor_adr : grasp_sensor_adr + 3])), 3)
            )
            metrics["reward"].append(round(float(controller.rewards.max()), 2))
            if renderer is not None:
                frames.append(renderer.frame(plant_task.data))

    result = {
        "episode": episode_idx,
        "success": bool(success),
        "success_time_s": success_time,
        "final_spill_frac": metrics["spill_frac"][-1] if metrics["spill_frac"] else 0.0,
        "max_spill_frac": max(metrics["spill_frac"], default=0.0),
        "final_tilt_deg": metrics["tilt_deg"][-1] if metrics["tilt_deg"] else 0.0,
        "final_spout_deg": metrics["spout_deg"][-1] if metrics["spout_deg"] else 90.0,
        "min_grasp_dist": min(metrics["grasp_dist"], default=float("nan")),
    }
    if is_adaptive:
        result["adaptive_updates"] = adaptive_updates
    if grasp_init_info is not None:
        result["grasp_init"] = grasp_init_info
    (out_dir / f"episode_{episode_idx}_metrics.json").write_text(json.dumps({**result, "curves": metrics}, indent=2))
    if renderer is not None and frames:
        video_path = save_video(frames, out_dir / f"episode_{episode_idx}.mp4", config.video_fps)
        result["video"] = str(video_path)
    return result


def main(config: MismatchConfig) -> None:
    run_name = config.run_name or (
        f"{config.task}_f{int(config.fill_ratio * 100)}_{config.planner_water_model}"
        + (f"_r{round(config.chunky_radius * 1000)}" if config.planner_water_model == "chunky" else "")
        + (f"_ps{round(config.pit_scale * 100)}" if config.pit_scale != 1.0 else "")
        + (f"_{config.planner_solver}" if config.planner_solver != "newton" else "")
        + (
            f"_{config.grasp_mode}_hs-{config.handle_style}_spout{int(config.target_spout_angle_deg)}"
            + ("_anyhold" if config.engage_any_hold else "")
            + ("_sg" if config.start_grasped else "")
            if config.task == "spot_pit_pour"
            else ""
        )
    )
    out_dir = Path(config.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    planner_task, plant_task, controller, plant_sim = build(config)
    print(
        f"[{run_name}] planner nq={planner_task.model.nq} ({config.planner_water_model}, "
        f"pit mass {float(planner_task.model.body('pit').mass[0]):.2f} kg) | "
        f"plant nq={plant_task.model.nq} ({plant_task._n_balls} balls)"
    )

    results = []
    for i in range(config.num_episodes):
        result = run_episode(config, planner_task, plant_task, controller, plant_sim, i, out_dir)
        print(f"[{run_name}] episode {i}: {result}")
        results.append(result)

    summary = {
        "run_name": run_name,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_final_spill": float(np.mean([r["final_spill_frac"] for r in results])),
        "episodes": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{run_name}] success {summary['success_rate']:.0%}, mean spill {summary['mean_final_spill']:.1%}")


if __name__ == "__main__":
    main(tyro.cli(MismatchConfig))
