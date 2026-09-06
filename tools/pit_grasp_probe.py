# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Grasp-physics probe for the pit pitcher.

Question: which grasp mode lets Spot LIFT the water-filled pitcher and command a
POUR ATTITUDE (spout angle vs vertical) in the air, contact-only (no weld)?

Modes (--grasp):
  plate  -- pinch the FLAT handle plate (35x12 mm box): planar patch on both jaw faces
  wall   -- pinch the +y wall's top edge (20 mm box)
  inside -- insert the gripper head into the mouth and OPEN it, jamming fingertip and
            palm against the opposing inner walls (expansion grip)
  hook   -- open finger through the handle loop (move-task baseline; weak authority)

Scripted sequence (no MPC): pitcher rests on the ground; the robot base is teleported
so the jaw meets the grasp point at a kinematically-searched arm config. Then:
engage -> lift -> wrist-pitch sweep. Reports grasp resistance, lift height, hang tilt,
SPOUT-ANGLE trace during the sweep (the pour-attitude authority), slip, spill, video.

Run:  MUJOCO_GL=egl pixi run python tools/pit_grasp_probe.py --grasp plate [--fill 0.5]
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from judo.tasks import get_task_registration
from judo.tasks.spot.spot_constants import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS

from sumo.run_mpc.run_mismatch import EpisodeRenderer, PlantSim, save_video
from sumo.tasks.spot.spot_pit_base import pitcher_tilt_deg, spill_fraction, spout_angle_deg
from sumo.tasks.spot.spot_pit_move import SpotPitMovePlant

OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "pit_grasp_probe"

ARM_JOINTS = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]

GRASP_SITES = {
    "plate": "site_grasp_handle",
    "wall": "site_grasp_wall",
    "inside": "site_grasp_inside",
    "hook": "site_hook",
}


def search_arm_config(
    model, data, target_z: float, base_qpos: np.ndarray, vertical_insert: bool = False
) -> tuple[list[float], np.ndarray]:
    """Grid-search an arm configuration whose jaw point sits at height target_z.

    Kinematic only (set qpos + mj_forward). Returns (arm6, jaw_offset_body).
    vertical_insert=True prefers the gripper pointing DOWN (for the inside mode);
    otherwise a level gripper is preferred.
    """
    adrs = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    base_adr = model.joint("base").qposadr[0]
    fngr = model.site("site_arm_link_fngr").id
    wr1 = model.site("site_arm_link_wr1").id

    qpos0 = data.qpos.copy()
    data.qpos[base_adr : base_adr + 7] = base_qpos
    best, best_score = None, np.inf
    for sh1 in np.arange(-1.6, 0.01, 0.1):
        for el0 in np.arange(0.6, 2.21, 0.1):
            for wr0 in np.arange(-1.2, 1.81, 0.2):
                arm = [0.0, sh1, el0, 0.0, wr0, 0.0]
                for adr, v in zip(adrs, arm, strict=True):
                    data.qpos[adr] = v
                mujoco.mj_forward(model, data)
                fp, wp = data.site_xpos[fngr], data.site_xpos[wr1]
                jaw = fp + 0.3 * (wp - fp)
                reach = np.linalg.norm(jaw[:2] - base_qpos[:2])
                if vertical_insert:
                    # fngr well below wr1: gripper pointing down into the mouth.
                    orient_term = abs((wp[2] - fp[2]) - 0.10)
                else:
                    orient_term = abs(fp[2] - wp[2])
                score = 8.0 * abs(jaw[2] - target_z) + 2.0 * abs(reach - 0.62) + 3.0 * orient_term
                if score < best_score:
                    body_rot = data.xmat[model.body("body").id].reshape(3, 3)
                    body_pos = data.xpos[model.body("body").id]
                    best_score = score
                    best = (arm, body_rot.T @ (jaw - body_pos))
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    assert best is not None
    return best


def run_phase(sim, action, seconds, renderer, frames):
    n = int(round(seconds / sim.task.dt))
    for k in range(n):
        sim.step(action)
        if k % 3 == 0 and renderer is not None:
            frames.append(renderer.frame(sim.task.data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill", type=float, default=0.5)
    parser.add_argument("--grasp", choices=list(GRASP_SITES), default="plate")
    parser.add_argument("--roll", type=float, default=0.0, help="wrist roll (wr1) during grasp/lift")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    task = SpotPitMovePlant(fill_ratio=args.fill)
    sim = PlantSim(task, get_task_registration("spot_pit_move_plant").locomotion_policy_path)
    sim.reset()
    renderer = EpisodeRenderer(task.model)
    frames: list[np.ndarray] = []
    model, data = task.model, task.data
    mode = args.grasp
    print(f"fill={args.fill} -> {task._n_balls} balls; grasp={mode}, roll={args.roll}")

    fngr_site = model.site("site_arm_link_fngr").id
    grasp_site = model.site(GRASP_SITES[mode]).id
    pit_adr = model.joint("pit_joint").qposadr[0]
    base_adr = model.joint("base").qposadr[0]
    grip_adr = model.joint("arm_f1x").qposadr[0]

    # Engage/hold gripper commands per mode: pinches CLOSE; hook stays open;
    # inside-expansion OPENS against the inner walls (the jaw actuator provides the
    # expansion force).
    approach_grip = GRIPPER_CLOSED_POS if mode == "inside" else GRIPPER_OPEN_POS
    hold_grip = {"plate": GRIPPER_CLOSED_POS, "wall": GRIPPER_CLOSED_POS, "inside": GRIPPER_OPEN_POS}.get(
        mode, GRIPPER_OPEN_POS
    )

    def action(arm6, gripper):
        arm6 = list(arm6)
        arm6[5] = args.roll
        return np.array([0.0, 0.0, 0.0, *arm6, gripper, 0.0])

    # Approach from the side the grasp point faces (outward from the pitcher center);
    # the inside mode approaches from +x (the handle side, arbitrary) and inserts down.
    anchor = data.site_xpos[grasp_site].copy()
    # Palm sits ~4.8 cm below the finger line: for plate/wall bites aim the jaw higher
    # so the palm slides UNDER the target instead of ramming it.
    palm_drop = 0.045 if mode in ("plate", "wall") else 0.0
    anchor[2] += palm_drop
    pit_xy = data.qpos[pit_adr : pit_adr + 2]
    outward = anchor[:2] - pit_xy
    if np.linalg.norm(outward) < 1e-6:
        outward = np.array([1.0, 0.0])
    yaw = float(np.arctan2(-outward[1], -outward[0]))
    print(f"  grasp point {np.round(anchor, 3)}, approach yaw {np.degrees(yaw):.0f} deg")
    base_q = np.array([0.0, 0.0, data.qpos[base_adr + 2], np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    grasp_arm, jaw_off_body = search_arm_config(model, data, anchor[2], base_q, vertical_insert=(mode == "inside"))
    print(f"  arm config {np.round(grasp_arm, 2)}, jaw offset {np.round(jaw_off_body, 3)}")

    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    base_xy = anchor[:2] - (rot @ jaw_off_body)[:2]
    data.qpos[base_adr : base_adr + 2] = base_xy
    data.qpos[base_adr + 3 : base_adr + 7] = base_q[3:]
    for adr, v in zip([model.joint(j).qposadr[0] for j in ARM_JOINTS], grasp_arm, strict=True):
        data.qpos[adr] = v
    data.qpos[grip_adr] = approach_grip
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    run_phase(sim, action(grasp_arm, approach_grip), 1.0, renderer, frames)
    fp = data.site_xpos[fngr_site]
    wp = data.site_xpos[model.site("site_arm_link_wr1").id]
    jaw_now = fp + 0.3 * (wp - fp)
    err = data.site_xpos[grasp_site] - jaw_now
    data.qpos[base_adr : base_adr + 2] += err[:2]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    run_phase(sim, action(grasp_arm, approach_grip), 0.4, renderer, frames)
    print(f"  post-settle jaw->target error {np.round(err, 3)} (xy re-aligned)")

    # Engage.
    run_phase(sim, action(grasp_arm, hold_grip), 1.5, renderer, frames)
    grip_pos = float(data.qpos[grip_adr])
    resistance = grip_pos - (GRIPPER_CLOSED_POS if hold_grip == GRIPPER_CLOSED_POS else GRIPPER_OPEN_POS)
    print(f"  gripper joint {grip_pos:.3f}; resistance vs command {resistance:.3f}")

    # Lift ~0.35 m.
    lift_arm, _ = search_arm_config(model, data, anchor[2] + 0.35, base_q, vertical_insert=(mode == "inside"))
    pre_z = float(data.qpos[pit_adr + 2])
    run_phase(sim, action(lift_arm, hold_grip), 3.0, renderer, frames)
    lifted_z = float(data.qpos[pit_adr + 2])
    hang_tilt = pitcher_tilt_deg(model, data)
    hang_spout = spout_angle_deg(model, data)
    slip = float(np.linalg.norm(data.site_xpos[fngr_site] - data.site_xpos[grasp_site]))
    spill_lift = spill_fraction(model, data, task._n_balls)
    print(
        f"  lift: z {pre_z:.3f}->{lifted_z:.3f}, hang tilt {hang_tilt:.1f}, spout {hang_spout:.1f} deg, "
        f"slip {slip:.3f}, spill {spill_lift:.1%}"
    )

    # Wrist-pitch sweep: the pour-attitude authority test. Record the spout angle.
    spout_trace, tilt_trace = [], []
    n = int(round(6.0 / task.dt))
    wr0_0 = lift_arm[4]
    for k in range(n):
        # Triangular sweep: wr0_0 -> +1.2 -> -1.2 -> wr0_0 (covers both directions).
        phase = (k + 1) / n
        if phase < 1 / 3:
            wr0 = wr0_0 + (1.2 - wr0_0) * (phase * 3)
        elif phase < 2 / 3:
            wr0 = 1.2 + (-1.2 - 1.2) * ((phase - 1 / 3) * 3)
        else:
            wr0 = -1.2 + (wr0_0 - (-1.2)) * ((phase - 2 / 3) * 3)
        arm = list(lift_arm)
        arm[4] = wr0
        sim.step(action(arm, hold_grip))
        if k % 3 == 0:
            frames.append(renderer.frame(task.data))
            spout_trace.append(round(spout_angle_deg(model, data), 1))
            tilt_trace.append(round(pitcher_tilt_deg(model, data), 1))
    spill_pour = spill_fraction(model, data, task._n_balls)
    still_held = float(np.linalg.norm(data.site_xpos[fngr_site] - data.site_xpos[grasp_site])) < 0.15
    authority = round(max(spout_trace) - min(spout_trace), 1) if spout_trace else 0.0
    print(f"  sweep: spout {spout_trace[::6]}, AUTHORITY {authority} deg, spill {spill_pour:.1%}, held {still_held}")

    video = save_video(frames, OUT_DIR / f"probe_{mode}_fill{int(args.fill * 100)}.mp4", fps=25)
    report = {
        "mode": mode,
        "fill_ratio": args.fill,
        "resistance": resistance,
        "lift_height_gain_m": lifted_z - pre_z,
        "hang_tilt_deg": hang_tilt,
        "hang_spout_deg": hang_spout,
        "slip_m": slip,
        "spout_authority_deg": authority,
        "spout_trace_deg": spout_trace,
        "spill_after_lift": spill_lift,
        "spill_after_sweep": spill_pour,
        "held_through_sweep": bool(still_held),
        "video": str(video),
    }
    (OUT_DIR / f"probe_{mode}_fill{int(args.fill * 100)}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "spout_trace_deg"}, indent=2))


if __name__ == "__main__":
    main()
