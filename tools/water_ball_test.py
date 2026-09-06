# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Water-in-container approximation test: fill the `pit` pitcher with sphere "water balls".

Experiments (all headless, EGL rendering for keyframes):
  A. settle    -- drop a grid of balls into the pitcher, check stability + containment
                  + free-surface flatness.
  B. shake     -- weld the pitcher to a mocap anchor and oscillate it laterally; verify
                  balls slosh but never tunnel through the walls.
  C. pour      -- mocap-tilt the pitcher to 130 deg about the spout rim; record the
                  fraction of balls poured out vs tilt angle (the "pour curve").
  D. perf      -- wall-clock physics cost vs ball count, pitcher-only and with the full
                  Spot robot model in the scene; translate to MPC planning-rate impact.

Run from the repo root:
    pixi run python tools/water_ball_test.py [--quick]

Outputs go to out/water_ball_test/ (metrics.json, keyframe PNGs, pour.mp4/gif).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PIT_XML = REPO_ROOT / "sumo" / "models" / "xml" / "objects" / "pit" / "pit.xml"
OUT_DIR = REPO_ROOT / "out" / "water_ball_test"

# Interior cavity of the pit body frame (must match pit.xml collision geometry).
CAVITY_HX, CAVITY_HY, CAVITY_TOP = 0.105, 0.066, 0.38
PIT_REST_Z = 0.02  # free-joint origin height when resting on flat ground
BALL_RADIUS = 0.015

TIMESTEP = 0.01  # matches spot_components/default.xml
OPTION_XML = f'<option timestep="{TIMESTEP}" solver="Newton" integrator="implicitfast" density="1"/>'


def ball_grid_xml(n_balls: int, radius: float = BALL_RADIUS, drop_z: float = 0.05) -> str:
    """Bodies for `n_balls` spheres in a jittered grid over the cavity opening."""
    margin = radius + 0.004
    xs = np.arange(-CAVITY_HX + margin, CAVITY_HX - margin + 1e-9, 2 * radius + 0.003)
    ys = np.arange(-CAVITY_HY + margin, CAVITY_HY - margin + 1e-9, 2 * radius + 0.003)
    rng = np.random.default_rng(0)
    bodies = []
    i = 0
    z = drop_z
    while i < n_balls:
        for x in xs:
            for y in ys:
                if i >= n_balls:
                    break
                jx, jy = rng.uniform(-1e-3, 1e-3, 2)
                bodies.append(
                    f'<body name="ball_{i}" pos="{x + jx:.4f} {y + jy:.4f} {z:.4f}">'
                    f'<freejoint name="ball_{i}_joint"/><geom name="ball_{i}_geom" class="water_ball" size="{radius}"/>'
                    "</body>"
                )
                i += 1
            if i >= n_balls:
                break
        z += 2 * radius + 0.004
    return "\n".join(bodies)


def scene_xml(n_balls: int, mocap_weld: bool, radius: float = BALL_RADIUS, option_xml: str = OPTION_XML) -> str:
    """Standalone scene: ground + pit (+ optional mocap weld) + water balls."""
    weld_body = f'<body name="pit_anchor" mocap="true" pos="0 0 {PIT_REST_Z}"/>' if mocap_weld else ""
    weld_eq = (
        '<equality><weld name="pit_weld" body1="pit_anchor" body2="pit" solref="0.01 1"/></equality>'
        if mocap_weld
        else ""
    )
    return f"""
<mujoco model="water_ball_test">
  {option_xml}
  <include file="{PIT_XML.as_posix()}"/>
  <worldbody>
    <light directional="true" diffuse=".7 .7 .7" pos="0 0 4" dir="0 0 -1"/>
    <light directional="true" diffuse=".3 .3 .3" pos="2 2 4" dir="-0.5 -0.5 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" friction="0.7" priority="5" rgba="0.85 0.85 0.88 1"/>
    {weld_body}
    {ball_grid_xml(n_balls, radius=radius)}
  </worldbody>
  {weld_eq}
</mujoco>
"""


def load(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # Rest the pit on the ground (free joint qpos is [x y z qw qx qy qz]).
    pit_adr = model.joint("pit_joint").qposadr[0]
    data.qpos[pit_adr + 2] = PIT_REST_Z
    data.qpos[pit_adr + 3 : pit_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    return model, data


def ball_positions(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    ids = [model.body(f"ball_{i}").id for i in range(count_balls(model))]
    return data.xpos[ids]


def ball_speeds(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    n = count_balls(model)
    speeds = np.empty(n)
    for i in range(n):
        adr = model.joint(f"ball_{i}_joint").dofadr[0]
        speeds[i] = np.linalg.norm(data.qvel[adr : adr + 3])
    return speeds


def count_balls(model: mujoco.MjModel) -> int:
    n = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ball_{n}") >= 0:
        n += 1
    return n


def pit_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    bid = model.body("pit").id
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def containment(model: mujoco.MjModel, data: mujoco.MjData, slack: float = 0.02) -> dict:
    """Fraction of balls inside the cavity, measured in the pit's local frame."""
    pos, rot = pit_pose(model, data)
    local = (ball_positions(model, data) - pos) @ rot  # world -> pit frame
    inside = (
        (np.abs(local[:, 0]) < CAVITY_HX + slack)
        & (np.abs(local[:, 1]) < CAVITY_HY + slack)
        & (local[:, 2] > -slack)
        & (local[:, 2] < CAVITY_TOP + 6 * BALL_RADIUS)  # allow heap above rim while filling
    )
    # tunneled = below the floor or inside a wall band but below rim (escape must be over the rim)
    escaped_low = local[:, 2] < -slack
    return {"inside_frac": float(inside.mean()), "escaped_below_floor": int(escaped_low.sum())}


def surface_flatness(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Std-dev (m) of the top-layer ball heights in the pit frame: a level water surface -> small."""
    pos, rot = pit_pose(model, data)
    local = (ball_positions(model, data) - pos) @ rot
    z = local[:, 2]
    top = z >= np.quantile(z, 0.85)
    return float(z[top].std())


def run_steps(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> float:
    n = int(round(seconds / model.opt.timestep))
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, data)
    return n / (time.perf_counter() - t0)


class Camera:
    def __init__(self, model: mujoco.MjModel, w: int = 640, h: int = 480):
        self.renderer = mujoco.Renderer(model, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = [0, 0, 0.25]
        self.cam.distance = 1.3
        self.cam.azimuth = 135
        self.cam.elevation = -20

    def shot(self, data: mujoco.MjData, path: Path) -> np.ndarray:
        self.renderer.update_scene(data, self.cam)
        img = self.renderer.render()
        try:
            import PIL.Image

            PIL.Image.fromarray(img).save(path)
        except ImportError:
            pass
        return img


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------
def experiment_settle(n_balls: int, render: bool) -> dict:
    model, data = load(scene_xml(n_balls, mocap_weld=False))
    cam = Camera(model) if render else None
    if cam:
        cam.shot(data, OUT_DIR / f"settle_{n_balls}_t0.png")
    sps = run_steps(model, data, 3.0)
    speeds = ball_speeds(model, data)
    metrics = {
        "n_balls": n_balls,
        "steps_per_sec": round(sps),
        "realtime_factor": round(sps * TIMESTEP, 2),
        "mean_speed_after_3s": float(speeds.mean()),
        "max_speed_after_3s": float(speeds.max()),
        "surface_std_m": round(surface_flatness(model, data), 4),
        "ncon_settled": int(data.ncon),
        **containment(model, data),
    }
    if cam:
        cam.shot(data, OUT_DIR / f"settle_{n_balls}_t3.png")
    return metrics


def experiment_shake(n_balls: int, render: bool, amp: float = 0.06, freq: float = 1.5) -> dict:
    model, data = load(scene_xml(n_balls, mocap_weld=True))
    cam = Camera(model) if render else None
    run_steps(model, data, 2.0)  # settle first
    n = int(round(3.0 / TIMESTEP))
    t0 = time.perf_counter()
    max_speed = 0.0
    for k in range(n):
        t = k * TIMESTEP
        data.mocap_pos[0] = [amp * np.sin(2 * np.pi * freq * t), 0, PIT_REST_Z]
        mujoco.mj_step(model, data)
        if k % 25 == 0:
            max_speed = max(max_speed, float(ball_speeds(model, data).max()))
    sps = n / (time.perf_counter() - t0)
    if cam:
        cam.shot(data, OUT_DIR / f"shake_{n_balls}_end.png")
    run_steps(model, data, 1.0)  # re-settle
    return {
        "n_balls": n_balls,
        "amp_m": amp,
        "freq_hz": freq,
        "steps_per_sec": round(sps),
        "max_ball_speed_during_shake": round(max_speed, 2),
        **containment(model, data),
    }


def experiment_pour(n_balls: int, render: bool, max_angle_deg: float = 130.0, duration: float = 8.0) -> dict:
    model, data = load(scene_xml(n_balls, mocap_weld=True))
    cam = Camera(model) if render else None
    run_steps(model, data, 2.0)  # settle

    # Tilt about the world y-axis so the -x side (spout) dips (negative rotation about +y
    # maps -x downward); pivot at the spout rim so the motion resembles a wrist pour
    # instead of orbiting the container.
    pivot_local = np.array([-CAVITY_HX, 0.0, CAVITY_TOP])  # spout rim in pit frame
    anchor0 = np.array([0.0, 0.0, PIT_REST_Z])
    n = int(round(duration / TIMESTEP))
    angles, out_frac = [], []
    frames = []
    frame_every = max(1, n // 120)
    for k in range(n):
        ang = np.deg2rad(max_angle_deg) * (k + 1) / n
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), -ang)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        # Keep the spout rim fixed in space: anchor = rim - R @ pivot_local
        rim_world = anchor0 + pivot_local
        data.mocap_pos[0] = rim_world - rot @ pivot_local
        data.mocap_quat[0] = quat
        mujoco.mj_step(model, data)
        if k % 10 == 0:
            pos, rotm = pit_pose(model, data)
            local = (ball_positions(model, data) - pos) @ rotm
            inside = (
                (np.abs(local[:, 0]) < CAVITY_HX + 0.02)
                & (np.abs(local[:, 1]) < CAVITY_HY + 0.02)
                & (local[:, 2] > -0.02)
                & (local[:, 2] < CAVITY_TOP + 0.02)
            )
            angles.append(np.rad2deg(ang))
            out_frac.append(float(1.0 - inside.mean()))
        if cam and k % frame_every == 0:
            frames.append(cam.shot(data, OUT_DIR / "pour_frame_tmp.png"))
    run_steps(model, data, 1.5)  # let poured balls come to rest
    if cam:
        cam.shot(data, OUT_DIR / f"pour_{n_balls}_end.png")
        _save_video(frames, OUT_DIR / f"pour_{n_balls}.mp4")

    # Poured balls should exit over the spout rim (held fixed at world x=-CAVITY_HX)
    # and land on the -x side of the anchor.
    world = ball_positions(model, data)
    poured = world[world[:, 2] < 0.05]
    spout_side_frac = float((poured[:, 0] < 0.0).mean()) if len(poured) else float("nan")
    landing = (
        {"mean_x": round(float(poured[:, 0].mean()), 3), "mean_y": round(float(poured[:, 1].mean()), 3)}
        if len(poured)
        else {}
    )
    stride = max(1, len(angles) // 14)
    curve = {f"{a:.0f}": round(f, 3) for a, f in zip(angles[::stride], out_frac[::stride], strict=True)}
    onset = next((a for a, f in zip(angles, out_frac, strict=True) if f > 0.05), None)
    return {
        "n_balls": n_balls,
        "pour_onset_deg": None if onset is None else round(onset, 1),
        "final_out_frac": round(out_frac[-1], 3),
        "poured_balls_on_spout_side_frac": spout_side_frac,
        "poured_landing_mean": landing,
        "pour_curve_angle_to_outfrac": curve,
    }


def _save_video(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=15, macro_block_size=1)
    except Exception:
        try:
            import PIL.Image

            imgs = [PIL.Image.fromarray(f) for f in frames]
            imgs[0].save(path.with_suffix(".gif"), save_all=True, append_images=imgs[1:], duration=66, loop=0)
        except Exception:
            pass


def experiment_perf(ball_counts: list[int]) -> list[dict]:
    """Physics cost vs ball count, pitcher-only scene (settled steady state)."""
    rows = []
    for n_balls in ball_counts:
        model, data = load(scene_xml(n_balls, mocap_weld=False))
        run_steps(model, data, 2.0)  # settle to steady contact count
        sps = run_steps(model, data, 2.0)
        rows.append(
            {
                "scene": "pit_only",
                "n_balls": n_balls,
                "ncon": int(data.ncon),
                "steps_per_sec": round(sps),
                "realtime_factor": round(sps * TIMESTEP, 1),
                "ms_per_step": round(1000.0 / sps, 3),
            }
        )
    return rows


SOLVER_VARIANTS: dict[str, str] = {
    "newton_dense_default": f'<option timestep="{TIMESTEP}" solver="Newton" integrator="implicitfast" density="1"/>',
    "newton_sparse": f'<option timestep="{TIMESTEP}" solver="Newton" integrator="implicitfast" density="1" jacobian="sparse"/>',
    "newton_sparse_island": (
        f'<option timestep="{TIMESTEP}" solver="Newton" integrator="implicitfast" density="1" jacobian="sparse">'
        '<flag island="enable"/></option>'
    ),
    "cg_sparse_iter50": (
        f'<option timestep="{TIMESTEP}" solver="CG" iterations="50" ls_iterations="20" '
        'integrator="implicitfast" density="1" jacobian="sparse"/>'
    ),
}


def experiment_solver_variants(n_balls: int) -> list[dict]:
    """Cost of the same settled scene under different solver configurations.

    This is the lever that decides whether ball-water is affordable inside MPC rollouts.
    """
    rows = []
    for name, option in SOLVER_VARIANTS.items():
        try:
            model, data = load(scene_xml(n_balls, mocap_weld=False, option_xml=option))
        except Exception as exc:
            rows.append({"variant": name, "error": repr(exc)})
            continue
        run_steps(model, data, 2.0)
        sps = run_steps(model, data, 2.0)
        speeds = ball_speeds(model, data)
        rows.append(
            {
                "variant": name,
                "n_balls": n_balls,
                "ncon": int(data.ncon),
                "steps_per_sec": round(sps),
                "realtime_factor": round(sps * TIMESTEP, 1),
                "max_ball_speed": float(speeds.max()),
                **containment(model, data),
            }
        )
    return rows


def experiment_big_balls(render: bool) -> dict:
    """Same water volume with fewer, larger balls (r=25 mm): the cheap 'chunky water'.

    200 balls at r=15 mm and 43 balls at r=25 mm carry the same solid volume (and the
    same bulk water mass at equal packing fraction).
    """
    radius = 0.025
    n_balls = int(round(200 * (0.015 / radius) ** 3))
    model, data = load(scene_xml(n_balls, mocap_weld=False, radius=radius))
    cam = Camera(model) if render else None
    run_steps(model, data, 2.0)
    sps = run_steps(model, data, 2.0)
    speeds = ball_speeds(model, data)
    if cam:
        cam.shot(data, OUT_DIR / f"bigballs_{n_balls}_settled.png")
    return {
        "radius_m": radius,
        "n_balls": n_balls,
        "ncon": int(data.ncon),
        "steps_per_sec": round(sps),
        "realtime_factor": round(sps * TIMESTEP, 1),
        "max_ball_speed": float(speeds.max()),
        "surface_std_m": round(surface_flatness(model, data), 4),
        **containment(model, data),
    }


def experiment_perf_with_spot(ball_counts: list[int]) -> list[dict]:
    """Same cost measurement with the full Spot robot model composed into the scene."""
    from judo import MODEL_PATH as JUDO_MODEL_PATH  # noqa: PLC0415

    from sumo.tasks.spot.spot_base import SpotBase  # noqa: PLC0415

    robot_xml = JUDO_MODEL_PATH / "xml" / "spot_primitive"
    rows = []
    for n_balls in ball_counts:
        scene = f"""
<mujoco model="spot_pit_water">
  <include file="{(robot_xml / "default.xml").as_posix()}"/>
  <include file="{(robot_xml / "assets.xml").as_posix()}"/>
  <worldbody>
    <light directional="true" diffuse=".6 .6 .6" pos="0 0 4" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="10 10 0.01" class="collision" priority="5" friction="0.7"/>
    <body name="body" pos="-1.2 0 0.52">
      <include file="{(robot_xml / "body.xml").as_posix()}"/>
      <include file="{(robot_xml / "legs.xml").as_posix()}"/>
      <include file="{(robot_xml / "arm.xml").as_posix()}"/>
    </body>
    {ball_grid_xml(n_balls)}
  </worldbody>
  <include file="{PIT_XML.as_posix()}"/>
  <include file="{(robot_xml / "actuator.xml").as_posix()}"/>
  <include file="{(robot_xml / "contact.xml").as_posix()}"/>
</mujoco>
"""
        # Reuse the repo's include/mesh resolution machinery, then load the plain model.
        materialized = (
            SpotBase._materialize_model_path_public(scene)
            if hasattr(SpotBase, "_materialize_model_path_public")
            else None
        )
        if materialized is None:
            tmp = OUT_DIR / f"spot_pit_{n_balls}.xml"
            tmp.write_text(scene)
            materialized = SpotBase._materialize_model_path(tmp)
        spec = mujoco.MjSpec.from_file(str(materialized))

        # Spot mesh assets resolve against the menagerie checkout (same fix as SpotAssetMixin).
        from sumo.tasks.spot.spot_base import _get_spot_menagerie_dir  # noqa: PLC0415

        menagerie_assets = _get_spot_menagerie_dir() / "assets"
        for mesh in spec.meshes:
            if "spot/meshes/" in mesh.file:
                mesh.file = str(menagerie_assets / Path(mesh.file).name)
        for texture in spec.textures:
            if "spot/textures/" in texture.file:
                texture.file = str(_get_spot_menagerie_dir() / "spot.png")
        model = spec.compile()
        data = mujoco.MjData(model)
        pit_adr = model.joint("pit_joint").qposadr[0]
        data.qpos[pit_adr + 2] = PIT_REST_Z
        data.qpos[pit_adr + 3 : pit_adr + 7] = [1, 0, 0, 0]
        # Stand Spot at its nominal standing configuration so contact count is realistic.
        from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT  # noqa: PLC0415

        base_adr = model.joint("base").qposadr[0]
        data.qpos[base_adr : base_adr + 7] = [-1.2, 0, STANDING_HEIGHT, 1, 0, 0, 0]
        data.qpos[base_adr + 7 : base_adr + 7 + 12] = LEGS_STANDING_POS
        mujoco.mj_forward(model, data)
        run_steps(model, data, 2.0)
        sps = run_steps(model, data, 2.0)
        rows.append(
            {
                "scene": "spot_plus_pit",
                "n_balls": n_balls,
                "ncon": int(data.ncon),
                "steps_per_sec": round(sps),
                "realtime_factor": round(sps * TIMESTEP, 1),
                "ms_per_step": round(1000.0 / sps, 3),
                "mpc_note": "24 rollouts x 250 steps (horizon 2.5 s) per plan iteration",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer ball counts, no video")
    parser.add_argument("--balls", type=int, default=200, help="ball count for behavior tests")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render = not args.quick
    counts = [50, 200] if args.quick else [50, 100, 200, 300]

    results: dict = {"ball_radius_m": BALL_RADIUS, "timestep_s": TIMESTEP}
    print("=== A. settle ===")
    results["settle"] = experiment_settle(args.balls, render)
    print(json.dumps(results["settle"], indent=2))

    print("=== B. shake ===")
    results["shake"] = experiment_shake(args.balls, render)
    print(json.dumps(results["shake"], indent=2))

    print("=== C. pour ===")
    results["pour"] = experiment_pour(args.balls, render)
    print(json.dumps(results["pour"], indent=2))

    print("=== D. perf (pit only) ===")
    results["perf_pit_only"] = experiment_perf(counts)
    for row in results["perf_pit_only"]:
        print(row)

    print("=== E. solver variants (200 balls) ===")
    results["solver_variants"] = experiment_solver_variants(200)
    for row in results["solver_variants"]:
        print(row)

    print("=== F. big balls (same water volume, r=25mm) ===")
    results["big_balls"] = experiment_big_balls(render)
    print(json.dumps(results["big_balls"], indent=2))

    print("=== D2. perf (with Spot) ===")
    try:
        results["perf_with_spot"] = experiment_perf_with_spot(counts)
        for row in results["perf_with_spot"]:
            print(row)
    except Exception as exc:  # robot assets may be missing outside the pixi env
        results["perf_with_spot_error"] = repr(exc)
        print(f"skipped: {exc!r}")

    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
