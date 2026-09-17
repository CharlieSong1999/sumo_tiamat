"""Research-only water-state reconstruction audit; no production task changes.

The physical plant keeps its continuous state. Only MPC's unobserved water state
is re-created at each tick, like SceneLayout. Robot and jug observations are exact;
this is NOT a full timing/perception/ROS rehearsal. Reuses the fixed-reward runner.
"""

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from sumo.tasks.spot import jug_water
from tools import jug_roll_targets as runner
from tools.jug_budget_study import digest, jsonable
from tools.jug_roll_horizon_study import SEEDS, TARGETS, summarize

ROOT = Path("out/jug_water_reconstruction_20260917")
RADII = {3: 0.045, 5: 0.04, 7: 0.035}
PROFILE = "spot_jug_roll_arm_gentle_coarse"
SOURCES = (
    "tools/jug_roll_water_reconstruction.py",
    "tools/jug_roll_targets.py",
    "tools/jug_budget_study.py",
    "tools/jug_demo.py",
    "sumo/tasks/spot/jug_water.py",
    "sumo/tasks/spot/spot_jug_manipulation.py",
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=jsonable, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fingerprints():
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}


def geometry_metrics(points, radius):
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    distances = distances[np.triu_indices(len(points), 1)]
    return dict(
        min_pair_distance=float(distances.min()) if len(distances) else 2 * radius,
        overlapping_pairs=int(np.sum(distances < 2 * radius - 1e-9)),
        max_overlap=float(np.maximum(2 * radius - distances, 0).max()) if len(distances) else 0,
    )


# The packing lives in the library now (sumo.tasks.spot.jug_water.packed_local_positions,
# adopted 2026-09-17 for the deployed coarse profiles); this name stays for the study.
packed_local_positions = jug_water.packed_local_positions


def reconstruct(task, state, mode):
    """Copy observation; keep measured robot/jug exactly, zero only water velocity."""
    qpos, qvel = state.qpos.copy(), state.qvel.copy()
    obj = task.object_pose_start
    quat = qpos[obj + 3 : obj + 7].copy()
    quat /= np.linalg.norm(quat)
    local = (packed_local_positions if mode == "packed" else jug_water.pooled_local_positions)(
        task.water_count, task.water_radius, quat
    )
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    world = local @ rotation.reshape(3, 3).T + qpos[obj : obj + 3]
    actual = []
    for i, name in enumerate(task.synthesized_joints):
        joint = task.model.joint(name)
        qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
        actual.append(qpos[qa : qa + 3].copy())
        qpos[qa : qa + 7] = [*world[i], 1.0, 0.0, 0.0, 0.0]
        qvel[va : va + 6] = 0
    stats = geometry_metrics(local, task.water_radius)
    stats["water_com_error_m"] = float(np.linalg.norm(world.mean(0) - np.mean(actual, axis=0)))
    return replace(state, qpos=qpos, qvel=qvel), stats


def run_one(root, balls, scenario, seed, mode="packed"):
    out = root / f"{mode}_{balls}" / scenario
    final_path = out / f"seed{seed}.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    original_make = runner.make_system
    diagnostics, initial_info = [], {}
    source_start = fingerprints()

    def make(*args, **kwargs):
        task, controller, plant, initial, settings = original_make(*args, **kwargs)
        assert task.water_count == balls and task.nu == 11
        assert task.water_mass == jug_water.water_parameters(0.1, 0.025)[1]
        assert settings["num_rollouts"] == 32 and settings["horizon"] == 1.5
        original_update = controller.update_states

        def update(state):
            before = time.perf_counter()
            replacement, stats = reconstruct(task, state, mode)
            stats["reconstruction_ms"] = (time.perf_counter() - before) * 1000
            diagnostics.append(stats)
            original_update(replacement)

        controller.update_states = update
        initial_info.update(
            robot_jug_initial_hash=digest(dict(qpos=initial["qpos"][:33], qvel=initial["qvel"][:31])),
            water_count=balls,
            water_radius=task.water_radius,
            water_mass=task.water_mass,
            model_solver=int(task.model.opt.solver),
            physics_dt=task.model.opt.timestep,
            physics_substeps=task.physics_substeps,
            nq=task.model.nq,
            nv=task.model.nv,
        )
        return task, controller, plant, initial, settings

    runner.make_system = make
    try:
        runner.run(out, scenario, seed, False, 32, 1.5, task_name=PROFILE, water_ball_radius=RADII[balls])
    finally:
        runner.make_system = original_make
    record = json.loads(final_path.read_text())
    assert len(diagnostics) == record["planning_updates"] == 600
    assert fingerprints() == source_start, "Shared source changed during episode; invalidate, do not compare"
    record["water_reconstruction"] = dict(
        mode=mode,
        plant="continuous same-count water; NEVER re-pooled",
        **initial_info,
        observations="Exact robot/jug and actual low-level policy history; water replaced every planning tick",
        diagnostic_summary=runner.aggregate_rows(diagnostics),
        overlapping_plan_count=sum(d["overlapping_pairs"] > 0 for d in diagnostics),
        source_sha256=source_start,
    )
    record["protocol"] += " Predictor water positions re-synthesized and velocities zeroed; plant unchanged."
    record["storage"] = "Settings, initial state, scalar summaries/bins only; no rollout/execution archives or video."
    write(final_path, record)
    validate(record, balls, scenario, seed, mode)
    return record


def validate(r, balls, scenario, seed, mode):
    assert r["scenario"] == scenario and r["seed"] == seed
    assert r["steps"] == 1500 and abs(r["duration"] - 30) < 1e-8
    assert r["planning_updates"] == 600 and r["candidate_evaluations"] == 19200
    assert r["predicted_policy_steps_per_plan"] == 2400
    ref = json.loads(Path("tests/data/jug_roll_arm_gentle_profile.json").read_text())
    expected = dict(ref["initial_config"], water_ball_radius=RADII[balls], goal_pos=runner.SCENARIOS[scenario][0])
    assert r["initial_config"] == expected
    assert r["planning_settings"] == ref["planning_settings"]
    meta = r["water_reconstruction"]
    assert meta["water_count"] == balls and meta["mode"] == mode
    if mode == "packed":
        assert meta["overlapping_plan_count"] == 0


def report(root):
    records = []
    initial_hashes, prefix_hashes, hashes = {}, set(), set()
    for mode in ("packed", "legacy"):
        for balls in RADII:
            for scenario in TARGETS:
                for seed in SEEDS:
                    path = root / f"{mode}_{balls}" / scenario / f"seed{seed}.json"
                    if not path.exists():
                        continue
                    r = json.loads(path.read_text())
                    if "water_reconstruction" not in r:
                        continue  # worker has not finalized/validated it yet
                    validate(r, balls, scenario, seed, mode)
                    meta = r["water_reconstruction"]
                    assert initial_hashes.setdefault(balls, r["physical_initial_hash"]) == r["physical_initial_hash"]
                    prefix_hashes.add(meta["robot_jug_initial_hash"])
                    hashes.add(json.dumps(meta["source_sha256"], sort_keys=True))
                    records.append(dict(variant=f"{mode}_{balls}", record=r, path=str(path.relative_to(root))))
    assert len(prefix_hashes) <= 1 and len(hashes) <= 1
    summary = {v: summarize([i for i in records if i["variant"] == v]) for v in sorted({i["variant"] for i in records})}
    by_target = {
        v: {s: summarize([i for i in records if i["variant"] == v and i["record"]["scenario"] == s]) for s in TARGETS}
        for v in summary
    }
    write(root / "comparison.json", dict(completed=len(records), summary=summary, by_target=by_target, records=records))
    rows = []
    for v, s in summary.items():
        rows.append(
            f"<tr><td>{v}</td><td>{s['count']}</td><td>{s['position_end_successes']}</td>"
            f"<td>{s['strict_end_successes']}</td><td>{s['mean_final_distance']:.3f}</td>"
            f"<td>{s['mean_executed_reward']:.2f}</td><td>{s['mean_hand_speed_p95']:.3f}</td>"
            f"<td>{s['max_hand_speed']:.3f}</td><td>{s['max_jug_speed']:.3f}</td><td>{s['falls']}</td></tr>"
        )
    details = []
    for i in records:
        r = i["record"]
        details.append(
            f"<tr><td>{i['variant']}</td><td>{r['scenario']}/{r['seed']}</td>"
            f"<td>{r['stages'][0]['end_position_settled']}/{r['completed_all']}</td>"
            f"<td>{r['final']['goal_distance']:.3f}</td>"
            f"<td>{r['water_reconstruction']['overlapping_plan_count']}</td>"
            f'<td><a href="{i["path"]}">JSON</a></td></tr>'
        )
    curves = []
    for scenario in TARGETS:
        for seed in SEEDS:
            lines = []
            for balls, color in ((3, "#38bdf8"), (5, "#a3e635"), (7, "#fb923c")):
                match = [
                    i
                    for i in records
                    if i["variant"] == f"packed_{balls}"
                    and i["record"]["scenario"] == scenario
                    and i["record"]["seed"] == seed
                ]
                if match:
                    pts = " ".join(
                        f"{35 + (b['start'] + 1) * 10:.1f},{160 - min(b['executed']['distance']['mean'], 2.5) * 56:.1f}"
                        for b in match[0]["record"]["diagnostic_bins"]
                    )
                    lines.append(f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="2"/>')
            curves.append(
                f'<article><h3>{scenario} / seed {seed}</h3><svg viewBox="0 0 365 185">'
                '<path d="M35 15V160H335" stroke="#94a3b8" fill="none"/>'
                '<path d="M35 140.4H335" stroke="#64748b" stroke-dasharray="4 3"/>'
                '<text x="0" y="20">2.5m</text><text x="25" y="180">0s</text><text x="305" y="180">30s</text>'
                + "".join(lines)
                + "</svg></article>"
            )
    analysis = (
        (root / "analysis.html").read_text()
        if (root / "analysis.html").exists()
        else "正在实验，部分结果不代表整体结论。"
    )
    status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>Jug water reconstruction</title>
<style>body{{font:15px system-ui;background:#0f172a;color:#e2e8f0;max-width:1400px;margin:30px auto;padding:0 20px}}
a{{color:#7dd3fc}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #334155}}
section,article{{background:#172033;padding:15px;margin:15px 0;border-radius:9px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
svg{{width:100%}}svg text{{fill:#cbd5e1;font-size:12px}}pre{{white-space:pre-wrap}}</style>
<h1>3 / 5 / 7 水球：每次重建预测初态</h1><p>完成 {len(records)}；状态 {html.escape(str(status))}</p>
<section>{analysis}</section><p>固定 reward、32×1 / 1.5s、30s、四方向×3 seed；每组模型水量1.903kg。
packed=实验性无重叠摆放；legacy=原部署摆放。执行水球连续演化，只替换 MPC 水状态；并非完整 ROS/感知/延迟 rehearsal。
不同球数使用各自 plant，不是同一个真实流体的 ground truth。耗时来自并发离线实验，不能证明实机20Hz。</p>
<table><tr><th>组</th><th>例数</th><th>稳定到位</th><th>严格roll</th><th>终点m</th><th>平均执行reward</th>
<th>手速P95均值</th><th>手峰速</th><th>桶峰速</th><th>跌倒</th></tr>{"".join(rows)}</table>
<h2>目标距离（2s 分箱均值，虚线0.35m）</h2><p>蓝=3球，绿=5球，橙=7球</p><div class="grid">{"".join(curves)}</div>
<h2>逐例</h2><table><tr><th>组</th><th>目标/seed</th><th>到位/roll</th><th>误差m</th><th>初态重叠计划数</th><th>记录</th></tr>{"".join(details)}</table>
<p><a href="comparison.json">汇总JSON</a> · <a href="passive.json">静置诊断</a> · <a href="manifest.json">协议/指纹</a></p></html>"""
    (root / "index.html").write_text(page)
    return summary


def batch(root, jobs):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = dict(
        source_sha256=fingerprints(),
        radii=RADII,
        mpc="32x1 H1.5",
        seconds=30,
        trials="packed 3/5/7 x four targets x seed0/1/2; legacy 5/7 front seed0/1/2",
        mode="reconstruct predicted water only; same-count continuous plant; zero reconstructed water velocities",
    )
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["source_sha256"] == manifest["source_sha256"]
    else:
        write(manifest_path, manifest)
    specs = [("packed", b, s, seed) for s in TARGETS for b in RADII for seed in SEEDS]
    specs += [("legacy", b, "front", seed) for b in (5, 7) for seed in SEEDS]

    def worker(spec):
        mode, balls, scenario, seed = spec
        folder = root / f"{mode}_{balls}" / scenario
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"seed{seed}.json"
        if path.exists():
            validate(json.loads(path.read_text()), balls, scenario, seed, mode)
            return spec
        cmd = [
            sys.executable,
            "-m",
            "tools.jug_roll_water_reconstruction",
            "one",
            "--root",
            str(root),
            "--balls",
            str(balls),
            "--scenario",
            scenario,
            "--seed",
            str(seed),
            "--mode",
            mode,
        ]
        with (folder / f"seed{seed}.log").open("w") as log:
            subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="egl"),
            )
        validate(json.loads(path.read_text()), balls, scenario, seed, mode)
        return spec

    write(root / "status.json", dict(state="running", expected=len(specs), jobs=jobs))
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            pending = {pool.submit(worker, spec) for spec in specs[:jobs]}
            cursor = jobs
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    print("finished", future.result(), flush=True)
                    report(root)
                    if cursor < len(specs):
                        pending.add(pool.submit(worker, specs[cursor]))
                        cursor += 1
        write(root / "status.json", dict(state="complete", expected=len(specs)))
    except Exception as exc:
        write(root / "status.json", dict(state="invalid_queue_stopped", error=str(exc)))
        raise
    finally:
        report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("one", "batch", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--balls", type=int, choices=RADII, default=7)
    parser.add_argument("--scenario", choices=TARGETS, default="front")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("packed", "legacy"), default="packed")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.command == "one":
        run_one(args.root, args.balls, args.scenario, args.seed, args.mode)
    elif args.command == "batch":
        batch(args.root, args.jobs)
    else:
        print(json.dumps(report(args.root), indent=2))


if __name__ == "__main__":
    main()
