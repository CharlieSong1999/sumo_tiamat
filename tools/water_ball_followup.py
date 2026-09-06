# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Follow-up: CG-solver cost with the full Spot scene, and re-render keyframes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import water_ball_test as wbt

OUT = wbt.OUT_DIR
CG = (
    f'<option timestep="{wbt.TIMESTEP}" solver="CG" iterations="50" ls_iterations="20" '
    'integrator="implicitfast" density="1" jacobian="sparse"/>'
)


def perf_with_spot_cg(counts: list[int]) -> list[dict]:
    base = wbt.OPTION_XML
    rows = []
    try:
        # experiment_perf_with_spot builds its scene from spot_primitive files whose
        # option block comes from default.xml (Newton). Patch the materialized model's
        # option in-place instead: load, then override opt fields before timing.
        for n_balls in counts:
            wbt.OPTION_XML = base
            row_source = wbt.experiment_perf_with_spot([n_balls])[0]  # Newton baseline (already measured)
            # Rebuild with CG by flipping solver options on the compiled model.
            wbt.OPTION_XML = base
            import numpy as np  # noqa: F401

            # Reuse the same builder but flip the solver on the loaded model directly.
            # Simplest: rebuild and mutate model.opt post-compile (solver is runtime-switchable).
            rows_cg = _spot_scene_timed(n_balls)
            rows.append({"newton_ms_per_step": row_source["ms_per_step"], **rows_cg})
    finally:
        wbt.OPTION_XML = base
    return rows


def _spot_scene_timed(n_balls: int) -> dict:
    """Build the Spot+pit scene via the existing helper, then switch solver to CG sparse."""
    from judo import MODEL_PATH as JUDO_MODEL_PATH

    from sumo.tasks.spot.spot_base import SpotBase, _get_spot_menagerie_dir
    from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT

    robot_xml = JUDO_MODEL_PATH / "xml" / "spot_primitive"
    scene = f"""
<mujoco model="spot_pit_water">
  <include file="{(robot_xml / "default.xml").as_posix()}"/>
  <include file="{(robot_xml / "assets.xml").as_posix()}"/>
  <worldbody>
    <geom name="ground" type="plane" size="10 10 0.01" class="collision" priority="5" friction="0.7"/>
    <body name="body" pos="-1.2 0 0.52">
      <include file="{(robot_xml / "body.xml").as_posix()}"/>
      <include file="{(robot_xml / "legs.xml").as_posix()}"/>
      <include file="{(robot_xml / "arm.xml").as_posix()}"/>
    </body>
    {wbt.ball_grid_xml(n_balls)}
  </worldbody>
  <include file="{wbt.PIT_XML.as_posix()}"/>
  <include file="{(robot_xml / "actuator.xml").as_posix()}"/>
  <include file="{(robot_xml / "contact.xml").as_posix()}"/>
</mujoco>
"""
    tmp = OUT / f"spot_pit_cg_{n_balls}.xml"
    tmp.write_text(scene)
    materialized = SpotBase._materialize_model_path(tmp)
    spec = mujoco.MjSpec.from_file(str(materialized))
    menagerie_assets = _get_spot_menagerie_dir() / "assets"
    for mesh in spec.meshes:
        if "spot/meshes/" in mesh.file:
            mesh.file = str(menagerie_assets / Path(mesh.file).name)
    for texture in spec.textures:
        if "spot/textures/" in texture.file:
            texture.file = str(_get_spot_menagerie_dir() / "spot.png")
    model = spec.compile()
    # Flip to CG + sparse at runtime.
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.iterations = 50
    model.opt.ls_iterations = 20
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE

    data = mujoco.MjData(model)
    pit_adr = model.joint("pit_joint").qposadr[0]
    data.qpos[pit_adr + 2] = wbt.PIT_REST_Z
    data.qpos[pit_adr + 3 : pit_adr + 7] = [1, 0, 0, 0]
    base_adr = model.joint("base").qposadr[0]
    data.qpos[base_adr : base_adr + 7] = [-1.2, 0, STANDING_HEIGHT, 1, 0, 0, 0]
    data.qpos[base_adr + 7 : base_adr + 7 + 12] = LEGS_STANDING_POS
    mujoco.mj_forward(model, data)
    wbt.run_steps(model, data, 2.0)
    sps = wbt.run_steps(model, data, 2.0)
    return {
        "scene": "spot_plus_pit_CG_sparse",
        "n_balls": n_balls,
        "ncon": int(data.ncon),
        "steps_per_sec": round(sps),
        "cg_ms_per_step": round(1000.0 / sps, 3),
    }


def rerender_keyframes() -> None:
    """Re-render settle + pour keyframes with the fixed translucent material."""
    model, data = wbt.load(wbt.scene_xml(200, mocap_weld=False))
    cam = wbt.Camera(model)
    wbt.run_steps(model, data, 3.0)
    cam.shot(data, OUT / "settle_200_final.png")

    wbt.experiment_pour(200, render=True)


if __name__ == "__main__":
    results = {}
    print("=== CG + Spot perf ===")
    results["spot_cg"] = perf_with_spot_cg([100, 200])
    for r in results["spot_cg"]:
        print(r)
    (OUT / "metrics_followup.json").write_text(json.dumps(results, indent=2))
    print("=== re-render ===")
    rerender_keyframes()
    print("done")
