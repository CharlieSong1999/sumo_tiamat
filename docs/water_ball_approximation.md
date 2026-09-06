# Water-in-container approximation with spheres ("ball water")

Feasibility study for simulating liquid in SUMO by filling a container with small
frictionless spheres, targeting future Spot tasks like *carry water without spilling*
and *pour water* (advisor agenda: "Water in bucket approximation", "Pour task",
"Modeling more complex tasks in SUMO: water in a bucket (balls)").

Reproduce with:

```bash
MUJOCO_GL=egl pixi run python tools/water_ball_test.py            # full suite
MUJOCO_GL=egl pixi run python tools/water_ball_followup.py        # CG-with-Spot numbers
```

Outputs (metrics JSON, keyframe PNGs, pour video) land in `out/water_ball_test/`.

## The container: `objects/pit/pit.xml`

The mesh in `sumo/models/meshes/pit/` ("Meshy_AI_Clear_rectangular_pit…", 574k faces)
is a **clear rectangular pitcher with a vertical loop handle and a pour spout** (the
raw mesh is Y-up, lying on its side in mesh coordinates). New object files follow the
bucket convention (visual mesh + primitive collision + standard sites):

- `sumo/models/xml/objects/pit/pit_defs.xml` — assets + `visual_pit` / `collision_pit`
  / `water_ball` defaults classes.
- `sumo/models/xml/objects/pit/pit.xml` — free-joint body: translucent visual mesh,
  floor box + 4 wall boxes (20 mm thick → a ball needs >3.5 m/s to tunnel in one 10 ms
  step), 3 handle capsules (outer bar r=8 mm = pinch-graspable, same as the bucket),
  `site_object` / grasp / approach sites.

At `scale=0.2` the pitcher is 0.40 m tall with a 0.21×0.13×0.38 m cavity (~10 L).
Body origin = center of the interior floor; free-joint rest height on flat ground is
`z=0.02`. The handle bar is the grasp target for a future pour task.

## The water: `water_ball` defaults class

Spheres r=15 mm, `condim=1` (frictionless — the pile shears and levels like liquid),
`density=1560` (≈ water bulk density 1000 kg/m³ at random-close-packing ≈0.64),
`priority=7` so ball contacts keep the frictionless model against the container
(priority 6) and ground (priority 5). 200 balls ≈ 4.4 L water-equivalent (~60% fill).

## Results (settled scenes, dt=10 ms as in the Spot tasks; single-thread `mj_step`)

Behavior (200 balls, Newton): settles dead (max speed 5e-5 m/s at 3 s), level surface
(top-layer height σ = 8 mm ≈ r/2), **zero escapes** through walls in settle + 6 cm/1.5 Hz
lateral shake. Pour (mocap-weld wrist tilt about the spout rim): onset 83°, 100% out by
~115°, 85% of poured balls exit on the spout side; the sharp late onset matches the
tall-narrow pitcher shape at 60% fill. Frictionless balls scatter far on open ground
after pouring (water-puddle-like; a receiving container catches them normally).

Cost — pitcher-only scene:

| balls | contacts | Newton (default) | CG+sparse (iter=50) |
|---|---|---|---|
| 50 | 167 | 0.36 ms/step (28× RT) | — |
| 100 | 384 | 4.4 ms/step (2.3× RT) | — |
| 200 | 803 | 12.2 ms/step (0.8× RT) | **0.69 ms/step (14.5× RT)** |
| 300 | 1269 | 25.2 ms/step (0.4× RT) | — |

Cost — full Spot robot + pitcher scene (the MPC-rollout-relevant number):

| balls | Newton | CG+sparse |
|---|---|---|
| 100 | 11.0 ms/step | **0.30 ms/step (36×)** |
| 200 | 65.9 ms/step | **0.71 ms/step (93×)** |

Same scene at the water_sim experiment fill ratios (Newton, settled, single-thread
`mj_step`; the 0-ball row IS the planner model of the dual-model tasks):

| balls (fill) | contacts | ms/step | vs 0 balls |
|---|---|---|---|
| 0 (planner model) | 9 | **0.05** | 1× |
| 98 (30%) | 358 | 11.1 | 222× |
| 163 (50%) | 628 | 40.8 | 816× |
| 277 (85%) | 1165 | 148.1 | 2962× |

Per MPC plan iteration (24 rollouts × 2.0 s horizon = 4800 physics steps of 10 ms,
physics cost only, single-thread): no-ball planner ≈ 0.24 s (≈10 ms per rollout when
the 24 rollouts run on parallel threads — fits the 50 ms/20 Hz replanning budget);
with balls in the planner: 53 s (30%), 196 s (50%), 711 s (85%) — 3-4 orders of
magnitude over budget even before thread-parallelism. This is why the dual-model
(planner without balls / plant with balls) design is not an optimization but a
prerequisite for the water tasks to run at all.

Newton's cost explodes because all balls + container form one contact island and the
dense factorization scales superlinearly with active constraints; `jacobian="sparse"`
alone and the island flag change nothing (one island). Switching the **solver to CG
with capped iterations** is the lever. Residual jitter under CG: max ball speed ~8 mm/s
(vs ~1e-5 Newton) — irrelevant for water semantics. Balls stay contained under CG.

An equal-water-volume "chunky water" variant (43 balls, r=25 mm) keeps Newton cheap
(0.43 ms/step, 23× RT) at the cost of coarser slosh granularity.

## Recommendations

1. **Simulation plant** (the interactive sim node): 200 balls + CG solver runs at 14×
   realtime — full water is affordable there.
2. **MPC rollouts**: per plan iteration SUMO rolls 24 × 250 steps (horizon 2.5 s at
   dt 10 ms). With CG+sparse, 200 balls ≈ 4.3 s single-thread per iteration (~0.3–0.5 s
   across cores) — borderline; 100 balls or 43 chunky balls is comfortably feasible.
   Validate that the CG solver does not degrade the Spot locomotion backend before
   adopting it for tasks (the ONNX policy rollouts were tuned under Newton).
3. **Recommended experiment design for a pour/carry task**: keep the *planning model*
   water-free (or chunky-water) while the *plant* simulates full water — deliberate
   model mismatch that directly instantiates the advisor's "COM changes during the
   action / do we need robust MPC" questions. Success metric for pour: fraction of
   balls transferred into the target container (machine-checkable, reward-independent).

## Known limitations

- The visual pour-spout lip is not in the collision model (balls pour over a straight
  rim); the spout mainly matters visually and for pour-direction realism at high tilt.
- The bundled PNG texture renders too dark in the default lighting, so `pit_material`
  is a plain translucent rgba; the fill level is visible through the walls.
- Collision walls are vertical (the visual cavity tapers ~1 cm wider toward the rim),
  so near the rim balls rest ~5–9 mm inside the visual wall.
- `condim=1` water never damps laterally through friction; if a task needs "syrup",
  raise `condim`/friction or add free-joint damping to the `water_ball` class.
