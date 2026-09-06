import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.spot.spot_pit_base import (
    BALLS_PER_FULL_CAVITY,
    apply_rigid_water_mass,
    fill_ratio_to_ball_count,
    project_state,
)
from sumo.tasks.spot.spot_pit_move import SpotPitMove, SpotPitMovePlant
from sumo.tasks.spot.spot_pit_pour import SpotPitPour

FILL = 0.3  # small fill keeps the plant model cheap in CI


def test_pit_tasks_registered():
    registered = get_registered_tasks()
    for name in ("spot_pit_move", "spot_pit_move_plant", "spot_pit_pour", "spot_pit_pour_plant"):
        assert name in registered


def test_fill_ratio_mapping():
    assert fill_ratio_to_ball_count(0.0) == 0
    assert fill_ratio_to_ball_count(1.0) == BALLS_PER_FULL_CAVITY
    assert fill_ratio_to_ball_count(0.5) == round(0.5 * BALLS_PER_FULL_CAVITY)
    with pytest.raises(ValueError):
        fill_ratio_to_ball_count(1.5)


@pytest.fixture(scope="module")
def planner_task():
    return SpotPitMove(fill_ratio=FILL, planner_water_model="empty")


@pytest.fixture(scope="module")
def plant_task():
    return SpotPitMovePlant(fill_ratio=FILL)


def test_planner_plant_prefix_identical(planner_task, plant_task):
    """Ball DOFs must sit strictly AFTER the shared robot/pitcher prefix."""
    nj = planner_task.model.njnt
    planner_names = [planner_task.model.joint(i).name for i in range(nj)]
    plant_names = [plant_task.model.joint(i).name for i in range(nj)]
    assert planner_names == plant_names
    n_balls = plant_task._n_balls
    assert plant_task.model.nq == planner_task.model.nq + 7 * n_balls
    assert plant_task.model.nv == planner_task.model.nv + 6 * n_balls


def test_state_projection_shapes(planner_task, plant_task):
    qpos = np.arange(plant_task.model.nq, dtype=float)
    qvel = np.arange(plant_task.model.nv, dtype=float)
    qp, qv = project_state(qpos, qvel, planner_task.model.nq, planner_task.model.nv)
    assert qp.shape == (planner_task.model.nq,)
    assert qv.shape == (planner_task.model.nv,)
    np.testing.assert_array_equal(qp, qpos[: planner_task.model.nq])


def test_rigid_mass_variant_heavier():
    empty = SpotPitMove(fill_ratio=FILL, planner_water_model="empty")
    rigid = SpotPitMove(fill_ratio=FILL, planner_water_model="rigid_mass")
    m_empty = float(empty.model.body("pit").mass[0])
    m_rigid = float(rigid.model.body("pit").mass[0])
    assert m_rigid > m_empty
    # ~22 g per ball
    assert m_rigid - m_empty == pytest.approx(0.022 * empty._n_balls, rel=0.1)


def test_plant_reset_pose_matches_nq(plant_task):
    assert len(plant_task.reset_pose) == plant_task.model.nq


@pytest.mark.parametrize("task_cls", [SpotPitMove, SpotPitPour])
def test_reward_shape(task_cls):
    task = task_cls(fill_ratio=FILL)
    batch, horizon = 4, 6
    rng = np.random.default_rng(0)
    states = rng.normal(size=(batch, horizon, task.model.nq + task.model.nv))
    sensors = rng.normal(size=(batch, horizon, task.model.nsensordata))
    controls = rng.normal(size=(batch, horizon, task.nu))
    r = task.reward(states, sensors, controls)
    assert r.shape == (batch,)
    assert np.isfinite(r).all()


def test_rigid_water_mass_math():
    """COM must sit between the empty pitcher's COM and the water block COM."""
    import mujoco

    from sumo.tasks.spot.spot_pit_base import PIT_XML_PATH

    spec = mujoco.MjSpec.from_file(str(PIT_XML_PATH))
    com_before = np.array(spec.body("pit").ipos)
    apply_rigid_water_mass(spec, 163)
    com_after = np.array(spec.body("pit").ipos)
    assert spec.body("pit").mass > 1.0
    assert com_after[2] < com_before[2]  # water (~0.11 m) sits below the empty COM (0.15 m)


def test_chunky_planner_variant():
    task = SpotPitMove(fill_ratio=FILL, planner_water_model="chunky", chunky_radius=0.025)
    assert task._n_chunky > 0
    assert task.model.nq == 33 + 7 * task._n_chunky
    assert len(task.reset_pose) == task.model.nq
    # Mass conservation vs the fine-ball water within one ball quantum.
    from sumo.tasks.spot.spot_pit_base import ball_mass

    fine_mass = task._n_balls * ball_mass()
    chunky_mass = task._n_chunky * ball_mass(0.025)
    assert abs(chunky_mass - fine_mass) < ball_mass(0.025)


def test_pit_scale_variant():
    """Scaled pitcher: mass ~ scale^3, water count ~ scale^3, grip cross-section unchanged."""
    s = 0.5
    task = SpotPitMove(fill_ratio=FILL, planner_water_model="empty", handle_style="cylinder", pit_scale=s)
    full = SpotPitMove(fill_ratio=FILL, planner_water_model="empty")
    assert float(task.model.body("pit").mass[0]) == pytest.approx(1.0 * s**3, rel=1e-6)
    assert task._n_balls == round(full._n_balls * s**3) or abs(task._n_balls - full._n_balls * s**3) <= 1
    # The grasp interface is hardware-matched: grip capsule radius stays 15 mm.
    assert float(task.model.geom("handle_grip").size[0]) == pytest.approx(0.015)
    assert len(task.reset_pose) == task.model.nq
    with pytest.raises(ValueError):
        SpotPitMove(fill_ratio=FILL, pit_scale=0.1)


def test_set_rigid_water_on_model_roundtrip():
    import mujoco

    from sumo.tasks.spot.spot_pit_base import (
        EMPTY_PIT_COM,
        EMPTY_PIT_MASS,
        set_rigid_water_on_model,
    )

    task = SpotPitMove(fill_ratio=FILL, planner_water_model="empty")
    model = task.model
    pit_id = model.body("pit").id
    set_rigid_water_on_model(model, 3.0, np.array([0.02, 0.0, 0.1]))
    assert model.body_mass[pit_id] == pytest.approx(4.0)
    assert model.body_ipos[pit_id][2] < EMPTY_PIT_COM[2]  # water pulls the COM down
    set_rigid_water_on_model(model, 0.0, np.zeros(3))  # empty again
    assert model.body_mass[pit_id] == pytest.approx(EMPTY_PIT_MASS)
    mujoco.mj_forward(model, task.data)  # model stays consistent
