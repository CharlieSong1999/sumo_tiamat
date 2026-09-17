"""Research reconstruction must not change observed state or overlap coarse balls."""

import mujoco
import numpy as np
import pytest
from judo.app.structs import MujocoState

from sumo.tasks.spot import jug_water
from sumo.tasks.spot.spot_jug_manipulation import SpotJugRollArmGentle, SpotJugRollArmGentleConfig
from tools.jug_roll_water_reconstruction import RADII, geometry_metrics, packed_local_positions, reconstruct


@pytest.mark.parametrize("balls", RADII)
def test_geometry_for_all_orientations(balls):
    radius = RADII[balls]
    rng = np.random.default_rng(991)
    quats = list(rng.normal(size=(200, 4)))
    quats += [np.array([np.cos(t / 2), np.sin(t / 2), 0, 0]) for t in np.linspace(0, np.pi, 181)]
    (bottom, wall), (top, _), *_ = jug_water.PROFILE
    for quat in quats:
        pts = packed_local_positions(balls, radius, quat)
        np.testing.assert_array_equal(pts, packed_local_positions(balls, radius, quat))
        assert pts.shape == (balls, 3)
        assert geometry_metrics(pts, radius)["overlapping_pairs"] == 0
        assert np.all(np.linalg.norm(pts[:, :2], axis=1) + radius <= wall + 1e-9)
        assert np.all(pts[:, 2] - radius >= bottom - 1e-9)
        assert np.all(pts[:, 2] + radius <= top + 1e-9)


@pytest.mark.parametrize("balls", RADII)
def test_state_reconstruction_only_changes_hidden_water(balls):
    task = SpotJugRollArmGentle(SpotJugRollArmGentleConfig(water_ball_radius=RADII[balls]))
    task.reset()
    state = MujocoState(
        3.0,
        task.data.qpos.copy(),
        np.arange(task.model.nv) * 0.01,
        task.data.mocap_pos.copy(),
        task.data.mocap_quat.copy(),
        {},
    )
    original_q, original_v = state.qpos.copy(), state.qvel.copy()
    new, metrics = reconstruct(task, state, "packed")
    assert metrics["overlapping_pairs"] == 0
    np.testing.assert_array_equal(state.qpos, original_q)
    np.testing.assert_array_equal(state.qvel, original_v)
    np.testing.assert_array_equal(new.qpos[:33], original_q[:33])
    np.testing.assert_array_equal(new.qvel[:31], original_v[:31])
    assert not new.qvel[31:].any()
    assert task.water_count == balls and task.model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert task.water_mass == pytest.approx(1.9032123084)
    legacy, _ = reconstruct(task, state, "legacy")
    # The deployed synthesize_qpos now places the 3/5/7-ball profiles with the packed
    # rule (adopted 2026-09-17); it must equal reconstruct(mode="packed"), and the
    # legacy row is what reconstruct(mode="legacy") reproduces.
    expected = original_q.copy()
    task.synthesize_qpos(expected)
    np.testing.assert_array_equal(new.qpos, expected)
    if balls == 3:
        np.testing.assert_array_equal(new.qpos, legacy.qpos)


@pytest.mark.parametrize("balls", (5, 7))
def test_legacy_horizontal_overlap_regression(balls):
    pts = jug_water.pooled_local_positions(balls, RADII[balls], [2**-0.5, 2**-0.5, 0.0, 0.0])
    assert geometry_metrics(pts, RADII[balls])["overlapping_pairs"] > 0
    assert (
        geometry_metrics(packed_local_positions(balls, RADII[balls], [2**-0.5, 2**-0.5, 0.0, 0.0]), RADII[balls])[
            "overlapping_pairs"
        ]
        == 0
    )
