"""Regression checks for the offline jug experiment runner."""

import json
import sys

import numpy as np
import pytest

from tools import jug_demo


@pytest.mark.parametrize(
    "arguments",
    [
        ["--iterations", "0"],
        ["--rollouts", "1", "--elites", "2"],
        ["--elites", "0"],
        ["--control-freq", "0"],
        ["--control-freq", "nan"],
        ["--control-freq", "51"],
    ],
)
def test_invalid_cem_budget_is_rejected_before_simulation(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["jug_demo", "spot_jug_upright", *arguments])
    with pytest.raises(SystemExit) as error:
        jug_demo.main()
    assert error.value.code == 2


def test_short_run_records_budget_and_serializable_timings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jug_demo",
            "spot_jug_upright",
            "--seconds",
            "0.04",
            "--rollouts",
            "2",
            "--iterations",
            "1",
            "--elites",
            "1",
            "--jug-contact-dim",
            "3",
            "--out",
            str(tmp_path),
        ],
    )
    jug_demo.main()
    record = json.loads((tmp_path / "spot_jug_upright_seed0.json").read_text())
    assert record["planning"]["candidates_per_update"] == 2
    assert record["planning"]["physics_steps_per_candidate"] == 200
    assert record["planning"]["updates"] == 2
    assert record["planning"]["mean_ms"] > 0
    assert record["jug_contact_dim"] == 3
    with np.load(tmp_path / "spot_jug_upright_seed0.npz") as archive:
        assert len(archive["planning_times"]) == 2
        assert len(archive["iteration_times"]) == 2


def test_planning_clock_preserves_50hz_plant_and_targets_20hz():
    stamps = [step * 0.02 for step in range(51) if jug_demo.planning_due(step, 0.02, 20)]
    np.testing.assert_allclose(stamps[:5], [0, 0.06, 0.10, 0.16, 0.20])
    assert len(stamps) == 21
    np.testing.assert_allclose(np.diff(stamps)[::2], 0.06)
    np.testing.assert_allclose(np.diff(stamps)[1::2], 0.04)
    assert [step for step in range(51) if jug_demo.planning_due(step, 0.02, 25)] == list(range(0, 51, 2))


def test_paper_schedule_uses_variance_not_standard_deviation(monkeypatch):
    config = jug_demo.CrossEntropyMethodConfig(num_rollouts=4, num_nodes=4, num_elites=3)
    optimizer = jug_demo.PaperVarianceCEM(config, 2)
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape))
    nominal = np.zeros((4, 2))
    candidates = optimizer.sample_control_knots(nominal)
    assert candidates.shape == (4, 4, 2)
    np.testing.assert_array_equal(candidates[0], nominal)
    np.testing.assert_allclose(candidates[1, :, 0] ** 2, np.linspace(0.02, 0.6, 4))
    optimizer.sigma[:] = 100  # Fixed schedule, not the stock adaptive-sigma ramp.
    np.testing.assert_array_equal(optimizer.sample_control_knots(nominal), candidates)


def test_paper_short_run_records_horizon_noise_and_actual_clock(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jug_demo",
            "spot_jug_upright",
            "--seconds",
            "0.12",
            "--rollouts",
            "2",
            "--iterations",
            "1",
            "--elites",
            "1",
            "--horizon",
            "1.5",
            "--control-freq",
            "20",
            "--noise-profile",
            "paper-variance",
            "--out",
            str(tmp_path),
        ],
    )
    jug_demo.main()
    record = json.loads((tmp_path / "spot_jug_upright_seed0.json").read_text())
    assert record["planning"]["physics_steps_per_candidate"] == 150
    assert record["planning"]["updates"] == 3
    assert record["planning"]["actual_intervals_ms"] == [40, 60]
    assert record["planning"]["target_period_ms"] == 50
    assert record["noise"]["profile"] == "paper-variance"
    with np.load(tmp_path / "spot_jug_upright_seed0.npz") as archive:
        np.testing.assert_allclose(archive["planning_sim_times"], [0, 0.06, 0.10])
