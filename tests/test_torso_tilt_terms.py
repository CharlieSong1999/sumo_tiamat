"""Torso roll/pitch reward knobs of the coarse arm-roll profile (2026-09-18): off by default, penalise tilt when on."""

import numpy as np

from sumo.tasks.spot.spot_jug_manipulation import (
    SpotJugRollArmGentleCoarse,
    SpotJugRollArmGentleCoarseConfig,
    torso_roll_pitch,
)


def _quat(roll=0.0, pitch=0.0):
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])  # w x y z, yaw 0


def test_roll_pitch_helper_round_trips():
    for r, p in ((0.0, 0.0), (0.2, 0.0), (0.0, -0.3), (0.15, 0.1)):
        roll, pitch = torso_roll_pitch(_quat(r, p)[None, None])
        np.testing.assert_allclose([roll.item(), pitch.item()], [r, p], atol=1e-9)


def _state(task, roll=0.0, pitch=0.0):
    q = np.array(task.reset_pose, dtype=float)
    b = task.body_pose_start
    q[b + 3 : b + 7] = _quat(roll, pitch)
    return np.concatenate([q, np.zeros(task.model.nv)])[None, None, :]


def test_off_by_default_and_absent_from_terms():
    task = SpotJugRollArmGentleCoarse()
    task.reset()
    assert task.config.w_torso_roll == 0.0 and task.config.w_torso_pitch == 0.0
    s = _state(task, roll=0.3)
    terms = task.reward_terms(s, np.zeros((1, 1, task.model.nsensordata)), np.zeros((1, 1, task.nu)))
    assert "torso_roll" not in terms and "torso_pitch" not in terms


def test_tilt_costs_when_enabled():
    task = SpotJugRollArmGentleCoarse(SpotJugRollArmGentleCoarseConfig(w_torso_roll=200.0, w_torso_pitch=100.0))
    task.reset()
    sens, u = np.zeros((1, 1, task.model.nsensordata)), np.zeros((1, 1, task.nu))
    level = task.reward_terms(_state(task), sens, u)
    rolled = task.reward_terms(_state(task, roll=0.2), sens, u)
    pitched = task.reward_terms(_state(task, pitch=0.2), sens, u)
    np.testing.assert_allclose(level["torso_roll"], 0.0, atol=1e-12)
    np.testing.assert_allclose(rolled["torso_roll"], -200.0 * 0.2**2, rtol=1e-9)
    np.testing.assert_allclose(pitched["torso_pitch"], -100.0 * 0.2**2, rtol=1e-9)
    assert task.reward(_state(task, roll=0.2), sens, u)[0] < task.reward(_state(task), sens, u)[0]
