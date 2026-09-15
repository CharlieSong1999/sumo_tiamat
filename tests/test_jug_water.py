import mujoco
import numpy as np
import pytest

from sumo.tasks.spot import jug_water
from sumo.tasks.spot.spot_jug_manipulation import (
    SpotJugLayDown,
    SpotJugMove,
    SpotJugRoll,
    SpotJugUpright,
    disable_velocity_rewards,
)


@pytest.fixture(params=[SpotJugUpright, SpotJugLayDown, SpotJugRoll, SpotJugMove], scope="module")
def wet_task(request):
    return request.param(request.param.config_t(water_fill_ratio=0.1))


def test_water_mass_reset_and_state_layout(wet_task):
    task = wet_task
    task.reset()
    assert task.water_count == 19
    assert task.water_mass == pytest.approx(1.9032123084)
    assert task.model.body("jug").mass[0] == pytest.approx(task.jug_mass)  # water is in the balls, not double-counted
    assert task.model.nq == 33 + 7 * task.water_count
    assert task.model.nv == 31 + 6 * task.water_count
    assert task.reset_pose.shape == (task.model.nq,)
    assert task.nu == (11 if task.mode == "upright" else 3)
    assert task.metrics(task.data)["water_retained_fraction"] == 1.0
    assert not task.data.qvel.any()
    poses = task.data.qpos.copy()
    task.reset()
    np.testing.assert_array_equal(task.data.qpos, poses)
    masses = [task.model.body(f"jug_water_{i}").mass[0] for i in range(task.water_count)]
    assert sum(masses) == pytest.approx(task.water_mass)


def test_water_collision_masks(wet_task):
    model = wet_task.model
    water = model.geom("jug_water_0_geom").id

    def collides(other):
        i = model.geom(other).id
        return bool(
            (model.geom_contype[water] & model.geom_conaffinity[i])
            | (model.geom_contype[i] & model.geom_conaffinity[water])
        )

    assert not collides("jug_collision")
    assert collides("jug_inner_floor")
    assert collides("jug_inner_1_0")
    assert collides("jug_inner_cap")
    assert collides("jug_water_1_geom")
    assert collides("ground")
    assert model.geom_condim[water] == 1


def test_water_reward_shape_and_velocity_ablation(wet_task):
    task = wet_task
    task.reset()
    disable_velocity_rewards(task.config)
    states = np.tile(np.r_[task.data.qpos, task.data.qvel], (3, 4, 1))
    sensors = np.tile(task.data.sensordata, (3, 4, 1))
    controls = np.zeros((3, 4, task.nu))
    reward = task.reward(states, sensors, controls)
    assert reward.shape == (3,)
    assert np.isfinite(reward).all()
    # No velocity of the jug, robot, or water directly enters the ablated reward.
    states[..., task.model.nq :] = np.random.default_rng(1).normal(size=(3, 4, task.model.nv))
    np.testing.assert_array_equal(task.reward(states, sensors, controls), reward)


def test_water_stays_inside_fixed_container(wet_task):
    task = wet_task
    task.reset()
    data = mujoco.MjData(task.model)
    data.qpos[:] = task.data.qpos
    fixed = task.data.qpos.copy()
    q = task.model.joint("jug_water_0_joint").qposadr[0]
    v = task.model.joint("jug_water_0_joint").dofadr[0]
    for _ in range(200):
        mujoco.mj_step(task.model, data)
        # Diagnostic only: hold the container while freely integrating particles.
        data.qpos[:q] = fixed[:q]
        data.qvel[:v] = 0
        mujoco.mj_forward(task.model, data)
        local = jug_water.ball_local_positions(task.model, data, task.water_count)
        assert jug_water.contained_mask(local).all()
    assert not data.warning.number.any()


@pytest.mark.parametrize("fill,radius", [(-0.1, 0.025), (0.5, 0.025), (0.1, 0), (0.1, 0.1)])
def test_invalid_water_settings(fill, radius):
    with pytest.raises(ValueError):
        jug_water.water_parameters(fill, radius)
