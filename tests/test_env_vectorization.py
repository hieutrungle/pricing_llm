"""Comprehensive vectorization and economic edge-case tests for DynamicPricingEnv."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from dynamic_pricing_rl.core.elasticity_math import compute_market_step
from dynamic_pricing_rl.envs.marketplace_env import DynamicPricingEnv


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_env(batch_size: torch.Size) -> DynamicPricingEnv:
    return DynamicPricingEnv(device=_device(), batch_size=batch_size)


def _step_from_state(env: DynamicPricingEnv, observation: torch.Tensor, action: torch.Tensor) -> TensorDict:
    step_input = TensorDict(
        {
            "observation": observation.to(device=env.device, dtype=torch.float32),
            "action": action.to(device=env.device, dtype=torch.float32),
        },
        batch_size=env.batch_size,
        device=env.device,
    )
    return env._step(step_input)


def test_spec_validation() -> None:
    env = _build_env(batch_size=torch.Size([32]))
    rollout_td = env.rollout(max_steps=5)

    obs = rollout_td.get("observation")
    action = rollout_td.get("action")
    reward = rollout_td.get("next").get("reward")

    expected_obs_shape = torch.Size([*env.batch_size, 5, env.observation_spec["observation"].shape[-1]])
    expected_action_shape = torch.Size([*env.batch_size, 5, env.action_spec.shape[-1]])
    expected_reward_shape = torch.Size([*env.batch_size, 5, env.reward_spec.shape[-1]])

    assert obs.shape == expected_obs_shape
    assert action.shape == expected_action_shape
    assert reward.shape == expected_reward_shape

    assert env.observation_spec["observation"].contains(obs[:, 0])
    assert env.action_spec.contains(action[:, 0])
    assert env.reward_spec.contains(reward[:, 0])


def test_massive_batch_vectorization() -> None:
    env = _build_env(batch_size=torch.Size([100000]))

    td = env.reset()
    random_actions = 0.5 + (2.5 * torch.rand((*env.batch_size, 1), device=env.device, dtype=torch.float32))
    action_td = TensorDict({"action": random_actions}, batch_size=env.batch_size, device=env.device)

    step_td = env.step(td.update(action_td))
    reward = step_td.get("next").get("reward")

    assert reward.shape == torch.Size([100000, 1])


def test_edge_case_extreme_surge_pricing() -> None:
    env = _build_env(batch_size=torch.Size([1]))

    observation = torch.tensor([[500.0, 200.0, 20.0, 12.0]], device=env.device, dtype=torch.float32)
    action = torch.tensor([[3.0]], device=env.device, dtype=torch.float32)

    out = _step_from_state(env=env, observation=observation, action=action)

    retained_riders = out.get("observation")[..., 0:1]
    engaged_drivers = out.get("observation")[..., 1:2]
    matches = torch.minimum(retained_riders, engaged_drivers)
    unmatched_riders = torch.clamp(retained_riders - matches, min=0.0)

    # Severe demand drop from surge pricing: retained riders fall below 40% of original demand.
    assert torch.all(retained_riders < (0.4 * observation[..., 0:1]))

    gamma_penalty = 2.0 * unmatched_riders
    assert torch.all(gamma_penalty > 0.0)

    # High multiplier should underperform an unrealistically perfect no-decay/no-penalty upper bound.
    ideal_upper_bound = observation[..., 0:1] * observation[..., 2:3] * 3.0
    reward = out.get("reward")
    assert torch.all(reward < (0.5 * ideal_upper_bound))
    assert torch.all(torch.isfinite(reward))


def test_edge_case_price_floor() -> None:
    env = _build_env(batch_size=torch.Size([1]))

    observation = torch.tensor([[150.0, 100.0, 20.0, 8.0]], device=env.device, dtype=torch.float32)
    low_action = torch.tensor([[0.5]], device=env.device, dtype=torch.float32)
    neutral_action = torch.tensor([[1.0]], device=env.device, dtype=torch.float32)

    floor_out = _step_from_state(env=env, observation=observation, action=low_action)
    neutral_out = _step_from_state(env=env, observation=observation, action=neutral_action)

    engaged_floor = floor_out.get("observation")[..., 1:2]
    engaged_neutral = neutral_out.get("observation")[..., 1:2]

    # Supply-side participation collapses at the floor price.
    assert torch.all(engaged_floor < (0.4 * observation[..., 1:2]))
    assert torch.all(engaged_floor < engaged_neutral)

    floor_reward = floor_out.get("reward")
    neutral_reward = neutral_out.get("reward")

    # Lower completion volume should reduce reward materially at the price floor.
    assert torch.all(floor_reward < neutral_reward)

    retained_floor = floor_out.get("observation")[..., 0:1]
    unmatched_floor = torch.clamp(retained_floor - torch.minimum(retained_floor, engaged_floor), min=0.0)
    assert torch.all(unmatched_floor > 0.0)


def test_edge_case_empty_market() -> None:
    env = _build_env(batch_size=torch.Size([1]))

    observation = torch.tensor([[0.0, 0.0, 20.0, 5.0]], device=env.device, dtype=torch.float32)
    action = torch.tensor([[1.0]], device=env.device, dtype=torch.float32)

    out = _step_from_state(env=env, observation=observation, action=action)
    reward = out.get("reward")

    assert torch.allclose(reward, torch.zeros_like(reward))
    assert torch.all(torch.isfinite(out.get("observation")))
    assert torch.all(torch.isfinite(reward))


def test_action_clamping_defense() -> None:
    env = _build_env(batch_size=torch.Size([1]))

    observation = torch.tensor([[100.0, 80.0, 20.0, 9.0]], device=env.device, dtype=torch.float32)
    extreme_action = torch.tensor([[10.0]], device=env.device, dtype=torch.float32)
    clamped_action = torch.tensor([[3.0]], device=env.device, dtype=torch.float32)

    out = _step_from_state(env=env, observation=observation, action=extreme_action)

    active_riders = observation[..., 0:1]
    active_drivers = observation[..., 1:2]
    base_price = observation[..., 2:3]
    time_of_day = observation[..., 3:4]

    expected_market = compute_market_step(
        active_riders=active_riders,
        active_drivers=active_drivers,
        base_price=base_price,
        price_multiplier=clamped_action,
    )
    expected_observation = torch.cat(
        [
            expected_market["retained_riders"],
            expected_market["engaged_drivers"],
            base_price,
            torch.remainder(time_of_day + 1.0, 24.0),
        ],
        dim=-1,
    )

    assert torch.allclose(out.get("observation"), expected_observation)
    assert torch.allclose(out.get("reward"), expected_market["total_profit"])
