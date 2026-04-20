"""TorchRL EnvBase implementation for GPU-native dynamic pricing."""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from dynamic_pricing_rl.core.elasticity_math import compute_market_step


class DynamicPricingEnv(EnvBase):
    """Dynamic marketplace simulator with fully tensorized step logic."""

    def __init__(
        self,
        device: Optional[torch.device | str] = None,
        batch_size: Optional[torch.Size] = None,
    ) -> None:
        resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_batch_size = batch_size if batch_size is not None else torch.Size([])
        super().__init__(device=resolved_device, batch_size=resolved_batch_size)

        self.rng = torch.Generator(device=self.device)
        self._set_seed(0)
        self._make_spec()

    def _make_spec(self) -> None:
        obs_shape = torch.Size([*self.batch_size, 4])
        action_shape = torch.Size([*self.batch_size, 1])
        reward_shape = torch.Size([*self.batch_size, 1])
        done_shape = torch.Size([*self.batch_size, 1])

        obs_low = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).expand(obs_shape)
        obs_high = torch.tensor([500.0, 200.0, 50.0, 23.0], device=self.device, dtype=torch.float32).expand(obs_shape)

        action_low = torch.full(action_shape, 0.5, device=self.device, dtype=torch.float32)
        action_high = torch.full(action_shape, 3.0, device=self.device, dtype=torch.float32)

        done_low = torch.zeros(done_shape, device=self.device, dtype=torch.bool)
        done_high = torch.ones(done_shape, device=self.device, dtype=torch.bool)

        self.observation_spec = Composite(
            observation=Bounded(
                low=obs_low,
                high=obs_high,
                shape=obs_shape,
                dtype=torch.float32,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )

        self.action_spec = Bounded(
            low=action_low,
            high=action_high,
            shape=action_shape,
            dtype=torch.float32,
            device=self.device,
        )

        self.reward_spec = Unbounded(
            shape=reward_shape,
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = Composite(
            done=Bounded(
                low=done_low,
                high=done_high,
                shape=done_shape,
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Bounded(
                low=done_low,
                high=done_high,
                shape=done_shape,
                dtype=torch.bool,
                device=self.device,
            ),
            truncated=Bounded(
                low=done_low,
                high=done_high,
                shape=done_shape,
                dtype=torch.bool,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )

        self.state_spec = self.observation_spec.clone()

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDict:
        del kwargs
        td_batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        batch_size = self.batch_size if td_batch_size == torch.Size([]) else td_batch_size

        sample_shape = (*batch_size, 1)

        active_riders = 50.0 + (450.0 * torch.rand(sample_shape, device=self.device, generator=self.rng))
        active_drivers = 20.0 + (180.0 * torch.rand(sample_shape, device=self.device, generator=self.rng))
        base_price = 5.0 + (45.0 * torch.rand(sample_shape, device=self.device, generator=self.rng))
        time_of_day_index = torch.randint(
            low=0,
            high=24,
            size=sample_shape,
            generator=self.rng,
            device=self.device,
        ).to(torch.float32)

        observation = torch.cat([active_riders, active_drivers, base_price, time_of_day_index], dim=-1)

        done = torch.zeros(sample_shape, device=self.device, dtype=torch.bool)
        terminated = torch.zeros(sample_shape, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(sample_shape, device=self.device, dtype=torch.bool)

        return TensorDict(
            {
                "observation": observation,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        observation = tensordict.get("observation")
        action = tensordict.get("action")

        if observation is None:
            raise KeyError("Expected key 'observation' in input TensorDict.")
        if action is None:
            raise KeyError("Expected key 'action' in input TensorDict.")

        obs = observation.to(device=self.device, dtype=torch.float32)
        price_multiplier = torch.clamp(action.to(device=self.device, dtype=torch.float32), min=0.5, max=3.0)

        active_riders = obs[..., 0:1]
        active_drivers = obs[..., 1:2]
        base_price = obs[..., 2:3]
        time_of_day_index = obs[..., 3:4]

        market = compute_market_step(
            active_riders=active_riders,
            active_drivers=active_drivers,
            base_price=base_price,
            price_multiplier=price_multiplier,
        )

        next_time = torch.remainder(time_of_day_index + 1.0, 24.0)
        next_observation = torch.cat(
            [
                market["retained_riders"],
                market["engaged_drivers"],
                base_price,
                next_time,
            ],
            dim=-1,
        )

        done_shape = (*obs.shape[:-1], 1)
        done = torch.zeros(done_shape, device=self.device, dtype=torch.bool)
        terminated = torch.zeros(done_shape, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(done_shape, device=self.device, dtype=torch.bool)

        return TensorDict(
            {
                "observation": next_observation,
                "reward": market["total_profit"],
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=obs.shape[:-1],
            device=self.device,
        )

    def _set_seed(self, seed: Optional[int]) -> int:
        safe_seed = 0 if seed is None else int(seed)

        if not hasattr(self, "rng"):
            self.rng = torch.Generator(device=self.device)

        self.rng.manual_seed(safe_seed)
        torch.manual_seed(safe_seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(safe_seed)

        return safe_seed
