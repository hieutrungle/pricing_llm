"""Pure PyTorch tensor math for marketplace elasticity and reward shaping."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

ALPHA = 0.5
BETA = 0.8
GAMMA = 2.0
DELTA = 5.0


def demand_decay(active_riders: Tensor, price_multiplier: Tensor, alpha: float = ALPHA) -> Tensor:
    """Compute retained riders under positive price shocks."""
    positive_markup = torch.clamp(price_multiplier - 1.0, min=0.0)
    retained_riders = active_riders * torch.exp(-alpha * positive_markup)
    return retained_riders


def supply_growth(active_drivers: Tensor, price_multiplier: Tensor, beta: float = BETA) -> Tensor:
    """Compute engaged drivers as incentives increase."""
    engaged_drivers = active_drivers * (1.0 - torch.exp(-beta * price_multiplier))
    return engaged_drivers


def compute_market_step(
    active_riders: Tensor,
    active_drivers: Tensor,
    base_price: Tensor,
    price_multiplier: Tensor,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
) -> Dict[str, Tensor]:
    """
    Compute batched market dynamics and reward strictly in PyTorch tensors.

    Inputs are expected to be broadcast-compatible tensors with trailing dim 1.
    """
    retained_riders = demand_decay(active_riders=active_riders, price_multiplier=price_multiplier, alpha=alpha)
    engaged_drivers = supply_growth(active_drivers=active_drivers, price_multiplier=price_multiplier, beta=beta)

    matches = torch.minimum(retained_riders, engaged_drivers)
    unmatched_riders = torch.clamp(retained_riders - matches, min=0.0)
    idle_drivers = torch.clamp(engaged_drivers - matches, min=0.0)

    total_profit = (matches * base_price * price_multiplier) - (gamma * unmatched_riders) - (delta * idle_drivers)

    return {
        "retained_riders": retained_riders,
        "engaged_drivers": engaged_drivers,
        "matches": matches,
        "unmatched_riders": unmatched_riders,
        "idle_drivers": idle_drivers,
        "total_profit": total_profit,
    }
