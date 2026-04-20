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

def get_optimal_multiplier(
    riders: float,
    drivers: float,
    base_price: float,
    device: torch.device,
    min_m: float = 0.5,
    max_m: float = 3.0,
    steps: int = 2500
) -> tuple[float, float]:
    """
    Brute-force the exact optimal price multiplier using PyTorch tensor broadcasting.
    Returns: (optimal_multiplier, maximum_possible_profit)
    """
    # 1. Create a dense 1D grid of multipliers to test (Shape: [steps, 1])
    m_grid = torch.linspace(min_m, max_m, steps, device=device).unsqueeze(-1)
    
    # 2. Expand the market state to match the grid size
    active_riders = torch.full((steps, 1), riders, device=device)
    active_drivers = torch.full((steps, 1), drivers, device=device)
    base_prices = torch.full((steps, 1), base_price, device=device)
    
    # 3. Calculate profit for all 2,500 possible multipliers simultaneously
    with torch.no_grad():
        market_out = compute_market_step(
            active_riders=active_riders,
            active_drivers=active_drivers,
            base_price=base_prices,
            price_multiplier=m_grid,
        )
        profits = market_out["total_profit"].squeeze(-1)
        
    # 4. Find the multiplier that yielded the absolute highest profit
    best_idx = torch.argmax(profits)
    
    return float(m_grid[best_idx].item()), float(profits[best_idx].item())