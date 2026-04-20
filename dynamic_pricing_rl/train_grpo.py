"""Phase 2 GRPO fine-tuning script for DynamicPricingEnv."""

from __future__ import annotations

import json
import random
import re
from typing import Any

import torch
from datasets import Dataset
from tensordict import TensorDict
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from dynamic_pricing_rl.envs.marketplace_env import DynamicPricingEnv


MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
INVALID_SAMPLE_PENALTY = -500.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_STATE_PATTERN = re.compile(
    r"State:\s*\[riders:\s*(?P<riders>-?\d+(?:\.\d+)?),\s*"
    r"drivers:\s*(?P<drivers>-?\d+(?:\.\d+)?),\s*"
    r"base_price:\s*(?P<base_price>-?\d+(?:\.\d+)?),\s*"
    r"time:\s*(?P<time>-?\d+(?:\.\d+)?)\]"
)
_JSON_BLOCK_PATTERN = re.compile(r"\{.*?\}", flags=re.DOTALL)
_ENV_CACHE: dict[int, DynamicPricingEnv] = {}


def generate_market_prompts(num_samples: int = 1000) -> Dataset:
    """Generate synthetic market states as prompt-only samples for GRPO."""
    rng = random.Random(3407)
    prompts: list[str] = []

    for _ in range(num_samples):
        riders = float(rng.randint(50, 500))
        drivers = float(rng.randint(20, 200))
        base_price = float(rng.randint(10, 50))
        time_of_day = float(rng.randint(0, 23))

        prompt = (
            "System: You are an AI pricing agent. Output ONLY valid JSON with a single key 'multiplier'.\n"
            f"User: State: [riders: {riders:.1f}, drivers: {drivers:.1f}, "
            f"base_price: {base_price:.1f}, time: {time_of_day:.1f}]. What is the optimal price multiplier?"
        )
        prompts.append(prompt)

    return Dataset.from_dict({"prompt": prompts})


def _get_env(batch_size: int) -> DynamicPricingEnv:
    """Reuse one environment per batch size for reward bridge throughput."""
    if batch_size not in _ENV_CACHE:
        _ENV_CACHE[batch_size] = DynamicPricingEnv(device=DEVICE, batch_size=torch.Size([batch_size]))
    return _ENV_CACHE[batch_size]


def _extract_completion_text(completion_sample: Any) -> str:
    """Extract generated text across several GRPO completion payload formats."""
    if completion_sample is None:
        return ""

    if isinstance(completion_sample, str):
        return completion_sample

    if isinstance(completion_sample, dict):
        for key in ("content", "text", "generated_text", "completion", "response", "output_text"):
            if key in completion_sample:
                return _extract_completion_text(completion_sample[key])
        return ""

    if isinstance(completion_sample, (list, tuple)):
        parts = [_extract_completion_text(item) for item in completion_sample]
        return "\n".join(part for part in parts if part)

    return str(completion_sample)


def _parse_multiplier_from_completion(completion_sample: Any) -> float | None:
    """
    Parse multiplier from model completion.

    Returns None when JSON is malformed or key/value is missing.
    """
    text = _extract_completion_text(completion_sample)
    if not text:
        return None

    for match in _JSON_BLOCK_PATTERN.finditer(text):
        json_candidate = match.group(0).strip()
        try:
            payload = json.loads(json_candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict) or "multiplier" not in payload:
            continue

        try:
            return float(payload["multiplier"])
        except (TypeError, ValueError):
            return None

    return None


def _parse_state_from_prompt(prompt: str) -> tuple[float, float, float, float] | None:
    """Extract riders/drivers/base_price/time from prompt string."""
    match = _STATE_PATTERN.search(prompt)
    if match is None:
        return None

    try:
        riders = float(match.group("riders"))
        drivers = float(match.group("drivers"))
        base_price = float(match.group("base_price"))
        time_of_day = float(match.group("time"))
    except ValueError:
        return None

    return riders, drivers, base_price, time_of_day


def env_reward_func(prompts: list[str], completions: list[list[dict]]) -> list[float]:
    """Bridge TRL GRPO completions to batched TorchRL rewards."""
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same batch length.")

    total_batch = len(prompts)
    rewards = [INVALID_SAMPLE_PENALTY for _ in prompts]
    if total_batch == 0:
        return rewards

    observation = torch.zeros((total_batch, 4), dtype=torch.float32, device=DEVICE)
    action = torch.ones((total_batch, 1), dtype=torch.float32, device=DEVICE)
    valid_mask = torch.zeros((total_batch,), dtype=torch.bool, device=DEVICE)

    for idx, (prompt, completion_sample) in enumerate(zip(prompts, completions)):
        state = _parse_state_from_prompt(prompt)
        multiplier = _parse_multiplier_from_completion(completion_sample)
        if state is None or multiplier is None:
            continue

        observation[idx, :] = torch.tensor([state[0], state[1], state[2], state[3]], device=DEVICE, dtype=torch.float32)
        action[idx, 0] = float(multiplier)
        valid_mask[idx] = True

    done = torch.zeros((total_batch, 1), dtype=torch.bool, device=DEVICE)
    terminated = torch.zeros((total_batch, 1), dtype=torch.bool, device=DEVICE)
    truncated = torch.zeros((total_batch, 1), dtype=torch.bool, device=DEVICE)

    td = TensorDict(
        {
            "observation": observation,
            "action": action,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        },
        batch_size=torch.Size([total_batch]),
        device=DEVICE,
    )

    env = _get_env(total_batch)
    try:
        with torch.no_grad():
            next_td = env.step(td).get("next")
            batch_rewards = next_td.get("reward").squeeze(-1)
    except Exception:
        return rewards

    valid_indices = torch.where(valid_mask)[0].detach().cpu().tolist()
    for index in valid_indices:
        rewards[index] = float(batch_rewards[index].item())

    return rewards


def _build_trainer(model: Any, tokenizer: Any, train_dataset: Dataset, config: GRPOConfig) -> GRPOTrainer:
    """Build GRPOTrainer with compatibility fallback for tokenizer arg name."""
    base_kwargs = {
        "model": model,
        "args": config,
        "train_dataset": train_dataset,
        "reward_funcs": [env_reward_func],
    }

    try:
        return GRPOTrainer(processing_class=tokenizer, **base_kwargs)
    except TypeError:
        return GRPOTrainer(tokenizer=tokenizer, **base_kwargs)


def main() -> None:
    train_dataset = generate_market_prompts(num_samples=1000)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=256,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=256,
    )

    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    grpo_config = GRPOConfig(
        output_dir="grpo_dynamic_pricing_outputs",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=1,
        save_steps=100,
        max_prompt_length=128,
        max_completion_length=32,
        num_generations=4,
        bf16=bf16_enabled,
        fp16=not bf16_enabled,
        report_to="none",
    )

    trainer = _build_trainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, config=grpo_config)
    trainer.train()

    model.save_pretrained("dynamic_pricing_lora")
    tokenizer.save_pretrained("dynamic_pricing_lora")


if __name__ == "__main__":
    main()
