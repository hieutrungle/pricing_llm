"""Phase 2 GRPO fine-tuning script for DynamicPricingEnv.

Hardware assumption:
- Single RTX 3090 (24 GB VRAM).

Default training config (tuned for practical 3090 runs):
- per_device_train_batch_size=2
- gradient_accumulation_steps=2
- max_prompt_length=256
- max_completion_length=64
- num_generations=2

Overrides:
- Use CLI flags, for example:
  python dynamic_pricing_rl/train_grpo.py --max-steps 3 --debug
"""

from __future__ import annotations

import argparse
from datetime import datetime
import inspect
import json
import logging
import random
import re
from ast import literal_eval
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from datasets import Dataset
from tensordict import TensorDict
import unsloth  # Required before importing trl to activate Unsloth patches.
import trl
from unsloth import FastLanguageModel
from transformers import TrainerCallback

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from dynamic_pricing_rl.envs.marketplace_env import DynamicPricingEnv
from dynamic_pricing_rl.core.elasticity_math import get_optimal_multiplier


LOGGER = logging.getLogger(__name__)

GRPOConfig = getattr(trl, "GRPOConfig")
GRPOTrainer = getattr(trl, "GRPOTrainer")

SEED = 3407
DEFAULT_SAVE_PATH = "dynamic_pricing_lora"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Parsing Utilities
# -----------------------------

_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_STATE_PATTERN = re.compile(
    rf"State\s*:\s*\[\s*"
    rf"riders\s*:\s*(?P<riders>{_NUMBER_PATTERN})\s*,\s*"
    rf"drivers\s*:\s*(?P<drivers>{_NUMBER_PATTERN})\s*,\s*"
    rf"base_price\s*:\s*(?P<base_price>{_NUMBER_PATTERN})\s*,\s*"
    rf"time\s*:\s*(?P<time>{_NUMBER_PATTERN})\s*\]",
    flags=re.IGNORECASE,
)
_JSON_BLOCK_PATTERN = re.compile(r"\{[^{}]*\}", flags=re.DOTALL)
_MULTIPLIER_PATTERN = re.compile(
    r"[\"']?(?:multiplier|price_multiplier)[\"']?\s*[:=]\s*"
    r"[\"']?(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)[\"']?",
    flags=re.IGNORECASE,
)


@dataclass
class TrainConfig:
    """Configuration for GRPO training and reward bridging."""

    model_name: str = "unsloth/gemma-4-E4B-it"
    output_dir: str = "grpo_dynamic_pricing_outputs"
    save_path: str = DEFAULT_SAVE_PATH
    logging_dir: str | None = None
    report_to: str = "none"

    num_samples: int = 1000
    num_train_epochs: int = 10
    max_steps: int = 200

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6

    max_seq_length: int = 256
    max_prompt_length: int = 256
    max_completion_length: int = 64
    num_generations: int = 2

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    load_in_4bit: bool = True
    fast_inference: bool = False

    invalid_sample_penalty: float = -5.0
    reward_scale: float = 1.0 / 1000.0
    reward_clamp_min: float | None = None
    reward_clamp_max: float | None = None

    seed: int = SEED
    logging_steps: int = 1
    save_steps: int = 100
    enable_thinking: bool = False
    debug: bool = False

    use_wandb: bool = False
    wandb_project: str = "pricing-llm"
    wandb_entity: str | None = "hieult"
    wandb_run_name: str | None = None
    wandb_mode: str = "offline"
    wandb_tags: str | None = None

    run_post_train_eval: bool = False
    eval_samples: int = 8


def _parse_args() -> TrainConfig:
    """Parse CLI args into a TrainConfig."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train Gemma-4-E4B with GRPO on DynamicPricingEnv.")

    parser.add_argument("--model-name", type=str, default=defaults.model_name)
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--num-train-epochs", type=int, default=defaults.num_train_epochs)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--max-prompt-length", type=int, default=defaults.max_prompt_length)
    parser.add_argument("--max-completion-length", type=int, default=defaults.max_completion_length)
    parser.add_argument("--num-generations", type=int, default=defaults.num_generations)

    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)

    parser.add_argument("--invalid-sample-penalty", type=float, default=defaults.invalid_sample_penalty)
    parser.add_argument("--reward-scale", type=float, default=defaults.reward_scale)
    parser.add_argument("--reward-clamp-min", type=float, default=defaults.reward_clamp_min)
    parser.add_argument("--reward-clamp-max", type=float, default=defaults.reward_clamp_max)

    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--save-path", type=str, default=defaults.save_path)
    parser.add_argument("--report-to", type=str, default=defaults.report_to)
    parser.add_argument("--logging-dir", type=str, default=defaults.logging_dir)
    parser.add_argument("--logging-steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--save-steps", type=int, default=defaults.save_steps)

    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb-entity", type=str, default=defaults.wandb_entity)
    parser.add_argument("--wandb-run-name", type=str, default=defaults.wandb_run_name)
    parser.add_argument("--wandb-mode", type=str, default=defaults.wandb_mode)
    parser.add_argument("--wandb-tags", type=str, default=defaults.wandb_tags)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--run-post-train-eval", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=defaults.eval_samples)

    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        save_path=args.save_path,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        num_samples=args.num_samples,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        invalid_sample_penalty=args.invalid_sample_penalty,
        reward_scale=args.reward_scale,
        reward_clamp_min=args.reward_clamp_min,
        reward_clamp_max=args.reward_clamp_max,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        enable_thinking=args.enable_thinking,
        debug=args.debug,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=args.wandb_tags,
        run_post_train_eval=args.run_post_train_eval,
        eval_samples=args.eval_samples,
    )


def _setup_logging(debug: bool) -> None:
    """Configure process-wide logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy (if available), and Torch for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        LOGGER.debug("NumPy is not installed; skipping NumPy seeding.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_market_prompt(
    tokenizer: Any,
    riders: float,
    drivers: float,
    base_price: float,
    time_of_day: float,
    config: TrainConfig,
) -> str:
    """Create a Gemma chat-template prompt that strongly biases JSON-only output."""
    system_prompt = (
        "You are an AI pricing agent. "
        "Return ONLY a valid JSON object with exactly one key \"multiplier\" and a numeric value. "
        "Do not output markdown or explanations. "
        "When you are finished, you MUST immediately output the word 'STOP'."
    )
    user_prompt = (
        f"State: [riders: {riders:.1f}, drivers: {drivers:.1f}, "
        f"base_price: {base_price:.1f}, time: {time_of_day:.1f}]. "
        "What is the optimal price multiplier?"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            if "enable_thinking" in inspect.signature(tokenizer.apply_chat_template).parameters:
                template_kwargs["enable_thinking"] = config.enable_thinking
            return tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **template_kwargs)

    return (
        f"System: {system_prompt}\n"
        f"User: {user_prompt}"
    )


def generate_market_prompts(tokenizer: Any, config: TrainConfig) -> Dataset:
    """Generate synthetic prompt-only samples for GRPO training."""
    rng = random.Random(config.seed)
    prompts: list[str] = []

    for _ in range(config.num_samples):
        riders = float(rng.randint(50, 500))
        drivers = float(rng.randint(20, 200))
        base_price = float(rng.randint(10, 50))
        time_of_day = float(rng.randint(0, 23))
        prompts.append(_format_market_prompt(tokenizer, riders, drivers, base_price, time_of_day, config))

    return Dataset.from_dict({"prompt": prompts})


def _extract_completion_text(completion_sample: Any) -> str:
    """Extract generated text across TRL completion payload variants."""
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
        # If payload is a list of message dicts, prefer the last non-empty text item.
        if completion_sample and all(isinstance(item, dict) for item in completion_sample):
            for item in reversed(completion_sample):
                text = _extract_completion_text(item)
                if text.strip():
                    return text
        parts = [_extract_completion_text(item) for item in completion_sample]
        return "\n".join(part for part in parts if part)

    return str(completion_sample)


def _parse_multiplier_from_completion(completion_sample: Any, *, debug: bool = False) -> float | None:
    """Parse multiplier from completion text with JSON-first strategy and regex fallback."""
    text = _extract_completion_text(completion_sample)
    if not text:
        return None
    if debug:
        LOGGER.debug("Raw completion text: %s", text[:1000])

    parse_text = text.split("STOP", 1)[0]
    parsed_multiplier: float | None = None

    for match in _JSON_BLOCK_PATTERN.finditer(parse_text):
        json_candidate = match.group(0).strip()
        payload: Any
        try:
            payload = json.loads(json_candidate)
        except json.JSONDecodeError:
            try:
                payload = literal_eval(json_candidate)
            except (ValueError, SyntaxError):
                continue

        if not isinstance(payload, dict):
            continue

        for key in ("multiplier", "price_multiplier"):
            if key in payload:
                try:
                    parsed_multiplier = float(payload[key])
                except (TypeError, ValueError):
                    continue

    if parsed_multiplier is not None:
        return parsed_multiplier

    regex_match: re.Match[str] | None = None
    for match in _MULTIPLIER_PATTERN.finditer(parse_text):
        regex_match = match
    if regex_match is None:
        return None

    try:
        return float(regex_match.group("value"))
    except (TypeError, ValueError):
        return None


def _parse_state_from_prompt(prompt: str) -> tuple[float, float, float, float] | None:
    """Extract riders/drivers/base_price/time from prompt text."""
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


class RewardBridge:
    """Bridge TRL GRPO completions to batched TorchRL rewards."""

    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        # Unsloth's GRPO wrapper expects reward callables to expose __name__.
        self.__name__ = "env_reward_func"
        self.config = config
        self.device = device
        self._env_cache: dict[int, DynamicPricingEnv] = {}
        self._batch_index = 0
        self._total_samples = 0
        self._total_valid = 0
        self._parse_success_history: list[float] = []
        self._reward_mean_history: list[float] = []
        self._reward_std_history: list[float] = []
        self._regret_mean_history: list[float] = []
        self._regret_std_history: list[float] = []
        self._m_error_mean_history: list[float] = []
        self._m_error_std_history: list[float] = []
        self._wandb_run: Any | None = None

    def set_wandb_run(self, wandb_run: Any | None) -> None:
        """Attach a W&B run for live metric logging."""
        self._wandb_run = wandb_run

    def _get_env(self, batch_size: int) -> DynamicPricingEnv:
        """Cache one environment per batch size to avoid repeated TorchRL env construction overhead."""
        if batch_size not in self._env_cache:
            self._env_cache[batch_size] = DynamicPricingEnv(device=self.device, batch_size=torch.Size([batch_size]))
        return self._env_cache[batch_size]

    def __call__(self, prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        self._batch_index += 1
        total_batch = len(prompts)
        rewards = [self.config.invalid_sample_penalty for _ in prompts]
        if total_batch == 0:
            return rewards

        completion_items = list(completions)
        if len(completion_items) != total_batch:
            LOGGER.warning(
                "RewardBridge received mismatched lengths: prompts=%d completions=%d",
                total_batch,
                len(completion_items),
            )
            if len(completion_items) < total_batch:
                completion_items.extend([None] * (total_batch - len(completion_items)))
            else:
                completion_items = completion_items[:total_batch]

        observation = torch.zeros((total_batch, 4), dtype=torch.float32, device=self.device)
        action = torch.ones((total_batch, 1), dtype=torch.float32, device=self.device)
        valid_mask = torch.zeros((total_batch,), dtype=torch.bool, device=self.device)
        first_valid_state: tuple[float, float, float, float] | None = None
        first_valid_multiplier: float | None = None

        for idx, (prompt, completion_sample) in enumerate(zip(prompts, completion_items)):
            state = _parse_state_from_prompt(prompt)
            multiplier = _parse_multiplier_from_completion(completion_sample, debug=self.config.debug)
            if state is None or multiplier is None:
                continue

            if first_valid_state is None:
                first_valid_state = state
                first_valid_multiplier = multiplier

            observation[idx, :] = torch.tensor([state[0], state[1], state[2], state[3]], dtype=torch.float32, device=self.device)
            action[idx, 0] = float(multiplier)
            valid_mask[idx] = True

        if not bool(valid_mask.any().item()):
            return rewards

        if observation.shape != (total_batch, 4) or action.shape != (total_batch, 1):
            LOGGER.warning("Unexpected reward bridge tensor shapes: observation=%s action=%s", observation.shape, action.shape)
            return rewards

        td = TensorDict(
            {
                "observation": observation,
                "action": action,
                "done": torch.zeros((total_batch, 1), dtype=torch.bool, device=self.device),
                "terminated": torch.zeros((total_batch, 1), dtype=torch.bool, device=self.device),
                "truncated": torch.zeros((total_batch, 1), dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([total_batch]),
            device=self.device,
        )

        env = self._get_env(total_batch)
        try:
            with torch.no_grad():
                step_output = env.step(td)
                next_td = step_output.get("next") if hasattr(step_output, "get") and step_output.get("next") is not None else step_output
                reward_tensor = next_td.get("reward") if hasattr(next_td, "get") else next_td["reward"]
        except Exception:
            LOGGER.exception(
                "env.step failed in reward bridge. first_valid_state=%s first_valid_multiplier=%s",
                first_valid_state,
                first_valid_multiplier,
            )
            return rewards

        if reward_tensor is None:
            LOGGER.warning("Reward tensor is missing from environment step output.")
            return rewards

        if reward_tensor.ndim == 2 and reward_tensor.shape[-1] == 1:
            reward_tensor = reward_tensor.squeeze(-1)
        elif reward_tensor.ndim != 1:
            LOGGER.warning("Unexpected reward tensor shape %s; flattening.", reward_tensor.shape)
            reward_tensor = reward_tensor.reshape(-1)

        valid_indices = torch.where(valid_mask)[0].detach().cpu().tolist()
        valid_regrets: list[float] = []
        valid_multiplier_errors: list[float] = []
        for index in valid_indices:
            if index >= reward_tensor.shape[0]:
                continue

            scaled_reward = reward_tensor[index] * self.config.reward_scale
            if self.config.reward_clamp_min is not None or self.config.reward_clamp_max is not None:
                min_value = self.config.reward_clamp_min if self.config.reward_clamp_min is not None else float("-inf")
                max_value = self.config.reward_clamp_max if self.config.reward_clamp_max is not None else float("inf")
                scaled_reward = torch.clamp(scaled_reward, min=min_value, max=max_value)

            rewards[index] = float(scaled_reward.item())

            state_tuple = _parse_state_from_prompt(prompts[index])
            multiplier = _parse_multiplier_from_completion(completion_items[index], debug=self.config.debug)
            if state_tuple is None or multiplier is None:
                continue

            riders = float(state_tuple[0])
            drivers = float(state_tuple[1])
            base_price = float(state_tuple[2])
            optimal_m, optimal_profit = get_optimal_multiplier(riders, drivers, base_price, DEVICE)
            llm_profit = rewards[index] / self.config.reward_scale
            valid_regrets.append(float(optimal_profit - llm_profit))
            valid_multiplier_errors.append(float(abs(optimal_m - multiplier)))

        valid_count = len(valid_indices)
        parse_success_rate = float(valid_count / total_batch) if total_batch > 0 else 0.0
        self._total_samples += total_batch
        self._total_valid += valid_count
        self._parse_success_history.append(parse_success_rate)

        reward_values = torch.tensor(rewards, dtype=torch.float32)
        reward_mean = float(reward_values.mean().item())
        reward_std = float(reward_values.std(unbiased=False).item())
        self._reward_mean_history.append(reward_mean)
        self._reward_std_history.append(reward_std)

        if valid_regrets:
            regret_tensor = torch.tensor(valid_regrets, dtype=torch.float32)
            m_error_tensor = torch.tensor(valid_multiplier_errors, dtype=torch.float32)
            regret_mean = float(regret_tensor.mean().item())
            regret_std = float(regret_tensor.std(unbiased=False).item())
            m_error_mean = float(m_error_tensor.mean().item())
            m_error_std = float(m_error_tensor.std(unbiased=False).item())
        else:
            regret_mean = 0.0
            regret_std = 0.0
            m_error_mean = 0.0
            m_error_std = 0.0

        self._regret_mean_history.append(regret_mean)
        self._regret_std_history.append(regret_std)
        self._m_error_mean_history.append(m_error_mean)
        self._m_error_std_history.append(m_error_std)

        LOGGER.info(
            "reward_bridge step=%d parse_success=%.3f valid=%d/%d reward_mean=%.4f reward_std=%.4f regret_mean=%.4f regret_std=%.4f m_error_mean=%.4f m_error_std=%.4f",
            self._batch_index,
            parse_success_rate,
            valid_count,
            total_batch,
            reward_mean,
            reward_std,
            regret_mean,
            regret_std,
            m_error_mean,
            m_error_std,
        )

        if self._wandb_run is not None:
            self._wandb_run.log(
                {
                    "train/parse_success": parse_success_rate,
                    "train/reward_mean": reward_mean,
                    "train/reward_std": reward_std,
                    "train/regret_mean": regret_mean,
                    "train/regret_std": regret_std,
                    "train/m_error_mean": m_error_mean,
                    "train/m_error_std": m_error_std,
                    "train/valid_samples": valid_count,
                    "train/total_samples": total_batch,
                },
                commit=False, # <--- ADD THIS
            )

        return rewards

    def summary(self) -> dict[str, float]:
        """Return aggregate parse/reward telemetry collected during training."""
        overall_parse_success = float(self._total_valid / self._total_samples) if self._total_samples > 0 else 0.0
        mean_parse_success = (
            float(sum(self._parse_success_history) / len(self._parse_success_history)) if self._parse_success_history else 0.0
        )
        mean_reward = float(sum(self._reward_mean_history) / len(self._reward_mean_history)) if self._reward_mean_history else 0.0
        mean_reward_std = (
            float(sum(self._reward_std_history) / len(self._reward_std_history)) if self._reward_std_history else 0.0
        )
        mean_regret = float(sum(self._regret_mean_history) / len(self._regret_mean_history)) if self._regret_mean_history else 0.0
        mean_regret_std = (
            float(sum(self._regret_std_history) / len(self._regret_std_history)) if self._regret_std_history else 0.0
        )
        mean_m_error = float(sum(self._m_error_mean_history) / len(self._m_error_mean_history)) if self._m_error_mean_history else 0.0
        mean_m_error_std = (
            float(sum(self._m_error_std_history) / len(self._m_error_std_history)) if self._m_error_std_history else 0.0
        )
        return {
            "overall_parse_success": overall_parse_success,
            "mean_parse_success": mean_parse_success,
            "mean_reward": mean_reward,
            "mean_reward_std": mean_reward_std,
            "mean_regret": mean_regret,
            "mean_regret_std": mean_regret_std,
            "mean_m_error": mean_m_error,
            "mean_m_error_std": mean_m_error_std,
            "steps_seen": float(self._batch_index),
        }


class EpochEvalCallback(TrainerCallback):
    """Triggers the custom evaluation loop at the end of each training epoch."""
    def __init__(self, tokenizer: Any, reward_bridge: RewardBridge, config: TrainConfig, wandb_run: Any | None):
        self.tokenizer = tokenizer
        self.reward_bridge = reward_bridge
        self.config = config
        self.wandb_run = wandb_run

    def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
        if model is None:
            return
            
        LOGGER.info("=== Running Evaluation for Epoch %s ===", state.epoch)
        model.eval()
        
        _run_post_training_eval(
            model=model,
            tokenizer=self.tokenizer,
            reward_bridge=self.reward_bridge,
            config=self.config,
            wandb_run=self.wandb_run,
            epoch=state.epoch,         # Pass the epoch
            step=state.global_step     # Pass the global step
        )
        
        model.train()
        

def _build_trainer(model: Any, tokenizer: Any, train_dataset: Dataset, config: Any, reward_func: Any, callbacks: list[TrainerCallback] | None = None) -> Any:
    """Build GRPOTrainer with compatibility fallback for tokenizer arg names."""
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": config,
        "train_dataset": train_dataset,
        "reward_funcs": [reward_func],
    }
    
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks

    trainer_init_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "GRPOTrainer.__init__ does not expose processing_class or tokenizer. "
            "Please update TRL/Unsloth compatibility layer."
        )

    return GRPOTrainer(**trainer_kwargs)


def _build_grpo_config(config: TrainConfig) -> Any:
    """Build GRPOConfig and keep compatibility across TRL versions."""
    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    report_to = "wandb" if config.use_wandb else config.report_to
    grpo_config_kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_train_epochs,
        # Keep mini-batch small for 3090 and accumulate to improve stability.
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        # Prompt length used for truncation/padding during GRPO rollout encoding.
        "max_prompt_length": config.max_prompt_length,
        # Keep completion short; task only needs small JSON + STOP marker.
        "max_completion_length": config.max_completion_length,
        # Number of rollouts sampled per prompt for GRPO group comparison.
        "num_generations": config.num_generations,
        "bf16": bf16_enabled,
        "fp16": not bf16_enabled,
        "report_to": report_to,
    }
    if config.max_steps != -1:
        grpo_config_kwargs["max_steps"] = config.max_steps
    else:
        LOGGER.info("max_steps=-1: training will run for the full dataset using num_train_epochs=%s.", config.num_train_epochs)
    if config.logging_dir:
        grpo_config_kwargs["logging_dir"] = config.logging_dir

    grpo_config_params = inspect.signature(GRPOConfig.__init__).parameters
    filtered_grpo_kwargs = {key: value for key, value in grpo_config_kwargs.items() if key in grpo_config_params}
    dropped_keys = sorted(set(grpo_config_kwargs) - set(filtered_grpo_kwargs))
    if dropped_keys:
        LOGGER.debug("Ignoring unsupported GRPOConfig keys for this version: %s", dropped_keys)

    return GRPOConfig(**filtered_grpo_kwargs)


def _maybe_init_wandb(config: TrainConfig, train_dataset: Dataset) -> Any | None:
    """Initialize W&B if requested and available."""
    if not config.use_wandb:
        return None
    if wandb is None:
        LOGGER.warning("wandb is not installed; continuing without W&B tracking.")
        return None

    tags = [tag.strip() for tag in config.wandb_tags.split(",") if tag.strip()] if config.wandb_tags else None
    effective_run_name = config.wandb_run_name or f"{config.wandb_project}_{datetime.now().strftime('%y%m%d')}"
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=effective_run_name,
        mode=cast(Any, config.wandb_mode),
        tags=tags,
        config=asdict(config),
    )
    if run is not None:
        run.summary["train_dataset_size"] = len(train_dataset)
    return run


def _load_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any]:
    """Load Gemma model/tokenizer and attach LoRA adapters with conservative defaults."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        fast_inference=config.fast_inference,
    )
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules="all-linear",
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )
    return model, tokenizer


def _run_post_training_eval(
    model: Any,
    tokenizer: Any,
    reward_bridge: RewardBridge,
    config: TrainConfig,
    wandb_run: Any | None = None,
    epoch: float | None = None,
    step: int | None = None,
) -> None:
    """Optional quick sanity evaluation on fixed synthetic states."""
    LOGGER.info("Running post-training sanity eval on %d samples.", config.eval_samples)
    
    # Using a fixed seed ensures we evaluate on the EXACT same states every epoch
    rng = random.Random(config.seed + 1)
    prompts: list[str] = []
    for _ in range(config.eval_samples):
        riders = float(rng.randint(50, 500))
        drivers = float(rng.randint(20, 200))
        base_price = float(rng.randint(10, 50))
        time_of_day = float(rng.randint(0, 23))
        prompts.append(_format_market_prompt(tokenizer, riders, drivers, base_price, time_of_day, config))
        
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    with torch.no_grad():
        encoded = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
        ).to(DEVICE)
        generated = model.generate(
            **encoded,
            max_new_tokens=config.max_completion_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    new_tokens = generated[:, prompt_len:]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    tokenizer.padding_side = original_padding_side

    eval_rewards = reward_bridge(prompts, completions)
    
    eval_regrets: list[float] = []
    eval_multiplier_errors: list[float] = []
    valid_parses = 0
    
    # Initialize a WandB Table to visualize outputs
    eval_table = None
    if wandb_run is not None:
        import wandb
        eval_table = wandb.Table(columns=["State", "LLM Multiplier", "Optimal Multiplier", "Regret ($)", "Completion Text"])
    
    for idx, (prompt, completion, reward) in enumerate(zip(prompts, completions, eval_rewards)):
        llm_multiplier = _parse_multiplier_from_completion(completion, debug=config.debug)
        state = _parse_state_from_prompt(prompt)
        
        state_str = "Parse Error"
        opt_m_str = "N/A"
        llm_m_str = "Invalid"
        regret_str = "N/A"
        
        if state is not None:
            riders, drivers, base_price, _ = state
            state_str = f"R:{riders:.0f} D:{drivers:.0f} P:${base_price:.0f}"
            optimal_m, optimal_profit = get_optimal_multiplier(riders, drivers, base_price, DEVICE)
            opt_m_str = f"{optimal_m:.2f}"
            
            if llm_multiplier is not None:
                valid_parses += 1
                llm_profit = reward / config.reward_scale
                regret = optimal_profit - llm_profit
                m_error = abs(optimal_m - llm_multiplier)
                
                eval_regrets.append(float(regret))
                eval_multiplier_errors.append(float(m_error))
                
                llm_m_str = f"{llm_multiplier:.2f}"
                regret_str = f"{regret:.2f}"

        # Add row to WandB table
        if eval_table is not None:
            eval_table.add_data(state_str, llm_m_str, opt_m_str, regret_str, completion[:200].replace('\n', ' '))

    # Log metrics
    if wandb_run is not None:
        parse_success_rate = valid_parses / len(prompts) if prompts else 0.0
        log_dict = {
            "eval/parse_success": parse_success_rate,
            "eval/predictions": eval_table
        }
        
        if eval_regrets:
            regret_tensor = torch.tensor(eval_regrets, dtype=torch.float32)
            m_error_tensor = torch.tensor(eval_multiplier_errors, dtype=torch.float32)
            log_dict.update({
                "eval/regret_mean": float(regret_tensor.mean().item()),
                "eval/regret_std": float(regret_tensor.std(unbiased=False).item()),
                "eval/m_error_mean": float(m_error_tensor.mean().item()),
                "eval/m_error_std": float(m_error_tensor.std(unbiased=False).item()),
            })
            
        if epoch is not None:
            log_dict["epoch"] = epoch
            
        # Use the actual global training step, not the eval bridge's step
        wandb_run.log(log_dict, step=step)


def main() -> None:
    """Entry point for GRPO training."""
    config = _parse_args()
    _setup_logging(config.debug)
    _set_seed(config.seed)

    LOGGER.info("Starting GRPO run on device=%s", DEVICE)
    LOGGER.info("Training config: %s", json.dumps(asdict(config), sort_keys=True, indent=2))

    model, tokenizer = _load_model_and_tokenizer(config)
    train_dataset = generate_market_prompts(tokenizer, config)
    wandb_run = _maybe_init_wandb(config, train_dataset)
    if config.debug:
        LOGGER.debug("Sample prompt: %s", train_dataset[0]["prompt"][:600])

    reward_bridge = RewardBridge(config=config, device=DEVICE)
    reward_bridge.set_wandb_run(wandb_run)
    
    # 1. Create a separate bridge just for evaluation so it doesn't pollute training metrics
    eval_reward_bridge = RewardBridge(config=config, device=DEVICE)
    
    # 2. Instantiate the callback
    eval_callback = EpochEvalCallback(
        tokenizer=tokenizer,
        reward_bridge=eval_reward_bridge, 
        config=config,
        wandb_run=wandb_run
    )

    grpo_config = _build_grpo_config(config)
    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=grpo_config,
        reward_func=reward_bridge,
        callbacks=[eval_callback]  # 3. Pass the callback to the trainer
    )
    
    LOGGER.info("Starting training...")
    trainer.train()
    LOGGER.info("Training complete. Saving artifacts to %s", config.save_path)

    bridge_summary = reward_bridge.summary()
    LOGGER.info("Reward bridge summary: %s", json.dumps(bridge_summary, sort_keys=True))
    if wandb_run is not None:
        wandb_run.summary.update(bridge_summary)
        wandb_run.summary["save_path"] = config.save_path

    model.save_pretrained(config.save_path)
    tokenizer.save_pretrained(config.save_path)

    if config.run_post_train_eval:
        _run_post_training_eval(model, tokenizer, reward_bridge, config, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
