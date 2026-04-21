"""Merge a LoRA adapter fine-tuned with Unsloth/GRPO into the base Gemma-4 model.

This script loads the adapter on CPU, folds the LoRA weights into the base
model, and saves a standard Hugging Face checkpoint suitable for deployment
with Transformers or vLLM.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from unsloth import FastLanguageModel
except ImportError:  # pragma: no cover - optional runtime dependency
    FastLanguageModel = None


LOGGER = logging.getLogger(__name__)

DEFAULT_LORA_DIR = Path("/home/hieule/research/pricing_llm/dynamic_pricing_lora")
DEFAULT_BASE_MODEL = "unsloth/gemma-4-E4B-it"
DEFAULT_OUTPUT_DIR = Path("/home/hieule/research/pricing_llm/gemma4-e4b-pricing-merged")
DEFAULT_MAX_SEQ_LENGTH = 256

REQUIRED_OUTPUT_FILES = ("config.json", "tokenizer_config.json")
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "spiece.model")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a Gemma-4 LoRA adapter into the base model.")
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR, help="Path to the trained LoRA adapter.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base model ID or local path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the merged model.")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="Weight dtype to use while loading and saving.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Sequence length used when loading with Unsloth.",
    )
    parser.add_argument(
        "--force-peft",
        action="store_true",
        help="Skip Unsloth loading and force plain PEFT merge.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading the tokenizer or model.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def _load_tokenizer(lora_dir: Path, base_model: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    source = lora_dir if (lora_dir / "tokenizer_config.json").exists() else base_model
    LOGGER.info("Loading tokenizer from %s", source)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_unsloth_model_and_tokenizer(
    lora_dir: Path,
    max_seq_length: int,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[Any, PreTrainedTokenizerBase]:
    if FastLanguageModel is None:
        raise RuntimeError("unsloth is not installed; cannot use Unsloth merge path.")

    kwargs: dict[str, Any] = {
        "model_name": str(lora_dir),
        "max_seq_length": max_seq_length,
        "dtype": torch_dtype,
        "load_in_4bit": False,
        "fast_inference": False,
    }
    signature = inspect.signature(FastLanguageModel.from_pretrained)
    if "trust_remote_code" in signature.parameters:
        kwargs["trust_remote_code"] = trust_remote_code
    if "device_map" in signature.parameters:
        kwargs["device_map"] = {"": "cpu"}

    LOGGER.info("Loading LoRA model via Unsloth from %s", lora_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    return model, tokenizer


def _save_merged_with_unsloth(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    torch_dtype: torch.dtype,
) -> None:
    if hasattr(model, "save_pretrained_merged"):
        LOGGER.info("Saving merged model with Unsloth save_pretrained_merged")
        try:
            model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
            return
        except TypeError:
            model.save_pretrained_merged(str(output_dir), tokenizer)
            return

    if not hasattr(model, "merge_and_unload"):
        raise TypeError("Loaded Unsloth model does not support merge_and_unload().")

    LOGGER.info("Merging LoRA adapter into base model weights")
    merged_model = model.merge_and_unload()
    merged_model = merged_model.to(dtype=torch_dtype)
    merged_model.config.torch_dtype = str(torch_dtype).replace("torch.", "")
    LOGGER.info("Saving merged model to %s", output_dir)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def _merge_with_plain_peft(
    lora_dir: Path,
    base_model: str,
    output_dir: Path,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> None:
    LOGGER.info("Loading PEFT model from %s on CPU", lora_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(lora_dir),
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if not hasattr(model, "merge_and_unload"):
        raise TypeError("Loaded PEFT model does not support merge_and_unload().")

    LOGGER.info("Merging LoRA adapter into base model weights")
    merged_model = model.merge_and_unload()
    merged_model = merged_model.to(dtype=torch_dtype)
    merged_model.config.torch_dtype = str(torch_dtype).replace("torch.", "")
    tokenizer = _load_tokenizer(lora_dir, base_model, trust_remote_code)

    LOGGER.info("Saving merged model to %s", output_dir)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def _validate_saved_artifacts(output_dir: Path) -> None:
    missing: list[str] = []

    for required_file in REQUIRED_OUTPUT_FILES:
        if not (output_dir / required_file).exists():
            missing.append(required_file)

    if not any((output_dir / filename).exists() for filename in MODEL_WEIGHT_FILES):
        missing.append("model weights (.safetensors or .bin)")

    if not any((output_dir / filename).exists() for filename in TOKENIZER_FILES):
        missing.append("tokenizer model file")

    if missing:
        raise RuntimeError(f"Merged export is incomplete. Missing artifacts: {', '.join(missing)}")


def merge_lora_adapter(
    lora_dir: Path,
    base_model: str,
    output_dir: Path,
    max_seq_length: int,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    force_peft: bool,
) -> None:
    """Load the adapter, merge it into the base weights, and save a standard checkpoint."""
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory does not exist: {lora_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if force_peft:
        LOGGER.info("--force-peft enabled. Skipping Unsloth merge path.")
        _merge_with_plain_peft(
            lora_dir=lora_dir,
            base_model=base_model,
            output_dir=output_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    else:
        try:
            model, tokenizer = _load_unsloth_model_and_tokenizer(
                lora_dir=lora_dir,
                max_seq_length=max_seq_length,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            _save_merged_with_unsloth(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                torch_dtype=torch_dtype,
            )
        except Exception as error:
            LOGGER.warning("Unsloth merge path failed (%s). Falling back to plain PEFT merge.", error)
            _merge_with_plain_peft(
                lora_dir=lora_dir,
                base_model=base_model,
                output_dir=output_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

    _validate_saved_artifacts(output_dir)

    LOGGER.info("Merge complete. Output directory is ready for deployment: %s", output_dir)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    torch_dtype = _resolve_dtype(args.torch_dtype)
    lora_dir = args.lora_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    LOGGER.info("Starting LoRA merge with dtype=%s", torch_dtype)
    merge_lora_adapter(
        lora_dir=lora_dir,
        base_model=args.base_model,
        output_dir=output_dir,
        max_seq_length=args.max_seq_length,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        force_peft=args.force_peft,
    )


if __name__ == "__main__":
    main()