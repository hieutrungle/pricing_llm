"""FastAPI server for dynamic pricing inference with a merged Gemma-4 model via vLLM."""

from __future__ import annotations

import json
import logging
import re
from ast import literal_eval
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, CONFIG_MAPPING, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams


LOGGER = logging.getLogger(__name__)

ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]


@dataclass(frozen=True)
class ServerConfig:
    model_path: str = "/home/hieule/research/pricing_llm/tmp_gemma4-e4b-pricing-merged"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    dtype: ModelDType = "bfloat16"
    gpu_memory_utilization: float = 0.80
    max_model_len: int = 2048
    enforce_eager: bool = True
    trust_remote_code: bool = False
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


CONFIG = ServerConfig()

_JSON_BLOCK_PATTERN = re.compile(r"\{[^{}]*\}", flags=re.DOTALL)
_MULTIPLIER_PATTERN = re.compile(
    r"[\"']?(?:multiplier|price_multiplier)[\"']?\s*[:=]\s*"
    r"[\"']?(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)[\"']?",
    flags=re.IGNORECASE,
)

_GLOBAL_LLM: Optional[LLM] = None
_GLOBAL_TOKENIZER: Optional[PreTrainedTokenizerBase] = None


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


_setup_logging()


def _read_model_type_from_config(model_path: str) -> Optional[str]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.warning("Failed to parse model config at %s", config_path)
        return None

    model_type = payload.get("model_type")
    return str(model_type) if isinstance(model_type, str) else None


def _ensure_transformers_supports_model_type(model_path: str) -> None:
    model_type = _read_model_type_from_config(model_path)
    if model_type is None:
        return
    if model_type in CONFIG_MAPPING:
        return

    raise RuntimeError(
        "Incompatible transformers version for this checkpoint. "
        f"model_type={model_type!r} is not registered in transformers CONFIG_MAPPING. "
        "Upgrade transformers (recommended latest) in this environment, for example: "
        "pip install --upgrade transformers"
    )


def _get_llm() -> LLM:
    global _GLOBAL_LLM
    if _GLOBAL_LLM is not None:
        return _GLOBAL_LLM

    _ensure_transformers_supports_model_type(CONFIG.model_path)
    LOGGER.info("Initializing vLLM model from %s", CONFIG.model_path)
    _GLOBAL_LLM = LLM(
        model=CONFIG.model_path,
        tensor_parallel_size=CONFIG.tensor_parallel_size,
        dtype=CONFIG.dtype,
        gpu_memory_utilization=CONFIG.gpu_memory_utilization,
        max_model_len=CONFIG.max_model_len,
        enforce_eager=CONFIG.enforce_eager,
        trust_remote_code=CONFIG.trust_remote_code,
    )
    return _GLOBAL_LLM


def _get_prompt_tokenizer() -> PreTrainedTokenizerBase:
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is not None:
        return _GLOBAL_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.model_path,
        trust_remote_code=CONFIG.trust_remote_code,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Loaded tokenizer is not a PreTrainedTokenizerBase instance")
    _GLOBAL_TOKENIZER = tokenizer
    return _GLOBAL_TOKENIZER


class PriceRequest(BaseModel):
    riders: float = Field(..., ge=0.0)
    drivers: float = Field(..., ge=0.0)
    base_price: float = Field(..., ge=0.0)
    time: float = Field(..., ge=0.0, le=23.0)


class PriceResponse(BaseModel):
    multiplier: float


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Initialize vLLM after process bootstrap is complete."""
    try:
        _get_llm()
    except Exception:
        LOGGER.exception("vLLM initialization failed during FastAPI lifespan startup")
        raise
    yield


app = FastAPI(title="Dynamic Pricing LLM (Gemma-4-E4B)", lifespan=_lifespan)


def format_pricing_prompt(riders: float, drivers: float, base_price: float, time_of_day: float) -> str:
    """Format a prompt consistent with the GRPO training contract."""
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

    tokenizer = _get_prompt_tokenizer()
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        if isinstance(formatted, str):
            return formatted
        return str(formatted)

    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def parse_multiplier_from_text(text: str) -> float:
    """Extract multiplier from model text using JSON-first parsing with regex fallback."""
    parse_text = text.split("STOP", 1)[0]
    parsed_multiplier: Optional[float] = None

    for match in _JSON_BLOCK_PATTERN.finditer(parse_text):
        payload = match.group(0).strip()
        data: Any
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            try:
                data = literal_eval(payload)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(data, dict):
            continue
        for key in ("multiplier", "price_multiplier"):
            if key in data:
                try:
                    parsed_multiplier = float(data[key])
                except (TypeError, ValueError):
                    continue

    if parsed_multiplier is not None:
        return parsed_multiplier

    regex_match: Optional[re.Match[str]] = None
    for match in _MULTIPLIER_PATTERN.finditer(parse_text):
        regex_match = match
    if regex_match is not None:
        try:
            return float(regex_match.group("value"))
        except (TypeError, ValueError):
            pass

    raise ValueError("Could not parse a valid multiplier from model output")


@app.post("/predict_price", response_model=PriceResponse)
def predict_price(request: PriceRequest) -> PriceResponse:
    prompt = format_pricing_prompt(
        riders=request.riders,
        drivers=request.drivers,
        base_price=request.base_price,
        time_of_day=request.time,
    )

    sampling_params = SamplingParams(
        max_tokens=CONFIG.max_tokens,
        temperature=CONFIG.temperature,
        top_p=CONFIG.top_p,
        stop=["STOP"],
    )

    text = ""
    try:
        outputs: List[Any] = _get_llm().generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            raise HTTPException(status_code=502, detail="vLLM returned an empty completion")
        text = outputs[0].outputs[0].text
        multiplier = parse_multiplier_from_text(text)
    except HTTPException:
        raise
    except ValueError as error:
        LOGGER.warning("Failed to parse completion into multiplier: %s | raw=%r", error, text[:500] if isinstance(text, str) else text)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Inference request failed")
        raise HTTPException(status_code=500, detail="Inference failed") from error

    return PriceResponse(multiplier=multiplier)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        reload=False,
    )
