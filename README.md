# Dynamic Pricing RL

GPU-native dynamic pricing simulator built with PyTorch and TorchRL, plus GRPO fine-tuning utilities.

## Project Layout

```
llm-marketplace/
├── dynamic_pricing_rl/
│   ├── __init__.py
│   ├── train_grpo.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── elasticity_math.py
│   └── envs/
│       ├── __init__.py
│       └── marketplace_env.py
├── tests/
│   ├── __init__.py
│   └── test_env_vectorization.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional explicit install commands for the GRPO stack:

   ```bash
   pip install unsloth vllm
   pip install trl==0.24.0 datasets==4.3.0 accelerate==1.13.0 peft==0.19.1 bitsandbytes==0.49.2 transformers==4.57.6
   ```

4. Run the vectorization smoke test:

   ```bash
   pytest -q tests/test_env_vectorization.py
   ```
