#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-dynamic_pricing_rl}"

mkdir -p "${PROJECT_ROOT}/src/envs"
mkdir -p "${PROJECT_ROOT}/src/core"
mkdir -p "${PROJECT_ROOT}/tests"

touch "${PROJECT_ROOT}/src/__init__.py"
touch "${PROJECT_ROOT}/src/envs/__init__.py"
touch "${PROJECT_ROOT}/src/core/__init__.py"
touch "${PROJECT_ROOT}/tests/__init__.py"

if [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  cat > "${PROJECT_ROOT}/requirements.txt" <<'REQS'
torch>=2.3.0
torchrl>=0.4.0
tensordict>=0.4.0
pytest>=8.0.0
REQS
fi

if [[ ! -f "${PROJECT_ROOT}/README.md" ]]; then
  cat > "${PROJECT_ROOT}/README.md" <<'README'
# Dynamic Pricing RL

GPU-native TorchRL environment for dynamic marketplace pricing experiments.
README
fi

echo "Project structure created at ${PROJECT_ROOT}"