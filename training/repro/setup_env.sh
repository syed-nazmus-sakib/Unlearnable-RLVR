#!/bin/bash
#
# Recreate the conda env this repo was reproduced in.
#
# Target env: `vllm` (Python 3.11). It was chosen because it was empty, so nothing
# was displaced. The other candidate, `taton`, is a live env on torch 2.9.1 /
# transformers 4.57.3 / numpy 2.2.6; this repo pins torch 2.6.0 / transformers
# 4.51.1 / numpy 1.26.4, so installing here would have downgraded it.
#
# Usage:  bash repro/setup_env.sh [env_name]

set -euo pipefail

ENV_NAME="${1:-vllm}"
PY="$(conda run -n "$ENV_NAME" which python 2>/dev/null || echo "$HOME/miniconda3/envs/$ENV_NAME/bin/python")"
PIP="${PY%/python}/pip"

echo "Installing into: $PY"
"$PY" -V

# 1. torch first, pinned, from the cu124 index. Installing it up front stops the
#    later resolves from pulling a different build.
"$PIP" install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. Inference + training stack. sglang 0.4.6.post1 is the rollout backend
#    (verl imports sglang.srt.entrypoints.verl_engine.VerlEngine).
#    hydra-core/omegaconf are the dev releases the repo pins.
"$PIP" install \
    "sglang==0.4.6.post1" "sgl-kernel==0.1.0" "xgrammar==0.1.17" "outlines==0.1.11" \
    "ray[default]==2.10.0" "tensordict==0.9.1" "transformers==4.51.1" "tokenizers==0.21.4" \
    "numpy==1.26.4" "datasets==4.0.0" "peft==0.17.0" "accelerate==1.10.0" \
    "pyarrow==21.0.0" "pandas==2.3.1" \
    "hydra-core==1.4.0.dev1" "omegaconf==2.4.0.dev3" "codetiming==1.4.0" "wandb==0.21.1" \
    "math-verify==0.6.0" "latex2sympy2_extended==1.0.9" "sympy==1.13.1" "word2number==1.1" \
    "matplotlib==3.10.9" "scikit-learn==1.7.2" "openai==1.99.5" "google-genai==1.60.0" \
    "dill==0.3.8" "pylatexenc" "torchdata==0.11.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 3. flash-attn is NOT optional: verl/workers/actor/dp_actor.py imports
#    flash_attn.bert_padding at module top level, so the FSDP actor path needs it.
#    Use the prebuilt wheel matching torch 2.6 / cp311 / CXX11 ABI False
#    (torch 2.6.0+cu124 from download.pytorch.org reports _GLIBCXX_USE_CXX11_ABI=False).
#    Building from source instead takes hours and more RAM than this box has.
"$PIP" install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
"$PIP" install "einops==0.8.1" "hf_transfer==0.1.9"

# 4. sglang's srt runtime deps. Plain `pip install sglang` does not pull these
#    (they live in the [srt] extra, which would also drag in a torch pin), so
#    install them --no-deps at the versions in requirements.txt.
#    flashinfer-python is deliberately skipped: it is sdist-only at 0.2.3 and
#    sglang falls back to a working attention backend without it on sm86.
"$PIP" install --no-deps \
    "pyzmq==27.0.1" "uvicorn==0.35.0" "uvloop==0.21.0" "fastapi==0.116.1" "starlette==0.47.2" \
    "python-multipart==0.0.20" "partial-json-parser==0.2.1.1.post6" "llguidance==0.7.30" \
    "compressed-tensors==0.10.2" "torchao==0.12.0" "torch_memory_saver==0.0.8" \
    "modelscope==1.28.2" "soundfile==0.13.1" "decord==0.6.0" "litellm==1.75.3" \
    "nvidia-ml-py==12.575.51" "pynvml==12.0.0" "python-dotenv==1.1.1" \
    "cachetools==5.5.2" "rsa==4.9.1" \
    "opentelemetry-api==1.36.0" "opentelemetry-sdk==1.36.0" "opentelemetry-proto==1.36.0" \
    "opentelemetry-semantic-conventions==0.57b0" "opentelemetry-exporter-prometheus==0.57b0" \
    "anthropic==0.62.0" "pybind11==3.0.0" "async-timeout==5.0.1" "tomli==2.4.0"

echo
echo "=== verifying imports ==="
"$PY" - <<'EOF'
import torch, numpy, transformers, ray, sglang, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
print("numpy", numpy.__version__, "transformers", transformers.__version__)
print("ray", ray.__version__, "sglang", sglang.__version__, "flash_attn", flash_attn.__version__)
from sglang.srt.entrypoints.verl_engine import VerlEngine
from flash_attn.bert_padding import pad_input, unpad_input
from math_verify import parse, verify
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
print("all repo imports OK")
EOF
