#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
uv venv --allow-existing --python 3.11 .venv_q38
uv pip install --python .venv_q38/bin/python \
  --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  -r requirements.txt
mkdir -p libcuda
CUDA_DRIVER=${CUDA_DRIVER:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}
test -f "$CUDA_DRIVER"
ln -sfn "$CUDA_DRIVER" libcuda/libcuda.so
.venv_q38/bin/python -c 'import torch,transformers,fla; print(torch.__version__,transformers.__version__,fla.__version__); assert torch.cuda.is_available(); assert torch.version.cuda == "12.6"'
echo SETUP_DONE
