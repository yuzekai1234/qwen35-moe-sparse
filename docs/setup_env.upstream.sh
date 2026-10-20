#!/bin/bash
# 在工作目录下建 .venv_q35（Python 3.11，torch 2.7.1+cu126 / transformers 5.19 / fla 0.5.2 / liger 0.8.4，见 requirements.txt）
# 用法: cd <工作目录> && bash env/setup_env.sh      （工作目录 = 拷过去的 hzh_q35_sparse 副本）
set -e
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh     # 或 pip install uv；装到 ~/.local/bin
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=900
uv venv --allow-existing --python 3.11 .venv_q35
# 镜像按机器能通的选: aliyun / tsinghua；torch cu126 轮子若镜像没有，加 --extra-index-url https://download.pytorch.org/whl/cu126
for i in 1 2 3; do uv pip install --python .venv_q35/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ -r env/requirements.txt && break; done
# Triton(fla kernel) 要找 libcuda.so: 建一个只含软链的目录，启动时 TRITON_LIBCUDA_PATH / LIBRARY_PATH 指向它
mkdir -p $HOME/libcuda && ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 $HOME/libcuda/libcuda.so
.venv_q35/bin/python -c "import torch,transformers,fla;print(torch.__version__,transformers.__version__,fla.__version__)"
echo SETUP_DONE
