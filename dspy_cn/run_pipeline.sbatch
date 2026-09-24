#!/bin/bash
#SBATCH --job-name=cn_llm_judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log

cd ~/CN-LLM-Framework
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# One-time isolated venv for the scoring client (venv, not conda -- avoids
# the memory-hungry solver that got OOM-killed on the login node earlier).
# Safe to re-run: skips rebuilding if it already exists.
if [ ! -f cn_eval_venv/bin/activate ]; then
  echo "Building cn_eval_venv..."
  python3 -m venv cn_eval_venv
  source cn_eval_venv/bin/activate
  pip install --upgrade pip
  pip install pandas PyYAML torch transformers sentence-transformers sentencepiece accelerate detoxify bert-score scikit-learn openai
  deactivate
fi

# Serve the judge model in its own pinned conda env.
# PYTHONNOUSERSITE=1 stops it from picking up ~/.local packages that could
# shadow vllm's pinned torch/transformers/triton versions.
(
  export PYTHONNOUSERSITE=1
  conda activate vllm
  export no_proxy="localhost,127.0.0.1"
  export NO_PROXY="localhost,127.0.0.1"
  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192
) > vllm_server.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to come up..."
for i in $(seq 1 90); do
  if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM ready after $((i*10))s"
    break
  fi
  sleep 10
done

# Run the scoring client in its isolated venv.
# PYTHONUNBUFFERED=1 forces print() to flush immediately -- otherwise
# Python fully buffers stdout when it's redirected to a file, and the
# per-batch progress lines don't show up in the log until much later
# (this made an earlier run look "stuck" when it was actually progressing).
(
  source cn_eval_venv/bin/activate
  export PYTHONUNBUFFERED=1
  python -m dspy_cn.evaluate_csv --dataset baseline
  python -m dspy_cn.evaluate_csv --dataset enhanced
)

kill $VLLM_PID
