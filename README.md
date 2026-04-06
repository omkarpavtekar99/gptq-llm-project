# gptq-LLM-Project

gptq-LLM-Project is a production-grade LLM Quality Observatory designed to run locally in WSL2 and grow phase by phase across:

- LLM serving and performance profiling
- speech AI pipelines
- evaluation and regression tracking
- drift detection
- routing and cost-quality experiments
- monitoring and dashboards

## Phase 0

Phase 0 scaffolds the project so later phases share one consistent package layout, one settings system, and one structured logging implementation.

### Included in this scaffold

- `pyproject.toml` for packaging, linting, typing, and pytest settings
- `requirements.txt` with pinned base dependencies
- `.env.example` with all planned config keys and defaults
- `config/settings.py` using `pydantic-settings`
- `mizan/logging_setup.py` with a reusable JSON formatter and logger factory
- `Makefile` with install, test, lint, serve, and dashboard targets
- matching tests for the Phase 0 config and logging modules

### WSL2 prerequisite check from this machine

- `python3 --version` returned `Python 3.12.3`
- `nvcc --version` returned `command not found`
- `pip --version` returned `command not found`

Install the missing WSL2 prerequisites before Phase 1:

```bash
sudo apt update
sudo apt install -y software-properties-common curl wget build-essential
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
python3.11 --version
pip3 --version
```

If CUDA toolkit is still missing in WSL2, install a vLLM-compatible toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1
nvcc --version
```

### Run Phase 0

From the `mizan_project/` folder:

```bash
python3.11 -m venv ~/mizan-env
source ~/mizan-env/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Phase 1

Phase 1 adds local vLLM serving, a full parameter sweep benchmark, MLflow logging, and a quality comparison between the GPTQ production path and a CPU reference run.

### Assumption

The CPU reference path uses a configurable Hugging Face Transformers model from `VLLM__CPU_REFERENCE_MODEL_NAME`, defaulting to `Qwen/Qwen2.5-7B-Instruct`. It is intended as a slow quality reference and may need a smaller CPU model on low-memory systems.

### WSL2 verification

Run these commands before installing or launching vLLM:

```bash
nvidia-smi
nvcc --version
python3.11 --version
```

If `nvcc` is missing or reports a toolkit older than 11.8, install a newer CUDA toolkit inside WSL2:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1
nvcc --version
```

### Install Phase 1 dependencies

```bash
source ~/mizan-env/bin/activate
cd /mnt/c/Users/omkar/OneDrive/Documents/New\ project/mizan_project
pip install -e ".[dev]"
```

### Start vLLM

The default `make serve` target launches the confirmed serving profile:

```bash
make serve
```

Equivalent command:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --quantization gptq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --port 8000 \
  --dtype float16
```

### Verify the server

Health check:

```bash
curl http://127.0.0.1:8000/v1/models
```

Sample completion:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Reply with the word healthy."}],
    "temperature": 0.0,
    "max_tokens": 16
  }'
```

### Run the benchmark

If you already started vLLM manually:

```bash
python scripts/benchmark_phase1.py
```

If you want the script to restart the server across the sweep:

```bash
python scripts/benchmark_phase1.py --manage-server
```

### Outputs

- `results/phase1_benchmark.csv`: one row per sweep combination with TTFT, ITL, throughput, and peak VRAM
- `results/phase1_summary.md`: the chosen config, quantization comparison, and lock-in recommendation
- MLflow artifacts and metrics under the `mizan` experiment
