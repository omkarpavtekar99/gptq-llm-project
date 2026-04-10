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

## Phase 2

Phase 2 adds a hardware-aware speech pipeline for VAD, ASR, diarization, and optional TTS. The implementation uses lazy loading and optional runtime checks so your WSL setup only pays for the stage you actually invoke.

### Included in this phase

- `mizan/speech/vad.py`: Silero VAD wrapper with energy gating and short-gap merging
- `mizan/speech/asr.py`: faster-whisper wrapper with streamed transcript chunks, WER scoring, and MLflow latency logging
- `mizan/speech/diarization.py`: pyannote diarization wrapper with RTTM export and DER scoring
- `mizan/speech/tts.py`: Kokoro TTS wrapper with P50/P95/P99 streaming latency tracking
- `mizan/speech/pipeline.py`: unified `SpeechPipeline` chaining VAD → ASR → Diarization → optional TTS
- `mizan/speech/benchmarking.py`: manifest-driven benchmark orchestration
- `scripts/benchmark_phase2.py`: CLI entrypoint writing the CSV and Markdown report

The audio loader uses `soundfile` first so it can read WAV variants that Python's built-in `wave` module rejects, including extensible WAV files.

### Install Phase 2 dependencies

```bash
source ~/mizan-env/bin/activate
cd /mnt/c/Users/omkar/OneDrive/Documents/New\ project/mizan_project
pip install -e ".[dev]"
```

If you plan to use pyannote diarization, export your Hugging Face token through `.env`:

```bash
DIARIZATION__HF_TOKEN=your_token_here
```

`pyannote.audio` is pinned to the 3.1.1 line here because the newer 4.x releases require a newer `opentelemetry-sdk` than the one allowed by the current `vllm==0.8.5.post1` stack.

`transformers` is intentionally kept as a compatible 4.x range instead of an exact pin so the eval stack can resolve alongside `vllm`, `sentence-transformers`, and the other Phase 3 packages without breaking the serving path.

ChromaDB is pinned to the newer 1.5.6 line because the older 0.5.x releases cap `tokenizers` below the version required by `vllm==0.8.5.post1`.

### Prepare the benchmark manifest

Copy the example manifest and point it at your 10-file evaluation set:

```bash
cp data/phase2_benchmark_manifest.example.json data/phase2_benchmark_manifest.json
```

Each manifest entry should include:

- `sample_id`
- `audio_path`
- `reference_transcript`
- `reference_rttm_path` when DER should be measured

Relative `audio_path` and `reference_rttm_path` values are resolved from the manifest file’s own folder.

### Run the Phase 2 benchmark

```bash
python scripts/benchmark_phase2.py
```

Or with the Make target:

```bash
make benchmark-phase2
```

### Outputs

- `results/phase2_speech_benchmark.csv`: one row per sample/model pair with WER, DER, latency, and VRAM delta
- `results/phase2_summary.md`: model-level averages and the recommended Whisper profile
- MLflow nested runs under the main project experiment

## Phase 3

Phase 3 adds the evaluation framework: immutable golden sets, a versioned prompt registry, ROUGE-L and BERTScore scoring, Qwen-as-judge scoring through the local vLLM endpoint, a regression harness, optional RAG evaluation, and online shadow evaluation logging.

### Included in this phase

- `mizan/eval/golden_sets.py`: JSON-backed immutable golden set storage
- `mizan/eval/prompt_registry.py`: YAML-backed versioned prompt templates with diff support
- `mizan/eval/engine.py`: generation, ROUGE-L, BERTScore, judge scoring, error taxonomy, and MLflow logging
- `mizan/eval/rag_eval.py`: public-domain sample corpus bootstrap, ChromaDB indexing, and RAGAS scoring
- `mizan/eval/regression.py`: baseline persistence and threshold checks
- `mizan/eval/shadow.py`: FastAPI middleware for 10% asynchronous shadow evaluation and SQLite logging
- `tests/test_regression.py`: regression harness with `--update-baseline`

### Install Phase 3 dependencies

```bash
source ~/mizan-env/bin/activate
cd /mnt/c/Users/omkar/OneDrive/Documents/New\ project/mizan_project
pip install -e ".[dev]"
```

### Golden sets and prompt templates

The repo now includes:

- `data/golden_sets/core_eval__1.0.0.json`
- `config/prompts/judge_response__1.0.0.yaml`
- `data/baselines/regression_baseline.json`

You can publish additional golden sets with `GoldenSetStore.publish()` and prompt versions with `PromptRegistry.save()`. Published versions are immutable.

### Run the regression harness

To refresh the baseline:

```bash
pytest tests/test_regression.py -v --update-baseline
```

To validate against the stored baseline:

```bash
pytest tests/test_regression.py -v
```

### RAG evaluation notes

The `RagEvaluator` writes a 20-document public-domain sample corpus under `data/rag_docs/` and uses:

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- ChromaDB as the local vector store
- RAGAS for `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`

### Shadow evaluation

Attach `ShadowEvaluationMiddleware` to a FastAPI app with a shared `EvalEngine` instance. The middleware samples 10% of requests by default, scores them asynchronously, writes results to MLflow, and appends them to `data/shadow_log.db`.

### Outputs

- MLflow eval runs under the shared experiment
- `data/baselines/regression_baseline.json`: stored regression baseline
- `data/shadow_log.db`: local shadow-eval log database
- Chroma persistence under `data/chroma/`

## Phase 4

Phase 4 adds drift detection and root-cause analysis around the Phase 3 evaluation outputs. The detector embeds eval outputs, compares them against a stored baseline, emits alert metadata, and provides a simulation path to demonstrate the signal firing on a prompt regression.

### Included in this phase

- `mizan/analysis/models.py`: drift and RCA report models
- `mizan/analysis/drift.py`: embedding baseline generation, JS divergence, cosine drift, rolling judge score, and alert logic
- `mizan/analysis/rca.py`: prompt-version, model-variant, and retrieval-config comparison helpers
- `scripts/simulate_drift.py`: baseline run, intentional prompt degradation, drift detection, and RCA output

### Baseline files

- `data/baselines/drift_baseline.pkl`: persisted embedding baseline written by `DriftDetector.recompute_baseline()`
- `results/phase4_drift_simulation.md`: simulation output written by `scripts/simulate_drift.py`

### Run the Phase 4 tests

```bash
pytest tests/analysis -v
```

### Run the drift simulation

Make sure your local vLLM server is running first, then:

```bash
python scripts/simulate_drift.py
```

### Outputs

- MLflow drift metrics and serialized report payloads
- `results/phase4_drift_simulation.md`: human-readable simulation summary
