SHELL := /bin/bash
ENV_FILE := $(if $(wildcard .env),.env,.env.example)

.PHONY: install test lint serve benchmark-phase1 dashboard

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	black .
	isort .
	mypy config mizan tests

serve:
	set -a && . ./$(ENV_FILE) && set +a && \
	python -m vllm.entrypoints.openai.api_server \
		--model "$${VLLM__MODEL_NAME}" \
		--quantization "$${VLLM__QUANTIZATION}" \
		--gpu-memory-utilization "$${VLLM__GPU_MEMORY_UTILIZATION}" \
		--max-model-len "$${VLLM__MAX_MODEL_LEN}" \
		--max-num-batched-tokens "$${VLLM__MAX_NUM_BATCHED_TOKENS}" \
		--port "$${VLLM__PORT}" \
		--dtype "$${VLLM__DTYPE}"

benchmark-phase1:
	python scripts/benchmark_phase1.py

dashboard:
	@echo "Phase 6 only: run 'docker compose up -d' after monitoring files are added."
