# ---- Config -----------------------------------------------------------------

UV        ?= uv
PY        ?= python3.12
VENV_DIR  ?= .venv
PYTHON    := $(VENV_DIR)/bin/python

# Force CPU for the smoke test
export CUDA_VISIBLE_DEVICES ?=
export OMP_NUM_THREADS ?= 8
export MKL_NUM_THREADS ?= 8

# ---- Phonies ----------------------------------------------------------------
.PHONY: help venv install-cpu install-dev check cpu-test clean clean-venv

# ---- Help -------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  venv         - create .venv with uv"
	@echo "  install-cpu  - install CPU wheels of torch + project (editable)"
	@echo "  install-dev  - install dev/profiling extras"
	@echo "  check        - import fsdp to verify installation"
	@echo "  cpu-test     - run CPU-only smoke test (world_size=1, fat=False)"
	@echo "  clean        - remove __pycache__ and logs"
	@echo "  clean-venv   - remove .venv"

# ---- Environment -------------------------------------------------------------
venv:
	$(UV) venv --python $(PY)
	@echo "Created venv in $(VENV_DIR)"

# Install CPU-only PyTorch first, then your project (editable).
install-cpu:
	$(UV) pip install --upgrade pip
	$(UV) pip install "torch>=2.3.0" torchvision torchaudio
	$(UV) pip install -e .

# Optional dev/profiling extras
install-dev: install-cpu
	$(UV) pip -p $(PYTHON) install -e ".[dev,profiling]"

# Quick import check
check: install-cpu
	$(PYTHON) -c "import fsdp; print('fsdp import OK')"

# CPU smoke test: runs the library entrypoint with world_size=1 (no NCCL)
cpu-test: install-cpu
	$(PYTHON) -c "from fsdp import run_on_cloud; print('[make] CPU smoke test'); run_on_cloud(world_size=1, fat=False, logdir='logs_cpu'); print('[make] Done. Trace in logs_cpu/')"

# ---- Housekeeping ------------------------------------------------------------
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	rm -rf logs_cpu || true

clean-venv:
	rm -rf $(VENV_DIR)
