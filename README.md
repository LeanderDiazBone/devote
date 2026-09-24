# Deep Epistemic Value Functions for Optimistic Exploration

This repository accompanies the paper *Deep Epistemic Value Functions for Optimistic Exploration*.

## Setup

From the repository root, create a Python environment and install the supplied requirements:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the uncertainty quantification experiments, use a separate environment and
install `requirements-uq.txt` there.

For the exploration and control runs, set the paths to writable directories:

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LOG_DIR=/absolute/path/to/paper_runs
export MUJOCO_GL=egl
mkdir -p "$LOG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
```

## Reproduce the experiments

Run the commands in the [pure exploration and complex control guide](experiments/paper/README.md) and the [uncertainty quantification guide](experiments/paper/uq/README.md) from the repository root.
