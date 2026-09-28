# Deep Epistemic Value Functions for Optimistic Exploration

This repository implements the methods and experiments presented in the paper *Deep Epistemic Value Functions for Optimistic Exploration*.

## Getting started

### Installation 📂

After cloning the repository, create a Python 3.11 environment and install the dependencies from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements select CUDA-enabled JAX on Linux and CPU JAX on macOS.

For the uncertainty quantification experiments, create a separate environment:

```bash
python3.11 -m venv .venv-uq
source .venv-uq/bin/activate
python -m pip install -r requirements-uq.txt
```

### Run setup

Run all commands from the repository root. Before running the reinforcement learning experiments, activate the main environment and set writable paths and your Weights & Biases destination:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LOG_DIR=/absolute/path/to/paper_runs
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT=devote
mkdir -p "$LOG_DIR"
wandb login
```

For headless rendering on Linux, also set:

```bash
export MUJOCO_GL=egl
```

For uncertainty quantification experiments, activate `.venv-uq` and follow the data preparation steps in the [uncertainty quantification README](experiments/paper/uq/README.md) before training.

## Documentation

The [`experiments/paper`](experiments/paper) directory contains the argument files and commands for reproducing the experiments. For example, after completing the setup above, run DEVOTE on PointMaze with seed 0:

```bash
cat experiments/paper/common.args \
    experiments/paper/diagnostics/trajectories.args \
    experiments/paper/tasks/pointmaze.args \
    experiments/paper/methods/devote.args \
    experiments/paper/seeds/0.args \
  | xargs python -m experiments.exp
```

Refer to the experiment READMEs for the full commands and configurations:

- [Pure exploration and complex control](experiments/paper/README.md): main learning curves, component ablations, length-scale sweeps, plasticity regularization, and stable value estimation.
- [Uncertainty quantification](experiments/paper/uq/README.md): data preparation, prediction baselines, and uncertainty estimation experiments.

<!-- TODO:
## Citation
 Add the BibTeX citation for Deep Epistemic Value Functions for Optimistic Exploration.
 -->
