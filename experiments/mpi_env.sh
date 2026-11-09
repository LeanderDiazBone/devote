# Source from the repository root on the MPI cluster (cluster_runner `setup`).
# Creates/updates the shared environment and sets the variables from the README.
if [ ! -d /fast ]; then
    echo "mpi_env.sh: expected the MPI cluster (/fast not found)." >&2
    return 1
fi
command -v uv >/dev/null 2>&1 || { echo "mpi_env.sh: uv not found on PATH." >&2; return 1; }

export DEVOTE_HOME="${DEVOTE_HOME:-/fast/$USER/devote}"
export DEVOTE_VENV="${DEVOTE_VENV:-/fast/$USER/envs/devote}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/fast/$USER/cache/uv}"
export UV_LINK_MODE=copy

if [ ! -x "$DEVOTE_VENV/bin/python" ]; then
    uv venv --python 3.11 "$DEVOTE_VENV" || return 1
fi
uv pip install --python "$DEVOTE_VENV/bin/python" -r requirements.txt || return 1
source "$DEVOTE_VENV/bin/activate" || return 1

# Runs write beneath the batch OUTPUT_DIR when launched by cluster_runner.
export LOG_DIR="${OUTPUT_DIR:-$DEVOTE_HOME/logs}"
export WANDB_ENTITY="${WANDB_ENTITY:-ldiazbone-eth-z-rich}"
export WANDB_PROJECT="${WANDB_PROJECT:-devote}"
export MUJOCO_GL=egl
mkdir -p "$DEVOTE_HOME" "$LOG_DIR" || return 1
