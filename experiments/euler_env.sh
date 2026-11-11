# Source on Euler before launching (copy of the untracked setup.sh in the Euler checkout).
module load eth_proxy
source /cluster/home/ldiazbone/devote/.devote/bin/activate
export CUDA_ROOT="$(python -c 'from nvidia import cuda_nvcc; print(next(iter(cuda_nvcc.__path__)))')"
export WANDB_ENTITY=ldiazbone-eth-z-rich
export MUJOCO_GL=egl
export LOG_DIR=/cluster/home/ldiazbone/scratch/devote
