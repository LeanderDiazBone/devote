"""Assemble and launch the sparse-reward DMC calibration runs.

Each run concatenates the paper argument files in the same order as the paper
README (common, diagnostics, task, method, seed; later files override earlier
ones), with the task files taken from experiments/rebuttal/tasks, and runs
`python -m experiments.exp` from the repository root.

    python experiments/rebuttal/launch.py --print          # show the commands
    python experiments/rebuttal/launch.py --mode local     # run them sequentially
    python experiments/rebuttal/launch.py --mode euler     # submit with sbatch
"""
import argparse
import os
import shlex
import sys


REBUTTAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(REBUTTAL_DIR))
PAPER_DIR = os.path.join(REPO_ROOT, 'experiments', 'paper')
USER_NAME = 'ldiazbone'
PROJECT_NAME = 'devote'

GROUP_PREFIX = 'sparse_dmc_calib'
TASKS = ['cartpole_swingup_sparse', 'finger_turn_hard', 'acrobot_swingup_sparse']
METHODS = ['devote', 'sac', 'sombrl', 'ts', 'rnd']
SEEDS = [0, 1, 2]


def read_args(path):
    flags = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line[2:].split('=', 1)
            flags[key] = value
    return flags


def assemble(task, method, seed, logs_dir):
    flags = {}
    for path in [
        os.path.join(PAPER_DIR, 'common.args'),
        os.path.join(PAPER_DIR, 'diagnostics/trajectories.args'),
        os.path.join(REBUTTAL_DIR, f'tasks/{task}.args'),
        os.path.join(PAPER_DIR, f'methods/{method}.args'),
        os.path.join(PAPER_DIR, f'seeds/{seed}.args'),
    ]:
        flags.update(read_args(path))
    flags['group_name'] = f'{GROUP_PREFIX}_{flags["group_name"]}'
    flags['exp_name'] = f'{method}_{task}_s{seed}'
    if logs_dir:
        flags['logs_dir'] = logs_dir
    cmd = f'{shlex.quote(sys.executable)} -u -m experiments.exp'
    for key, value in flags.items():
        cmd += f' --{key}={shlex.quote(value)}'
    return flags, cmd


def main(args):
    logs_dir = None
    if args.mode == 'euler':
        logs_dir = f'/cluster/scratch/{USER_NAME}/{PROJECT_NAME}/'

    runs = [
        assemble(task, method, seed, logs_dir)
        for task in TASKS for method in METHODS for seed in SEEDS
    ]

    if args.mode == 'local':
        cmds = [f'cd {shlex.quote(REPO_ROOT)} && {cmd}' for _, cmd in runs]
    elif args.mode == 'euler':
        out_dir = os.path.join(logs_dir, 'slurm', GROUP_PREFIX)
        cmds = []
        for flags, cmd in runs:
            out_file = os.path.join(out_dir, flags['exp_name'] + '_%j.out')
            wrap = f'hostname && nvidia-smi && cd {shlex.quote(REPO_ROOT)} && {cmd}'
            cmds.append(
                f'sbatch --time={args.num_hours}:59:00 --mem-per-cpu=32000 '
                f'--cpus-per-task {args.num_cpus} --account=ls_krausea '
                f'--gpus=rtx_4090:{args.num_gpus} --output={out_file} '
                f'--wrap={shlex.quote(wrap)}')
    else:
        raise NotImplementedError(args.mode)

    if args.print:
        for cmd in cmds:
            print(cmd)
        print(f'# {len(cmds)} runs')
        return
    answer = input(f'About to launch {len(cmds)} runs ({args.mode}). Proceed? [yes/no] ')
    if answer != 'yes':
        return
    if args.mode == 'euler':
        os.makedirs(out_dir, exist_ok=True)
    for cmd in cmds:
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='euler', choices=['euler', 'local'])
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_hours', type=int, default=12)
    parser.add_argument('--print', default=False, action='store_true')
    main(parser.parse_args())
