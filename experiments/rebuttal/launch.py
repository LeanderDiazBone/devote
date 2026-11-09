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
VERIFY_METHODS = ['bro', 'maxinforl']  # reproduce the reference papers on cheetah_run


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


def assemble(task, method, seed, logs_dir, verify=False):
    flags = {}
    paths = [
        os.path.join(PAPER_DIR, 'common.args'),
        os.path.join(PAPER_DIR, 'diagnostics/trajectories.args'),
        os.path.join(REBUTTAL_DIR, f'tasks/{task}.args'),
        os.path.join(PAPER_DIR, f'methods/{method}.args'),
        os.path.join(PAPER_DIR, f'seeds/{seed}.args'),
    ]
    if verify:
        paths = [
            os.path.join(PAPER_DIR, 'common.args'),
            os.path.join(PAPER_DIR, f'methods/{method}.args'),
            os.path.join(REBUTTAL_DIR, f'verify/{method}_dmc.args'),
            os.path.join(PAPER_DIR, f'seeds/{seed}.args'),
        ]
    for path in paths:
        flags.update(read_args(path))
    if not verify:
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
    elif args.mode == 'mpi':
        logs_dir = f'/fast/{USER_NAME}/{PROJECT_NAME}/'

    runs = [
        assemble(task, method, seed, logs_dir)
        for task in TASKS for method in METHODS for seed in SEEDS
    ]
    group = GROUP_PREFIX
    if args.verify:
        runs = [
            assemble('cheetah_run', method, seed, logs_dir, verify=True)
            for method in VERIFY_METHODS for seed in SEEDS
        ]
        group = 'verify'

    if args.mode == 'local':
        cmds = [f'cd {shlex.quote(REPO_ROOT)} && {cmd}' for _, cmd in runs]
    elif args.mode == 'euler':
        out_dir = os.path.join(logs_dir, 'slurm', group)
        cmds = []
        for flags, cmd in runs:
            out_file = os.path.join(out_dir, flags['exp_name'] + '_%j.out')
            wrap = f'hostname && nvidia-smi && cd {shlex.quote(REPO_ROOT)} && {cmd}'
            cmds.append(
                f'sbatch --time={args.num_hours}:59:00 --mem-per-cpu=32000 '
                f'--cpus-per-task {args.num_cpus} --account=ls_krausea '
                f'--gpus=rtx_4090:{args.num_gpus} --output={out_file} '
                f'--wrap={shlex.quote(wrap)}')
    elif args.mode == 'mpi':
        out_dir = os.path.join(logs_dir, 'condor', group)
        cmds, jobs = [], []
        for flags, cmd in runs:
            base = os.path.join(out_dir, flags['exp_name'])
            wrap = f'hostname && nvidia-smi && cd {shlex.quote(REPO_ROOT)} && {cmd}'
            submit = '\n'.join([
                'universe = vanilla',
                f'executable = {base}.sh',
                'getenv = HOME,USER,PATH,WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,MUJOCO_GL',
                f'request_cpus = {args.num_cpus}',
                f'request_gpus = {args.num_gpus}',
                f'request_memory = {32000 * args.num_cpus}MB',
                f'output = {base}_$(Cluster).out',
                f'error = {base}_$(Cluster).err',
                f'log = {base}_$(Cluster).log',
                'periodic_hold = (JobStatus == 2) && '
                f'((time() - EnteredCurrentStatus) > {(args.num_hours + 1) * 3600})',
                'queue', ''])
            jobs.append((base, wrap, submit))
            cmds.append(f'condor_submit_bid {args.bid} {shlex.quote(base + ".sub")}')
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
    if args.mode in ('euler', 'mpi'):
        os.makedirs(out_dir, exist_ok=True)
    if args.mode == 'mpi':
        for base, wrap, submit in jobs:
            with open(base + '.sh', 'w') as f:
                f.write(f'#!/bin/bash\n{wrap}\n')
            os.chmod(base + '.sh', 0o755)
            with open(base + '.sub', 'w') as f:
                f.write(submit)
    for cmd in cmds:
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='euler', choices=['euler', 'mpi', 'local'])
    parser.add_argument('--bid', type=int, default=50)
    parser.add_argument('--verify', default=False, action='store_true',
                        help='Launch the BRO/MaxInfoRL cheetah_run reproduction runs instead.')
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_hours', type=int, default=12)
    parser.add_argument('--print', default=False, action='store_true')
    main(parser.parse_args())
