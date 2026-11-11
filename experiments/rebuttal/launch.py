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
REWARD_FREE_PREFIX = 'reward_free_dmc'  # reward_scale=0, only coverage is of interest
REWARD_FREE_TASKS = TASKS + ['reacher_hard', 'hopper_hop', 'cheetah_run', 'pointmaze']
REWARD_FREE_METHODS = METHODS + VERIFY_METHODS + ['maxinforl_shared']  # maxinforl with the shared SAC settings
DEVOTE_LENGTHS = [f'devote_length_{s}' for s in ['2_5', '5', '7_5']]  # 'devote' itself is length scale 1
REWARD_FREE_RERUN = {  # second reward-free round
    'hopper_hop': REWARD_FREE_METHODS + DEVOTE_LENGTHS,  # coverage grid fixed after the first round
    'cheetah_run': ['maxinforl_shared'],
    'pointmaze': ['maxinforl', 'maxinforl_shared', 'bro'],
}
REWARD_PREFIX = 'reward_dmc'  # the reward-free DMC tasks with their task-file reward_scale, paper RL setup
REWARD_TASKS = TASKS + ['reacher_hard', 'hopper_hop', 'cheetah_run']
MLP_PRIOR_TASKS = ['pointmaze', 'cheetah_run']  # reward-free coverage, DEVOTE with a tuned MLP prior instead of the RFN
MLP_PRIOR_METHODS = [f'mlp_prior_{act}_length_{s}' for act in ['relu', 'sigmoid', 'tanh'] for s in ['0_5', '1', '2_5']]
SOFT_RESET_TASKS = ['pointmaze', 'cheetah_run']  # reward-free coverage, baselines with DEVOTE's soft resets
SOFT_RESET_METHODS = [f'{m}_soft_reset' for m in ['sac', 'ts', 'rnd', 'sombrl']]
COMPLEX_PREFIX = 'complex_control_v2'  # v2: 192-wide actor/critic, paper Euler launch settings
COMPLEX_TASKS = ['antmaze', 'pick_and_place']  # task files from experiments/paper/tasks
COMPLEX_METHODS = [f'devote_length_{s}' for s in ['1', '2_5', '5', '7_5']]
COMPLEX_BASELINES = VERIFY_METHODS  # paper method args on the complex control tasks, no verify/*_dmc.args
MPI_GPU = 'A100'  # matched against the slot's GPUs_DeviceName (A100-SXM4-40GB/80GB)
MPI_EXCLUDE = ['i201', 'g155']  # JAX failed to find the assigned GPU on these nodes
EULER_EXCLUDE = ['eu-g6-064']  # as in the paper's optimistic-curiosity launcher: persistent CUDA_ERROR_UNKNOWN


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


def assemble(task, method, seed, logs_dir, verify=False, reward_free=False, rewarded=False):
    flags = {}
    soft_reset = method.endswith('_soft_reset')  # sac_soft_reset: sac with the soft resets of common.args
    base, _, scale = method.removesuffix('_soft_reset').partition('_length_')  # devote_length_2_5: devote + length_scales/2_5.args
    act = base.partition('mlp_prior_')[2]  # mlp_prior_tanh_length_2_5: devote + mlp_prior/tanh.args at length 2.5
    base = 'devote' if act else base
    paths = [
        os.path.join(PAPER_DIR, 'common.args'),
        os.path.join(PAPER_DIR, 'diagnostics/trajectories.args'),
        os.path.join(PAPER_DIR if task in COMPLEX_TASKS else REBUTTAL_DIR, f'tasks/{task}.args'),
        os.path.join(PAPER_DIR, f'methods/{base}.args'),
        os.path.join(PAPER_DIR, f'seeds/{seed}.args'),
    ]
    if scale and not act:
        paths.insert(-1, os.path.join(PAPER_DIR, f'length_scales/{scale}.args'))
    if act:
        paths.insert(-1, os.path.join(REBUTTAL_DIR, f'mlp_prior/{act}.args'))
    if verify:
        paths = [
            os.path.join(PAPER_DIR, 'common.args'),
            os.path.join(PAPER_DIR, f'methods/{method}.args'),
            os.path.join(REBUTTAL_DIR, f'verify/{method}_dmc.args'),
            os.path.join(PAPER_DIR, f'seeds/{seed}.args'),
        ]
    for path in paths:
        flags.update(read_args(path))
    if (reward_free or task in COMPLEX_TASKS) and base != 'bro':  # paper pure-exploration/complex-control runs used 192-wide actor/critic; BRO keeps its own widths
        flags['actor_units'] = flags['critic_units'] = '192'
    if act:
        flags['prior_length_scale'] = scale.replace('_', '.')
        flags['group_name'] = f'devote_{method}_old_setup'  # paper setup with the TS/RND 2-layer MLP prior; only act and input scale vary
    if soft_reset:
        flags['reset_soft_alpha'] = read_args(paths[0])['reset_soft_alpha']  # undo the method's reset_soft_alpha=0
        if flags['policy_mode'] == 'actor':  # SAC/TS have none of DEVOTE's reset modules; reset their acting actor instead
            flags['reset_keys'] = flags['reset_keys'].replace('|exp_actor[^/]*)', '|exp_actor[^/]*|actor_sac)')
        flags['group_name'] += '_soft_reset'
    if reward_free:
        flags['reward_scale'] = '0'
        flags['group_name'] = f'{REWARD_FREE_PREFIX}_{flags["group_name"]}'
    elif rewarded:
        flags['group_name'] = f'{REWARD_PREFIX}_{flags["group_name"]}'
    elif task in COMPLEX_TASKS:
        flags['group_name'] = f'{COMPLEX_PREFIX}_{flags["group_name"]}'
    elif not verify:
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
    if args.reward_free:
        runs = [
            assemble(task, method, seed, logs_dir, reward_free=True)
            for task in REWARD_FREE_TASKS for method in REWARD_FREE_METHODS for seed in SEEDS
        ]
        group = REWARD_FREE_PREFIX
    if args.reward_free_rerun:
        runs = [
            assemble(task, method, seed, logs_dir, reward_free=True)
            for task, methods in REWARD_FREE_RERUN.items() for method in methods for seed in SEEDS
        ]
        group = REWARD_FREE_PREFIX
    if args.mlp_prior:
        runs = [
            assemble(task, method, seed, logs_dir, reward_free=True)
            for task in MLP_PRIOR_TASKS for method in MLP_PRIOR_METHODS for seed in SEEDS
        ]
        group = REWARD_FREE_PREFIX
    if args.soft_reset:
        runs = [
            assemble(task, method, seed, logs_dir, reward_free=True)
            for task in SOFT_RESET_TASKS for method in SOFT_RESET_METHODS for seed in SEEDS
        ]
        group = REWARD_FREE_PREFIX
    if args.complex_control:
        runs = [
            assemble(task, method, seed, logs_dir)
            for task in COMPLEX_TASKS for method in COMPLEX_METHODS for seed in SEEDS
        ]
        group = COMPLEX_PREFIX
    if args.complex_baselines:
        runs = [
            assemble(task, method, seed, logs_dir)
            for task in COMPLEX_TASKS for method in COMPLEX_BASELINES for seed in SEEDS
        ]
        group = COMPLEX_PREFIX
    if args.rewarded:
        runs = [
            assemble(task, method, seed, logs_dir, rewarded=True)
            for task in REWARD_TASKS for method in METHODS for seed in SEEDS
        ]
        group = REWARD_PREFIX

    if args.mode == 'local':
        cmds = [f'cd {shlex.quote(REPO_ROOT)} && {cmd}' for _, cmd in runs]
    elif args.mode == 'euler':
        out_dir = os.path.join(logs_dir, 'slurm', group)
        cmds = []
        for flags, cmd in runs:
            out_file = os.path.join(out_dir, flags['exp_name'] + '_%j.out')
            wrap = f'hostname && nvidia-smi && cd {shlex.quote(REPO_ROOT)} && {cmd}'
            cmds.append(
                f'sbatch --time={args.num_hours}:59:00 --mem-per-cpu=32000 '  # paper launcher settings
                f'--cpus-per-task {args.num_cpus or 1} --account=ls_krausea --exclude={",".join(EULER_EXCLUDE)} '
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
                f'request_cpus = {args.num_cpus or 4}',
                f'request_gpus = {args.num_gpus}',
                'request_memory = 32000MB',
                f'requirements = regexp("{MPI_GPU}", TARGET.GPUs_DeviceName)' + ''.join(
                    f' && (TARGET.Machine != "{node}.internal.cluster.is.localnet")'
                    for node in MPI_EXCLUDE),
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
    parser.add_argument('--reward_free', default=False, action='store_true',
                        help='Launch the reward-free coverage runs (all methods) instead.')
    parser.add_argument('--reward_free_rerun', default=False, action='store_true',
                        help='Launch the second reward-free round (REWARD_FREE_RERUN) instead.')
    parser.add_argument('--mlp_prior', default=False, action='store_true',
                        help='Launch the reward-free MLP-prior sweep (MLP_PRIOR_METHODS) instead.')
    parser.add_argument('--soft_reset', default=False, action='store_true',
                        help='Launch the reward-free baselines with DEVOTE soft resets (SOFT_RESET_METHODS) instead.')
    parser.add_argument('--complex_control', default=False, action='store_true',
                        help='Launch the DEVOTE length-scale sweep on antmaze/pick_and_place instead.')
    parser.add_argument('--complex_baselines', default=False, action='store_true',
                        help='Launch BRO/MaxInfoRL (COMPLEX_BASELINES) on antmaze/pick_and_place instead.')
    parser.add_argument('--rewarded', default=False, action='store_true',
                        help='Launch all METHODS with rewards on the six reward-free DMC tasks (REWARD_TASKS) instead.')
    parser.add_argument('--num_cpus', type=int, default=None, help='Default: 1 on Euler (paper setting), 4 on MPI.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_hours', type=int, default=12)
    parser.add_argument('--print', default=False, action='store_true')
    main(parser.parse_args())
