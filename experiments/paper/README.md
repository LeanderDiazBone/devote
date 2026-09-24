# Run the paper's exploration and control experiments

Uncertainty quantification arguments and commands are in [uq/README.md](uq/README.md).

Run these commands from the repository root in the Python environment used for
this project. Set `LOG_DIR` to an absolute, writable output directory:

```sh
export LOG_DIR=/absolute/path/to/paper_runs
```

Assemble one run by listing its argument files after `common.args`: trajectory
diagnostics, task, method, optional length scale, optional ablation, then seed.
`cat` passes their contents to the experiment launcher. Later files override
earlier values of the same flag.

For example, run PointMaze DEVOTE with seed 0:

```sh
cat experiments/paper/common.args \
    experiments/paper/diagnostics/trajectories.args \
    experiments/paper/tasks/pointmaze.args \
    experiments/paper/methods/devote.args \
    experiments/paper/seeds/0.args \
  | xargs python -m experiments.exp
```

## Main learning curves

The pure exploration tasks are `deepsea`, `pointmaze`, and `cheetah`. The
complex control tasks are `antmaze` and `pick_and_place`. Run the five methods
and five configured seeds for each task with:

```sh
for task in deepsea pointmaze cheetah antmaze pick_and_place; do
  for method in devote sac sombrl ts rnd; do
    for seed in 0 1 2 3 4; do
      cat experiments/paper/common.args \
          experiments/paper/diagnostics/trajectories.args \
          "experiments/paper/tasks/$task.args" \
          "experiments/paper/methods/$method.args" \
          "experiments/paper/seeds/$seed.args" \
        | xargs python -m experiments.exp
    done
  done
done
```

This launches 125 runs sequentially. To run just one condition, use the
PointMaze example above and substitute the task, method, and seed filenames.

## Pure exploration component ablations

Run the four configured ablations on DeepSea, PointMaze, and Cheetah. The
ablation runs use length scale 2.5 before the ablation file sets its final
group name. The main DEVOTE runs above supply each task's reference curve.

```sh
for task in deepsea pointmaze cheetah; do
  for ablation in no_rfn no_resets no_tud no_tud_opt; do
    for seed in 0 1 2 3 4; do
      cat experiments/paper/common.args \
          experiments/paper/diagnostics/trajectories.args \
          "experiments/paper/tasks/$task.args" \
          experiments/paper/methods/devote.args \
          experiments/paper/length_scales/2_5.args \
          "experiments/paper/ablations/$ablation.args" \
          "experiments/paper/seeds/$seed.args" \
        | xargs python -m experiments.exp
    done
  done
done
```

## Pure exploration length-scale sweeps

Run the DEVOTE sweep and the corresponding `no_tud_opt` sweep for length scales
1, 2.5, 5, and 7.5 on all three pure exploration tasks:

```sh
for task in deepsea pointmaze cheetah; do
  for scale in 1 2_5 5 7_5; do
    for seed in 0 1 2 3 4; do
      cat experiments/paper/common.args \
          experiments/paper/diagnostics/trajectories.args \
          "experiments/paper/tasks/$task.args" \
          experiments/paper/methods/devote.args \
          "experiments/paper/length_scales/$scale.args" \
          "experiments/paper/seeds/$seed.args" \
        | xargs python -m experiments.exp

      cat experiments/paper/common.args \
          experiments/paper/diagnostics/trajectories.args \
          "experiments/paper/tasks/$task.args" \
          experiments/paper/methods/devote.args \
          "experiments/paper/length_scales/$scale.args" \
          experiments/paper/ablations/no_tud_opt.args \
          "experiments/paper/seeds/$seed.args" \
        | xargs python -m experiments.exp
    done
  done
done
```


## Plasticity regularization

Run them with the shared paper arguments, DEVOTE method, and trajectory
diagnostics:

```sh
for strength in 0 0_1 0_25 0_5 0_75; do
  cat experiments/paper/common.args \
      experiments/paper/diagnostics/trajectories.args \
      experiments/paper/methods/devote.args \
      experiments/paper/plasticity/task.args \
      "experiments/paper/reset_strengths/$strength.args" \
      experiments/paper/seeds/0.args \
    | xargs python -m experiments.exp
done
```

## Stable value estimation



```sh
for task in cheetah cartpole; do
  for method in td tud; do
    for scale in 0_5 1 2_5 5 10; do
      cat experiments/paper/common.args \
          experiments/paper/policy_eval/common.args \
          "experiments/paper/policy_eval/$task.args" \
          "experiments/paper/policy_eval/$method.args" \
          experiments/paper/length_scales/$scale.args" \
          "experiments/paper/policy_eval/checkpoints/${task}_seed_0.args" \
          experiments/paper/seeds/0.args \
        | xargs python -m experiments.exp
    done
  done
done
```
