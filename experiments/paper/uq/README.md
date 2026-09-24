# Uncertainty quantification paper runs

Run these commands from the repository root. `experiments/uq_exp.py` accepts a
stage followed by ordinary command-line options or `@` argument files. Later
files override earlier options. The stages write their data beneath
`experiments/uncertainty_quantification/data/`.

Prepare the UCI datasets and the synthetic GP data:

```sh
UQ=experiments/paper/uq
python -m experiments.uq_exp download "@$UQ/uci_data.args"
python -m experiments.uq_exp splits "@$UQ/gap_splits.args"
python -m experiments.uq_exp synth "@$UQ/synthetic_gp.args"
```

Train the single-MLP and bootstrap-ensemble prediction baselines:

```sh
python -m experiments.uq_exp predictions "@$UQ/mean_mlp.args"
python -m experiments.uq_exp predictions "@$UQ/bootstrap_ensemble.args"
```

Train the ENN uncertainty estimators across modes, MLP widths and depths, and
RFN length scales:

```sh
for mode in ensemble input output; do
  for width in 50 100 200 400; do
    for depth in 1 2; do
      python -m experiments.uq_exp uncertainties \
          "@$UQ/enn_common.args" "@$UQ/enn_mlp.args" \
          "@$UQ/modes/$mode.args" \
          "@$UQ/mlp_widths/$width.args" \
          "@$UQ/mlp_depths/$depth.args"
    done
  done
  for length in 0_25 0_5 1 2 4; do
    python -m experiments.uq_exp uncertainties \
        "@$UQ/enn_common.args" "@$UQ/enn_rfn.args" \
        "@$UQ/modes/$mode.args" \
        "@$UQ/rfn_lengths/$length.args"
  done
done
```

`predictions` and `uncertainties` process every split currently in the data
directory. Add `--datasets NAME` to limit a run to one or more datasets. Use
`python -m experiments.uq_exp STAGE --help` for stage-specific options.
