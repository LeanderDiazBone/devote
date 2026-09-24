"""Run a UQ benchmark stage with optional paper argument files.

Run from the repository root, for example:
    python -m experiments.uq_exp synth @experiments/paper/uq/synthetic_gp.args
"""

from __future__ import annotations

import argparse
import importlib
import shlex
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


STAGES = {
    "download": "download_data",
    "splits": "generate_splits",
    "synth": "generate_synth",
    "predictions": "generate_predictions",
    "uncertainties": "estimate_uncertainties",
}


def expand_arg_files(args: list[str]) -> list[str]:
    """Expand @files in order; later options override earlier options."""
    expanded = []
    for arg in args:
        if arg.startswith("@"):
            path = Path(arg[1:])
            if not path.is_file():
                raise FileNotFoundError(f"UQ argument file not found: {path}")
            expanded.extend(shlex.split(path.read_text(), comments=True))
        else:
            expanded.append(arg)
    return expanded


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run a UQ data or benchmark stage. Pass @file.args to load paper settings."
    )
    parser.add_argument("stage", choices=STAGES)
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return
    stage = parser.parse_args(argv[:1]).stage
    module = importlib.import_module(
        f"experiments.uncertainty_quantification.{STAGES[stage]}"
    )
    module.main(expand_arg_files(argv[1:]))


if __name__ == "__main__":
    main()
