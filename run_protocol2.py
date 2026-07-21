"""Single entrypoint for Protocol II training and testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol2.configs import FAN_CONFIG, FACEMESH_CONFIG
from protocol2.run import test_from_args, train_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Protocol II training or testing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register detector-specific train/test commands with shared defaults.
    for name, config in (("facemesh", FACEMESH_CONFIG), ("fan", FAN_CONFIG)):
        train_parser = subparsers.add_parser(f"train-{name}", help=f"Train the Protocol II {name.upper()} model")
        train_parser.add_argument("--data", type=Path, required=True)
        train_parser.add_argument("--cluster-indices", type=Path, default=config.cluster_indices)
        train_parser.add_argument("--output-dir", type=Path, default=config.output_dir)
        train_parser.add_argument("--epochs", type=int, default=config.epochs)
        train_parser.add_argument("--batch-size", type=int, default=config.batch_size)
        train_parser.add_argument("--validation-split", type=float, default=config.validation_split)
        train_parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
        train_parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
        train_parser.add_argument("--pitch-weight", type=float, default=config.pitch_weight)
        train_parser.add_argument("--jitter-std", type=float, default=0.05)
        train_parser.add_argument("--flip-prob", type=float, default=0.5)
        train_parser.add_argument("--seed", type=int, default=7)
        train_parser.add_argument("--device", type=str, default=None)

        test_parser = subparsers.add_parser(f"test-{name}", help=f"Evaluate the Protocol II {name.upper()} model")
        test_parser.add_argument("--data", type=Path, required=True)
        test_parser.add_argument("--checkpoint", type=Path, required=True)
        test_parser.add_argument("--cluster-indices", type=Path, default=config.cluster_indices)
        test_parser.add_argument("--output-dir", type=Path, default=config.output_dir)
        test_parser.add_argument("--batch-size", type=int, default=config.batch_size)
        test_parser.add_argument("--device", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Dispatch by subcommand prefix to keep the CLI compact.
    if args.command.startswith("train-"):
        train_from_args(args)
    else:
        test_from_args(args)


if __name__ == "__main__":
    main()
