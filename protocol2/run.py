"""Shared Protocol II command-line plumbing."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.protocol_ii_gads import (
    build_model,
    create_dataset,
    dump_json,
    infer_landmark_count,
    load_checkpoint,
    load_cluster_indices,
    load_npz_arrays,
    make_loader,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_model,
    evaluate_model,
)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, required=True, help="Path to an NPZ file with landmark and pose arrays")
    parser.add_argument("--cluster-indices", type=Path, default=None, help="JSON file with cluster index groups")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pitch-weight", type=float, default=1.0)
    parser.add_argument("--jitter-std", type=float, default=0.05)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    return parser


def train_from_args(args: argparse.Namespace) -> Path:
    # 1) Resolve runtime/config inputs.
    set_seed(args.seed)
    device = resolve_device(args.device)
    cluster_indices = load_cluster_indices(args.cluster_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "gads_best.pt"

    # 2) Load arrays and validate landmark dimensionality against cluster indices.
    raw_landmarks, raw_poses = load_npz_arrays(args.data)
    if raw_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError("The NPZ file does not contain enough landmarks for the provided cluster indices.")

    # 3) Build augmented train/validation loaders for Protocol II.
    raw_data = {"landmark": raw_landmarks, "pose": raw_poses}
    train_loader, validate_loader = create_dataset(
        raw_data=raw_data,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
        jitter_std=args.jitter_std,
        flip_prob=args.flip_prob,
    )

    # 4) Train, checkpoint, and persist metrics.
    model = build_model(device=device, cluster_indices=cluster_indices)
    history = train_model(
        model=model,
        train_loader=train_loader,
        validate_loader=validate_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pitch_weight=args.pitch_weight,
        best_model_path=best_model_path,
    )

    save_checkpoint(best_model_path, model, metadata={"device": str(device), "seed": args.seed, "cluster_indices": str(args.cluster_indices)})
    dump_json(args.output_dir / "history.json", {k: [float(v) for v in values] for k, values in history.items()})
    return best_model_path


def test_from_args(args: argparse.Namespace) -> Path:
    # Load checkpoint, evaluate once, then export predictions and aggregate metrics.
    device = resolve_device(args.device)
    cluster_indices = load_cluster_indices(args.cluster_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(device=device, cluster_indices=cluster_indices)
    metadata = load_checkpoint(args.checkpoint, model, map_location=device)
    print(f"Loaded checkpoint metadata: {metadata}")

    raw_landmarks, raw_poses = load_npz_arrays(args.data)
    if raw_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError("The NPZ file does not contain enough landmarks for the provided cluster indices.")

    raw_data = {"landmark": raw_landmarks, "pose": raw_poses}
    loader = make_loader(raw_data=raw_data, batch_size=args.batch_size, augment=False)
    metrics = evaluate_model(model, loader)
    rows = metrics.pop("rows")

    import pandas as pd

    predictions_df = pd.DataFrame(rows)
    predictions_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "metrics.json"
    predictions_df.to_csv(predictions_path, index=False)
    dump_json(summary_path, metrics)
    return predictions_path
