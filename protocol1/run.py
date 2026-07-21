"""Shared Protocol I command-line plumbing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.protocol_ii_gads import (
    HeadPoseDataset,
    build_model,
    dump_json,
    evaluate_model,
    infer_landmark_count,
    load_checkpoint,
    load_cluster_indices,
    load_npz_arrays,
    save_checkpoint,
    set_seed,
    train_model,
    resolve_device,
)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-data", type=Path, default=None, help="Path to the Protocol I training NPZ")
    parser.add_argument("--test-data", type=Path, default=None, help="Path to the Protocol I test NPZ")
    parser.add_argument("--cluster-indices", type=Path, default=None, help="JSON file with cluster index groups")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pitch-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    return parser


def _make_loaders(landmarks, poses, batch_size: int, validation_split: float, seed: int) -> tuple[DataLoader, DataLoader]:
    # Protocol I keeps preprocessing fixed and only splits into train/validation subsets.
    dataset = HeadPoseDataset(landmarks=landmarks, poses=poses)
    dataset_size = len(dataset)
    validation_size = max(1, int(dataset_size * validation_split))
    train_size = max(1, dataset_size - validation_size)

    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_subset = torch.utils.data.random_split(dataset, [train_size, validation_size], generator=generator)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def train_from_args(args: argparse.Namespace) -> Path:
    # 1) Resolve runtime/config inputs.
    set_seed(args.seed)
    device = resolve_device(args.device)
    cluster_indices = load_cluster_indices(args.cluster_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "gads_best.pt"

    if args.train_data is None or args.test_data is None:
        raise ValueError("Both --train-data and --test-data must be provided or set by the wrapper.")

    # 2) Load arrays and validate landmark dimensionality against cluster indices.
    train_landmarks, train_poses = load_npz_arrays(args.train_data)
    test_landmarks, test_poses = load_npz_arrays(args.test_data)

    if train_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError("The training NPZ file does not contain enough landmarks for the provided cluster indices.")
    if test_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError("The test NPZ file does not contain enough landmarks for the provided cluster indices.")

    # 3) Train, checkpoint, and persist metrics.
    train_loader, validate_loader = _make_loaders(train_landmarks, train_poses, batch_size=args.batch_size, validation_split=args.validation_split, seed=args.seed)

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
    # Load checkpoint, run evaluation, and export per-sample predictions + summary metrics.
    device = resolve_device(args.device)
    cluster_indices = load_cluster_indices(args.cluster_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(device=device, cluster_indices=cluster_indices)
    metadata = load_checkpoint(args.checkpoint, model, map_location=device)
    print(f"Loaded checkpoint metadata: {metadata}")

    test_landmarks, test_poses = load_npz_arrays(args.test_data)
    if test_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError("The test NPZ file does not contain enough landmarks for the provided cluster indices.")

    if args.test_data is None:
        raise ValueError("--test-data must be provided or set by the wrapper.")

    test_dataset = HeadPoseDataset(landmarks=test_landmarks, poses=test_poses)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_model(model, test_loader)
    rows = metrics.pop("rows")
    predictions_df = pd.DataFrame(rows)

    predictions_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "metrics.json"
    predictions_df.to_csv(predictions_path, index=False)
    dump_json(summary_path, metrics)
    return predictions_path
