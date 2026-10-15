"""Shared Protocol I command-line plumbing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.stagnet import (
    HeadPoseDataset,
    build_model,
    dump_json,
    evaluate_model,
    infer_landmark_count,
    load_checkpoint,
    load_cluster_indices,
    load_npz_arrays,
    set_seed,
    train_model,
    resolve_device,
)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-data", type=Path, default=None, help="Path to the Protocol I training NPZ")
    parser.add_argument("--test-data", type=Path, default=None, help="Path to the AFLW validation/test NPZ")
    parser.add_argument("--biwi-test-data", type=Path, default=None, help="Path to the full BIWI validation/test NPZ")
    parser.add_argument(
        "--evaluation-dataset",
        choices=("aflw", "biwi"),
        default="aflw",
        help="Dataset used for validation and best-checkpoint selection",
    )
    parser.add_argument("--cluster-indices", type=Path, default=None, help="JSON file with cluster index groups")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pitch-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N test samples (default: all)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    return parser


def _make_loaders(
    train_landmarks,
    train_poses,
    validation_landmarks,
    validation_poses,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = HeadPoseDataset(landmarks=train_landmarks, poses=train_poses)
    validation_dataset = HeadPoseDataset(landmarks=validation_landmarks, poses=validation_poses)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def train_from_args(args: argparse.Namespace) -> Path:
    # 1) Resolve runtime/config inputs.
    set_seed(args.seed)
    device = resolve_device(args.device)
    cluster_indices = load_cluster_indices(args.cluster_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "stagnet_best.pt"

    if args.train_data is None:
        raise ValueError("--train-data must be provided or set by the wrapper.")

    evaluation_paths = {
        "aflw": args.test_data,
        "biwi": args.biwi_test_data,
    }
    validation_data_path = evaluation_paths[args.evaluation_dataset]
    if validation_data_path is None:
        option = "--test-data" if args.evaluation_dataset == "aflw" else "--biwi-test-data"
        raise ValueError(f"{option} must be provided or set by the wrapper.")

    # 2) Load arrays and validate landmark dimensionality against cluster indices.
    train_landmarks, train_poses = load_npz_arrays(args.train_data)
    validation_landmarks, validation_poses = load_npz_arrays(validation_data_path)

    if train_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError("The training NPZ file does not contain enough landmarks for the provided cluster indices.")
    if validation_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError(
            f"The {args.evaluation_dataset.upper()} NPZ file does not contain enough landmarks "
            "for the provided cluster indices."
        )

    # 3) Train on all 300W-LP samples; validate on the selected full evaluation dataset.
    train_loader, validate_loader = _make_loaders(
        train_landmarks,
        train_poses,
        validation_landmarks,
        validation_poses,
        batch_size=args.batch_size,
    )

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
        checkpoint_metadata={
            "model": "STAGNET",
            "protocol": "I",
            "device": str(device),
            "seed": args.seed,
            "validation_dataset": args.evaluation_dataset,
            "cluster_indices": cluster_indices,
        },
    )

    if not best_model_path.is_file() or best_model_path.stat().st_size == 0:
        raise RuntimeError(f"Training did not produce a checkpoint at {best_model_path}.")
    metadata = load_checkpoint(best_model_path, model, map_location=device)
    dump_json(args.output_dir / "history.json", {k: [float(v) for v in values] for k, values in history.items()})
    print(f"Saved and reloaded best STAGNET checkpoint: {best_model_path} (metadata: {metadata})")

    final_metrics = evaluate_model(model, validate_loader)
    final_rows = final_metrics.pop("rows")
    final_metrics["num_samples"] = len(validation_poses)
    final_metrics["evaluation_dataset"] = args.evaluation_dataset
    dump_json(args.output_dir / f"final_{args.evaluation_dataset}_metrics.json", final_metrics)
    pd.DataFrame(final_rows).to_csv(
        args.output_dir / f"final_{args.evaluation_dataset}_predictions.csv",
        index=False,
    )
    print(
        f"Final {args.evaluation_dataset.upper()} MAE (same dataset used for checkpoint selection): "
        f"pitch={final_metrics['mae_pitch']:.3f}, "
        f"yaw={final_metrics['mae_yaw']:.3f}, "
        f"roll={final_metrics['mae_roll']:.3f}, "
        f"mean={final_metrics['mae_mean']:.3f}"
    )
    return best_model_path


def test_from_args(args: argparse.Namespace) -> Path:
    # Evaluate only the selected dataset.
    device = resolve_device(args.device)
    cluster_indices = load_cluster_indices(args.cluster_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(device=device, cluster_indices=cluster_indices)
    metadata = load_checkpoint(args.checkpoint, model, map_location=device)
    print(f"Loaded checkpoint metadata: {metadata}")

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")

    evaluation_paths = {
        "aflw": (args.test_data, "", "AFLW"),
        "biwi": (args.biwi_test_data, "biwi_", "BIWI"),
    }
    data_path, output_prefix, dataset_name = evaluation_paths[args.evaluation_dataset]
    option = "--test-data" if args.evaluation_dataset == "aflw" else "--biwi-test-data"
    if data_path is None:
        raise ValueError(f"{option} must be provided or set by the wrapper.")

    test_landmarks, test_poses = load_npz_arrays(data_path)
    if test_landmarks.shape[1] < infer_landmark_count(cluster_indices):
        raise ValueError(
            f"The {dataset_name} NPZ file does not contain enough landmarks "
            "for the provided cluster indices."
        )

    test_dataset = HeadPoseDataset(landmarks=test_landmarks, poses=test_poses)
    if args.max_samples is not None:
        test_dataset = Subset(test_dataset, range(min(args.max_samples, len(test_dataset))))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    evaluation_started = time.perf_counter()
    metrics = evaluate_model(model, test_loader)
    metrics["num_samples"] = len(test_dataset)
    metrics["evaluation_seconds"] = time.perf_counter() - evaluation_started
    rows = metrics.pop("rows")

    predictions_file = args.output_dir / f"{output_prefix}predictions.csv"
    summary_file = args.output_dir / f"{output_prefix}metrics.json"
    pd.DataFrame(rows).to_csv(predictions_file, index=False)
    dump_json(summary_file, metrics)
    print(
        f"{dataset_name} MAE (degrees): "
        f"pitch={metrics['mae_pitch']:.3f}, "
        f"yaw={metrics['mae_yaw']:.3f}, "
        f"roll={metrics['mae_roll']:.3f}, "
        f"mean={metrics['mae_mean']:.3f}"
    )
    print(
        f"Evaluated {metrics['num_samples']} samples in "
        f"{metrics['evaluation_seconds']:.3f} seconds "
        f"({device}; inference and metric calculation only)."
    )
    return predictions_file
