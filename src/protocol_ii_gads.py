"""Shared model, dataset, and training utilities for STAGNet head-pose runs.

The code here is detector-agnostic: the only detector-specific inputs are the
NPZ file containing landmarks/poses and the landmark cluster indices used by
the GADS model.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# Default cluster layout used when no detector-specific JSON is provided.
POSE_ORDER = ("yaw", "pitch", "roll")
DEFAULT_CLUSTER_INDICES = [
    [6, 10, 31, 32, 33, 34, 49, 50, 51, 52, 55, 56, 57, 61, 62, 64, 65, 66],
    [0, 1, 2, 3],
    [13, 14, 15, 16],
    [17, 18, 19, 20, 21, 27, 28, 29],
    [22, 23, 24, 25, 26, 43, 44],
]


def load_cluster_indices(path: str | Path | None = None) -> list[list[int]]:
    if path is None:
        return [list(cluster) for cluster in DEFAULT_CLUSTER_INDICES]

    cluster_path = Path(path)
    clusters = json.loads(cluster_path.read_text())
    if not isinstance(clusters, list) or not all(isinstance(cluster, list) for cluster in clusters):
        raise ValueError("Cluster indices JSON must be a list of index lists.")
    return [[int(index) for index in cluster] for cluster in clusters]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_npz_arrays(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    if "landmark" in data and "pose" in data:
        return data["landmark"], data["pose"]
    if "landmarks" in data and "poses" in data:
        return data["landmarks"], data["poses"]
    raise KeyError(
        "NPZ file must contain either ('landmark', 'pose') or ('landmarks', 'poses') arrays."
    )


class DeepSet(nn.Module):
    def __init__(
        self,
        infeatures: int,
        outfeatures: int,
        nencode_layers: int,
        ndecode_layers: int,
        hidden_units: int,
        dropout: float,
        pool_mode: str = "cat_mean_max_min",
    ):
        super().__init__()
        self.pool_mode = pool_mode

        encoder_layers: list[nn.Module] = []
        for layer_index in range(nencode_layers):
            input_dim = infeatures if layer_index == 0 else hidden_units
            encoder_layers.append(nn.Linear(input_dim, hidden_units))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        if pool_mode == "cat_mean_max":
            decoder_in_features = hidden_units * 2
        elif pool_mode == "cat_mean_max_min":
            decoder_in_features = hidden_units * 3
        else:
            decoder_in_features = hidden_units

        decoder_layers: list[nn.Module] = []
        for layer_index in range(ndecode_layers):
            decoder_in = decoder_in_features if layer_index == 0 else hidden_units
            is_last = layer_index == ndecode_layers - 1
            decoder_layers.append(nn.Linear(decoder_in, outfeatures if is_last else hidden_units))
            if not is_last:
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "mean":
            return torch.mean(x, dim=1)
        if self.pool_mode == "max":
            return torch.max(x, dim=1).values
        if self.pool_mode == "cat_mean_max":
            return torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1).values], dim=1)
        if self.pool_mode == "cat_mean_max_min":
            return torch.cat(
                [torch.mean(x, dim=1), torch.max(x, dim=1).values, torch.min(x, dim=1).values],
                dim=1,
            )
        raise ValueError(f"Unknown pool mode: {self.pool_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, num_features = x.size()
        x_flat = x.reshape(-1, num_features)
        x_encoded = self.encoder(x_flat).reshape(batch_size, num_points, -1)
        x_agg = self.aggregate(x_encoded)
        return self.decoder(x_agg)


class GADS(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        cluster_indices: Sequence[Sequence[int]],
        deepset_infeatures: int,
        deepset_outfeatures: int,
        deepset_nencode_layers: int,
        deepset_ndecode_layers: int,
        deepset_hidden_units: int,
        dropout: float,
        num_heads: int,
        multihead_outdim: int,
        final_out: int = 3,
        shared_ndecode_layers: int = 2,
        head_hidden: int = 64,
        attn_dropout: float = 0.2,
        token_drop_p: float = 0.1,
        pool_mode: str = "mean_max",
    ):
        super().__init__()

        if deepset_outfeatures != multihead_outdim:
            raise ValueError("deepset_outfeatures must match multihead_outdim.")
        if final_out != 3:
            raise ValueError("This model predicts yaw, pitch, and roll only.")

        self.cluster_indices = [list(cluster) for cluster in cluster_indices]
        self.token_drop_p = token_drop_p
        self.pool_mode = pool_mode

        self.deepsets = nn.ModuleList(
            [
                DeepSet(
                    infeatures=deepset_infeatures,
                    outfeatures=deepset_outfeatures,
                    nencode_layers=deepset_nencode_layers,
                    ndecode_layers=deepset_ndecode_layers,
                    hidden_units=deepset_hidden_units,
                    dropout=dropout,
                    pool_mode="cat_mean_max",
                )
                for _ in range(num_clusters)
            ]
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=multihead_outdim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.pre_attn_ln = nn.LayerNorm(multihead_outdim)
        self.post_attn_ln = nn.LayerNorm(multihead_outdim)
        self.attn_resid_dropout = nn.Dropout(dropout)

        geom_dim = 9
        pooled_dim = multihead_outdim if pool_mode == "mean" else 2 * multihead_outdim
        fused_dim = pooled_dim + geom_dim

        self.geom_dropout = nn.Dropout(dropout)
        self.pre_shared_dropout = nn.Dropout(dropout)

        shared_layers: list[nn.Module] = []
        current_dim = fused_dim
        for _ in range(max(shared_ndecode_layers - 1, 0)):
            next_dim = max(current_dim // 2, head_hidden)
            shared_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim
        shared_layers.append(nn.Linear(current_dim, final_out))
        self.head = nn.Sequential(*shared_layers)

    @staticmethod
    def _cov_eigvals(points: torch.Tensor) -> torch.Tensor:
        centered = points - points.mean(dim=1, keepdim=True)
        denom = max(points.shape[1] - 1, 1)
        cov = torch.matmul(centered.transpose(1, 2), centered) / denom
        eigvals = torch.linalg.eigvalsh(cov).clamp_min_(0.0)
        return eigvals

    def _token_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_drop_p <= 0:
            return tokens
        keep_mask = torch.rand(tokens.shape[:2], device=tokens.device) > self.token_drop_p
        if keep_mask.sum(dim=1).min().item() == 0:
            keep_mask[:, 0] = True
        return tokens * keep_mask.unsqueeze(-1).type_as(tokens)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        cluster_tokens = []
        for cluster, deepset in zip(self.cluster_indices, self.deepsets):
            cluster_tokens.append(deepset(x[:, cluster, :]))
        tokens = torch.stack(cluster_tokens, dim=1)
        tokens = self._token_dropout(tokens)

        attn_input = self.pre_attn_ln(tokens)
        attn_out, _ = self.multihead_attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = self.post_attn_ln(tokens + self.attn_resid_dropout(attn_out))

        if self.pool_mode == "mean":
            pooled = tokens.mean(dim=1)
        else:
            pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=1)

        mean_xyz = x.mean(dim=1)
        std_xyz = x.std(dim=1, unbiased=False)
        eigvals = self._cov_eigvals(x)
        geom = torch.cat([mean_xyz, std_xyz, eigvals], dim=1)

        return torch.cat([pooled, self.geom_dropout(geom)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pre_shared_dropout(self.forward_features(x)))


class HeadPoseDataset(Dataset):
    def __init__(
        self,
        landmarks: np.ndarray,
        poses: np.ndarray,
        augment: bool = False,
        jitter_std: float = 0.0075,
        flip_prob: float = 0.5,
        flip_pairs: Sequence[tuple[int, int]] | None = None,
    ):
        self.landmarks = torch.tensor(landmarks, dtype=torch.float32)
        self.poses = torch.tensor(poses, dtype=torch.float32)
        self.augment = augment
        self.jitter_std = jitter_std
        self.flip_prob = flip_prob
        self.flip_pairs = list(flip_pairs) if flip_pairs is not None else []

    def __len__(self) -> int:
        return self.poses.shape[0]

    def _horizontal_flip(self, landmarks: torch.Tensor, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flipped = landmarks.clone()
        if flipped.numel() == 0:
            return flipped, pose

        x_values = flipped[:, 0]
        if torch.max(x_values) <= 1.5 and torch.min(x_values) >= -0.5:
            flipped[:, 0] = 1.0 - flipped[:, 0]
        else:
            flipped[:, 0] = -flipped[:, 0]

        for left_idx, right_idx in self.flip_pairs:
            flipped[[left_idx, right_idx]] = flipped[[right_idx, left_idx]]

        flipped_pose = pose.clone()
        flipped_pose[1] = -flipped_pose[1]
        flipped_pose[2] = -flipped_pose[2]
        return flipped, flipped_pose

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        landmarks = self.landmarks[index].clone()
        pose = self.poses[index].clone()

        if self.augment:
            if self.jitter_std > 0:
                landmarks = landmarks + torch.randn_like(landmarks) * self.jitter_std
            if self.flip_prob > 0 and torch.rand(1).item() < self.flip_prob:
                landmarks, pose = self._horizontal_flip(landmarks, pose)

        return {"landmark": landmarks, "pose": pose}


def _extract_arrays(raw_data: np.lib.npyio.NpzFile | dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "landmark" in raw_data and "pose" in raw_data:
        return raw_data["landmark"], raw_data["pose"]
    if "landmarks" in raw_data and "poses" in raw_data:
        return raw_data["landmarks"], raw_data["poses"]
    raise KeyError("Expected 'landmark'/'pose' or 'landmarks'/'poses' arrays in the dataset.")


def create_dataset(
    raw_data: np.lib.npyio.NpzFile | dict[str, np.ndarray],
    batch_size: int,
    validation_split: float,
    seed: int,
    jitter_std: float,
    flip_prob: float,
    num_workers: int = 0,
    flip_pairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[DataLoader, DataLoader]:
    # Build a deterministic split; apply augmentation only to the training subset.
    landmarks, poses = _extract_arrays(raw_data)
    base_dataset = HeadPoseDataset(
        landmarks=landmarks,
        poses=poses,
        augment=False,
        jitter_std=jitter_std,
        flip_prob=flip_prob,
        flip_pairs=flip_pairs,
    )

    total_size = len(base_dataset)
    val_size = max(1, int(total_size * validation_split))
    train_size = max(1, total_size - val_size)
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_size, total_size - train_size], generator=generator)

    train_subset.dataset = HeadPoseDataset(
        landmarks=landmarks,
        poses=poses,
        augment=True,
        jitter_std=jitter_std,
        flip_prob=flip_prob,
        flip_pairs=flip_pairs,
    )
    val_subset.dataset = HeadPoseDataset(
        landmarks=landmarks,
        poses=poses,
        augment=False,
        jitter_std=0.0,
        flip_prob=0.0,
        flip_pairs=flip_pairs,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def make_loader(
    raw_data: np.lib.npyio.NpzFile | dict[str, np.ndarray],
    batch_size: int,
    augment: bool = False,
    jitter_std: float = 0.0,
    flip_prob: float = 0.0,
    num_workers: int = 0,
    flip_pairs: Sequence[tuple[int, int]] | None = None,
) -> DataLoader:
    landmarks, poses = _extract_arrays(raw_data)
    dataset = HeadPoseDataset(
        landmarks=landmarks,
        poses=poses,
        augment=augment,
        jitter_std=jitter_std,
        flip_prob=flip_prob,
        flip_pairs=flip_pairs,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def progress_cosine(step: int, total_steps: int) -> float:
    ratio = min(max(step, 0), total_steps) / max(1, total_steps)
    return 0.5 * (1 - math.cos(math.pi * ratio))


class Engine:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, total_steps: int, pitch_weight: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.global_step = 0
        self.pitch_weight = pitch_weight
        self.base_loss = nn.SmoothL1Loss()

    def weighted_pose_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        yaw_loss = self.base_loss(predictions[:, 0], targets[:, 0])
        pitch_loss = self.base_loss(predictions[:, 1], targets[:, 1])
        roll_loss = self.base_loss(predictions[:, 2], targets[:, 2])
        total_loss = yaw_loss + self.pitch_weight * pitch_loss + roll_loss
        return total_loss, {
            "yaw_loss": yaw_loss.item(),
            "pitch_loss": pitch_loss.item(),
            "roll_loss": roll_loss.item(),
        }

    @staticmethod
    def mae_per_angle(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(predictions - targets), dim=0)

    def train(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_mae = torch.zeros(3, device=next(self.model.parameters()).device)
        metric_totals = {"yaw_loss": 0.0, "pitch_loss": 0.0, "roll_loss": 0.0}
        sample_count = 0

        for batch in loader:
            landmarks = batch["landmark"].to(next(self.model.parameters()).device)
            poses = batch["pose"].to(next(self.model.parameters()).device)

            predictions = self.model(landmarks)
            loss, loss_parts = self.weighted_pose_loss(predictions, poses)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            batch_size = landmarks.size(0)
            sample_count += batch_size
            running_loss += loss.item() * batch_size
            running_mae += torch.sum(torch.abs(predictions - poses), dim=0)
            for key, value in loss_parts.items():
                metric_totals[key] += value * batch_size

            self.global_step += 1

        return {
            "loss": running_loss / sample_count,
            "mae_yaw": (running_mae[0] / sample_count).item(),
            "mae_pitch": (running_mae[1] / sample_count).item(),
            "mae_roll": (running_mae[2] / sample_count).item(),
            "yaw_loss": metric_totals["yaw_loss"] / sample_count,
            "pitch_loss": metric_totals["pitch_loss"] / sample_count,
            "roll_loss": metric_totals["roll_loss"] / sample_count,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        running_mae = torch.zeros(3, device=next(self.model.parameters()).device)
        sample_count = 0

        for batch in loader:
            landmarks = batch["landmark"].to(next(self.model.parameters()).device)
            poses = batch["pose"].to(next(self.model.parameters()).device)
            predictions = self.model(landmarks)
            loss, _ = self.weighted_pose_loss(predictions, poses)

            batch_size = landmarks.size(0)
            sample_count += batch_size
            running_loss += loss.item() * batch_size
            running_mae += torch.sum(torch.abs(predictions - poses), dim=0)

        return {
            "loss": running_loss / sample_count,
            "mae_yaw": (running_mae[0] / sample_count).item(),
            "mae_pitch": (running_mae[1] / sample_count).item(),
            "mae_roll": (running_mae[2] / sample_count).item(),
            "mae_mean": (running_mae.sum() / (3 * sample_count)).item(),
        }


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


def build_model(
    device: torch.device,
    cluster_indices: Sequence[Sequence[int]] | None = None,
    deepset_hidden_units: int = 64,
    dropout: float = 0.01,
    num_heads: int = 2,
    multihead_outdim: int = 32,
) -> GADS:
    # Assemble GADS with a configurable cluster layout and initialize linear weights.
    indices = cluster_indices if cluster_indices is not None else DEFAULT_CLUSTER_INDICES
    model = GADS(
        num_clusters=len(indices),
        cluster_indices=indices,
        deepset_infeatures=3,
        deepset_outfeatures=multihead_outdim,
        deepset_nencode_layers=1,
        deepset_ndecode_layers=1,
        deepset_hidden_units=deepset_hidden_units,
        dropout=dropout,
        num_heads=num_heads,
        multihead_outdim=multihead_outdim,
    )
    model.apply(init_weights)
    return model.to(device)


def infer_landmark_count(cluster_indices: Sequence[Sequence[int]]) -> int:
    return max(index for cluster in cluster_indices for index in cluster) + 1


def save_checkpoint(path: str | Path, model: nn.Module, metadata: dict[str, object] | None = None) -> None:
    payload = {"model_state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, map_location: str | torch.device | None = None) -> dict[str, object]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    return payload.get("metadata", {})


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> dict[str, object]:
    device = next(model.parameters()).device
    model.eval()

    total_mae = torch.zeros(3, device=device)
    total_mse = torch.zeros(3, device=device)
    rows: list[dict[str, float]] = []
    sample_count = 0

    for batch in loader:
        landmarks = batch["landmark"].to(device)
        poses = batch["pose"].to(device)
        predictions = model(landmarks)

        errors = predictions - poses
        abs_errors = errors.abs()
        total_mae += abs_errors.sum(dim=0)
        total_mse += (errors ** 2).sum(dim=0)
        sample_count += landmarks.size(0)

        for pred, gt, err in zip(predictions.cpu(), poses.cpu(), abs_errors.cpu()):
            rows.append(
                {
                    "gt_yaw": float(gt[0]),
                    "gt_pitch": float(gt[1]),
                    "gt_roll": float(gt[2]),
                    "pred_yaw": float(pred[0]),
                    "pred_pitch": float(pred[1]),
                    "pred_roll": float(pred[2]),
                    "yaw_abs_error": float(err[0]),
                    "pitch_abs_error": float(err[1]),
                    "roll_abs_error": float(err[2]),
                }
            )

    mae = total_mae / sample_count
    rmse = torch.sqrt(total_mse / sample_count)
    return {
        "mae_yaw": float(mae[0]),
        "mae_pitch": float(mae[1]),
        "mae_roll": float(mae[2]),
        "mae_mean": float(mae.mean()),
        "rmse_yaw": float(rmse[0]),
        "rmse_pitch": float(rmse[1]),
        "rmse_roll": float(rmse[2]),
        "rows": rows,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validate_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    pitch_weight: float,
    best_model_path: str | Path,
) -> dict[str, list[float]]:
    # Main optimization loop with cosine restarts and best-checkpoint tracking.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    engine = Engine(model, optimizer, total_steps=num_epochs * max(len(train_loader), 1), pitch_weight=pitch_weight)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_mae": [],
        "train_yaw_loss": [],
        "train_pitch_loss": [],
        "train_roll_loss": [],
    }
    best_valid_mae = float("inf")

    for epoch in range(num_epochs):
        train_metrics = engine.train(train_loader)
        valid_metrics = engine.evaluate(validate_loader)
        scheduler.step(epoch + progress_cosine(epoch + 1, num_epochs))

        history["train_loss"].append(train_metrics["loss"])
        history["val_mae"].append(valid_metrics["mae_mean"])
        history["train_yaw_loss"].append(train_metrics["yaw_loss"])
        history["train_pitch_loss"].append(train_metrics["pitch_loss"])
        history["train_roll_loss"].append(train_metrics["roll_loss"])

        if valid_metrics["mae_mean"] < best_valid_mae:
            best_valid_mae = valid_metrics["mae_mean"]
            save_checkpoint(best_model_path, model, metadata={"best_valid_mae": best_valid_mae, "epoch": epoch + 1})

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_mae={valid_metrics['mae_mean']:.4f}"
        )

    return history


def dump_json(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
