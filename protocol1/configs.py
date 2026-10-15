"""Protocol I detector configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Protocol1Config:
    name: str
    train_data: Path
    test_data: Path
    biwi_test_data: Path
    cluster_indices: Path
    output_dir: Path
    batch_size: int = 128
    epochs: int = 130
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    pitch_weight: float = 1.0


ROOT = Path(__file__).resolve().parents[1]
# Protocol I datasets are stored under datasets/all.
DATA_ROOT = ROOT / "datasets" / "all"


FACEMESH_CONFIG = Protocol1Config(
    name="facemesh",
    train_data=DATA_ROOT / "300WLP_FaceMesh.npz",
    test_data=DATA_ROOT / "AFLW_FaceMesh.npz",
    biwi_test_data=DATA_ROOT / "BIWI_FaceMesh.npz",
    cluster_indices=ROOT / "configs" / "protocol1_facemesh_cluster_indices.json",
    output_dir=ROOT / "results" / "protocol1_facemesh",
)

FAN_CONFIG = Protocol1Config(
    name="fan",
    train_data=DATA_ROOT / "300WLP_FAN.npz",
    test_data=DATA_ROOT / "AFLW_fan.npz",
    biwi_test_data=DATA_ROOT / "BIWI_FAN.npz",
    cluster_indices=ROOT / "configs" / "protocol1_fan_cluster_indices.json",
    output_dir=ROOT / "results" / "protocol1_fan",
)
