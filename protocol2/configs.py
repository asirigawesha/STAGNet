"""Protocol II detector configurations.

Each configuration defines the cluster layout and the default training
hyperparameters for one detector family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Protocol2Config:
    name: str
    cluster_indices: Path
    output_dir: Path
    batch_size: int = 128
    epochs: int = 150
    validation_split: float = 0.3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    pitch_weight: float = 1.0


ROOT = Path(__file__).resolve().parents[1]
# Protocol II runs use a single NPZ input; detector configs mainly differ by cluster indices/output paths.


FACEMESH_CONFIG = Protocol2Config(
    name="facemesh",
    cluster_indices=ROOT / "configs" / "protocol2_facemesh_cluster_indices.json",
    output_dir=ROOT / "results" / "protocol2_facemesh",
)

FAN_CONFIG = Protocol2Config(
    name="fan",
    cluster_indices=ROOT / "configs" / "protocol2_fan_cluster_indices.json",
    output_dir=ROOT / "results" / "protocol2_fan",
)
