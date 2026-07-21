# STAG-Net: Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation

STAG-Net is a landmark-based head pose estimation framework designed for lightweight and research-oriented deployment. It predicts **yaw**, **pitch**, and **roll** from facial landmarks using a pipeline built around spatial landmark grouping, DeepSet-based group encoding, inter-group attention, and compact regression heads.

> [!NOTE]
> This repository now includes reusable Python code for Protocol I and Protocol II training and testing, plus BIWI landmark builder scripts for FaceMesh and FAN. 300W-LP and AFLW landmark extraction code will be added later.

---

## Available in the Current Release

- [x] Prepared dataset access links
- [x] BIWI FaceMesh builder script
- [x] BIWI FAN builder script
- [x] STAG-Net model implementation
- [x] Training pipeline
- [x] Evaluation pipeline
- [x] Protocol I training and testing code
- [x] Protocol II training and testing code
- [x] Detector-specific Protocol I and Protocol II config files
- [x] Shared detector-agnostic GADS model and data utilities


---

## Evaluation Protocols

### Protocol I: Cross-Dataset Evaluation

The model is trained on **300W-LP** and evaluated on **AFLW2000** and **BIWI**. This protocol measures cross-dataset generalisation from synthetic training data to real-world test sets. The Protocol I Python entrypoint is [`run_protocol1.py`](run_protocol1.py).

### Protocol II: Subject-Independent BIWI Evaluation

BIWI is divided at the sequence level into **16 training sequences** and **8 testing sequences**, corresponding to approximately two-thirds of the sequences for training and one-third for testing. The Protocol II Python entrypoint is [`run_protocol2.py`](run_protocol2.py).

---

## Landmark Detectors

### FaceMesh

FaceMesh is packaged in MediaPipe: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker

### Face Alignment Network (FAN)

Both detectors are supported in the training and testing pipelines, and BIWI builder scripts are provided for each detector in the datasets folder.

---

## Datasets

The paper uses three datasets: **300W-LP**, **AFLW2000**, and **BIWI**. The prepared files in this repository follow a consistent pose order of `[pitch, yaw, roll]`.

Dataset archives are not committed to Git. Prepared versions are shared through Google Drive for research reproducibility and consistent preprocessing.

Prepared dataset folder:

- [Google Drive Folder](https://drive.google.com/drive/folders/1s5tV8XYzb3rpd4x2NDV4nfngADPlXzVr?usp=sharing)

The folder includes:

- FaceMesh-extracted landmark files for **300W-LP**, **AFLW2000**, and **BIWI**.
- FAN-extracted landmark files for **300W-LP**, **AFLW2000**, and **BIWI**.
- BIWI Protocol II train/test landmark files for each detector (FaceMesh and FAN).
- RGB datasets resized to **224 × 224**.

| Dataset | Purpose | Prepared dataset link | Original source |
|---|---|---|---|
| 300W-LP | Protocol I training | [Google Drive Folder](https://drive.google.com/drive/folders/1s5tV8XYzb3rpd4x2NDV4nfngADPlXzVr?usp=sharing) | [Original source](https://www.tensorflow.org/datasets/catalog/the300w_lp) |
| AFLW2000 | Protocol I testing | [Google Drive Folder](https://drive.google.com/drive/folders/1s5tV8XYzb3rpd4x2NDV4nfngADPlXzVr?usp=sharing) | [Original source](https://www.tensorflow.org/datasets/catalog/aflw2k3d) |
| BIWI | Protocol I and Protocol II evaluation | [Google Drive Folder](https://drive.google.com/drive/folders/1s5tV8XYzb3rpd4x2NDV4nfngADPlXzVr?usp=sharing) | [Original source](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose) |

> Users are responsible for complying with the original licenses and terms of use of each dataset. The Google Drive links are provided only to support research reproducibility and consistent preprocessing.

See [`datasets/README.md`](datasets/README.md) for additional dataset instructions.

---

## Repository Structure

```text
STAGNet/
├── README.md                          # This file
├── .gitignore
├── run_protocol1.py                   # Protocol I CLI with train/test subcommands
├── run_protocol2.py                   # Protocol II CLI with train/test subcommands
├── protocol1/                         # Protocol I shared helpers and configs
├── protocol2/                         # Protocol II shared helpers and configs
├── configs/
│   ├── protocol1_facemesh_cluster_indices.json
│   ├── protocol1_fan_cluster_indices.json
│   ├── protocol2_facemesh_cluster_indices.json
│   └── protocol2_fan_cluster_indices.json
├── src/
│   └── protocol_ii_gads.py            # Shared detector-agnostic model/data code
└── datasets/
  ├── README.md
  ├── build_biwi_facemesh.py         # BIWI FaceMesh landmark builder
  └── build_biwi_fan.py              # BIWI FAN landmark builder
```

### What the Repository Contains

- `run_protocol1.py` and `run_protocol2.py`: the main command-line entrypoints, each with train and test subcommands.
- `protocol1/` and `protocol2/`: protocol-specific defaults, data paths, and shared orchestration logic.
- `src/protocol_ii_gads.py`: the shared GADS model, dataset wrappers, training loop, evaluation loop, checkpoint utilities, and detector-agnostic helpers.
- `configs/`: cluster-index JSON files for FaceMesh and FAN, for both Protocol I and Protocol II.
- `datasets/build_biwi_facemesh.py`: BIWI FaceMesh builder (split, detect, normalize, save).
- `datasets/build_biwi_fan.py`: BIWI FAN builder (split, detect, normalize, save).


---

## Python Entry Points

Protocol I uses one CLI with subcommands:

```bash
python run_protocol1.py train-facemesh
python run_protocol1.py test-facemesh --test-data <path> --checkpoint <path-to-model.pt>
python run_protocol1.py train-fan
python run_protocol1.py test-fan --test-data <path> --checkpoint <path-to-model.pt>
```

Protocol II uses one CLI with subcommands:

```bash
python run_protocol2.py train-facemesh --data <path-to-npz>
python run_protocol2.py test-facemesh --data <path-to-npz> --checkpoint <path-to-model.pt>
python run_protocol2.py train-fan --data <path-to-npz>
python run_protocol2.py test-fan --data <path-to-npz> --checkpoint <path-to-model.pt>
```

For Protocol I, the defaults point at the dataset files used in this repository release. For Protocol II, pass the NPZ file you want to train or evaluate on together with the matching cluster-index JSON file.

---

## BIWI Builder Scripts

BIWI landmark extraction is now provided as Python scripts:

- `datasets/build_biwi_facemesh.py`
- `datasets/build_biwi_fan.py`

Each script includes:

- subject-wise train/test split
- detector loop
- landmark normalization
- range analysis
- NPZ export

The training and evaluation code for both protocols lives in [run_protocol1.py](run_protocol1.py) and [run_protocol2.py](run_protocol2.py).

## Expected Data Convention

Each generated sample should associate:

- facial landmark coordinates;
- pitch, yaw, and roll angles;
- sample or frame identifier, when available;
- detector validity information, when available.

The exact saved array names and file format follow the corresponding builder script implementation.

---

## Important Evaluation Note

A landmark-based regressor can only process frames for which the selected detector produces a valid landmark set. Detector coverage and pose-estimation MAE should therefore be considered together, particularly for extreme head poses.

---


## Citation

The citation information will be updated after the final publication details become available.

```bibtex
@article{stagnet,
  title   = {STAG-Net: Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```

---

## License

The repository license will be added with the complete code release. The datasets remain subject to their respective original licenses and terms of use.

---

## Contact

For questions about the repository or the STAG-Net implementation, please open a GitHub issue or contact:

- **Name:** Asiri Gawesha Lindamulage
- **Email:** asirigawesha@gmail.com, asiri.l@sliit.lk

---

## Acknowledgements

We gratefully acknowledge the developers and maintainers of FaceMesh, Face Alignment Network (FAN), and the creators and distributors of the 300W-LP, AFLW2000, and BIWI datasets.
