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
- [x] Shared detector-agnostic STAGNET model and data utilities


---

## Evaluation Protocols

### Protocol I: Cross-Dataset Evaluation

The model is trained on all **300W-LP** samples. During training, choose **AFLW2000** or the full **BIWI** set as the validation dataset for best-checkpoint selection with `--evaluation-dataset aflw|biwi`. The selected dataset is also evaluated at the end of training and its metrics/predictions are saved. Since the full selected set is used for checkpoint selection, that final MAE is not an independent test estimate. The Protocol I Python entrypoint is [`run_protocol1.py`](run_protocol1.py).

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

The paper uses three datasets: **300W-LP**, **AFLW2000**, and **BIWI**. Pose arrays in prepared datasets use the order `[pitch, yaw, roll]`.

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

### Changing Dataset Locations

The Protocol I defaults expect these prepared NPZ files under `datasets/all/`:

| Detector | Training (300W-LP) | AFLW validation/test | Full BIWI validation/test |
|---|---|---|---|
| FaceMesh | `300WLP_FaceMesh.npz` | `AFLW_FaceMesh.npz` | `BIWI_FaceMesh.npz` |
| FAN | `300WLP_FAN.npz` | `AFLW_fan.npz` | `BIWI_FAN.npz` |

If you store these files in another directory, either pass their locations on the command line or change `DATA_ROOT` and the corresponding filenames in [`protocol1/configs.py`](protocol1/configs.py). For Protocol I training, `--train-data` sets the 300W-LP file, `--test-data` sets AFLW, and `--biwi-test-data` sets BIWI. Choose the validation/checkpoint-selection dataset with `--evaluation-dataset aflw` or `--evaluation-dataset biwi`. For Protocol I testing, select one dataset per run with `--evaluation-dataset aflw` (default) or `--evaluation-dataset biwi`; `--test-data` sets the AFLW file and `--biwi-test-data` sets the BIWI file.

Protocol II accepts its dataset path directly through `--data` for either training or testing, so you do not need to change its config to point at a different NPZ. For example, use `--data /path/to/BIWI_FAN.npz`.

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
│   └── stagnet.py                     # STAGNET model and shared training/evaluation utilities
└── datasets/
  ├── README.md
  ├── build_biwi_facemesh.py         # BIWI FaceMesh landmark builder
  └── build_biwi_fan.py              # BIWI FAN landmark builder
```

### What the Repository Contains

- `run_protocol1.py` and `run_protocol2.py`: the main command-line entrypoints, each with train and test subcommands.
- `protocol1/` and `protocol2/`: protocol-specific defaults, data paths, and shared orchestration logic.
- `src/stagnet.py`: the STAGNET model, dataset wrappers, training loop, evaluation loop, checkpoint utilities, and detector-agnostic helpers.
- `configs/`: cluster-index JSON files for FaceMesh and FAN, for both Protocol I and Protocol II.
- `datasets/build_biwi_facemesh.py`: BIWI FaceMesh builder (split, detect, normalize, save).
- `datasets/build_biwi_fan.py`: BIWI FAN builder (split, detect, normalize, save).


---

## Python Entry Points

Protocol I uses one CLI with subcommands:

```bash
python run_protocol1.py train-facemesh
python run_protocol1.py train-fan --evaluation-dataset biwi
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

For Protocol I, the defaults point at the dataset files used in this repository release. Set `--evaluation-dataset aflw` (the default) or `--evaluation-dataset biwi` when training to select which full dataset validates and selects the best checkpoint. For Protocol II, pass the NPZ file you want to train or evaluate on together with the matching cluster-index JSON file.

### Quick FaceMesh AFLW Demo for Reviewers

This demo runs evaluation only; it does not train a model or extract landmarks. Download the prepared FaceMesh `AFLW_FaceMesh.npz` dataset and obtain a FaceMesh checkpoint already trained on 300W-LP.

Checkpoints produced by Protocol I training are saved as `results/protocol1_facemesh/stagnet_best.pt` for FaceMesh and `results/protocol1_fan/stagnet_best.pt` for FAN. These model files are not included in the repository. If you download a pretrained FaceMesh checkpoint, place it in the repository's ignored `models/` directory, for example:

```text
STAGNet/
└── models/
    └── protocol1_facemesh/
        └── stagnet_best.pt
```

Create the directory if needed (`mkdir -p models/protocol1_facemesh`). You can instead keep the checkpoint anywhere accessible; pass its actual location to `--checkpoint`. From the repository root, run:

```bash
python3 run_protocol1.py test-facemesh \
  --test-data datasets/all/AFLW_FaceMesh.npz \
  --checkpoint models/protocol1_facemesh/stagnet_best.pt \
  --cluster-indices configs/protocol1_facemesh_cluster_indices.json \
  --output-dir results/reviewer_demo_facemesh \
  --batch-size 256 \
  --max-samples 200
```

The command evaluates 200 AFLW samples; Protocol I testing evaluates only the selected dataset. This is a smoke demo, not a full AFLW2000 benchmark result. It prints pitch/yaw/roll and mean MAE, and writes `predictions.csv` and `metrics.json` under `results/reviewer_demo_facemesh/`.

For a full AFLW evaluation, remove `--max-samples 200`. To evaluate BIWI instead, run the command with `--evaluation-dataset biwi`; it reads the configured BIWI dataset and writes `biwi_predictions.csv` and `biwi_metrics.json`. The checkpoint must use the FaceMesh detector and matching cluster-index configuration. This repository does not include pretrained checkpoint files or dataset archives; obtain both separately. Runtime depends on hardware and excludes Python startup, file loading, and writing results.

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
- pitch, yaw, and roll angles, in that order;
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
