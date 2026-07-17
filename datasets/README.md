# Datasets

> [!WARNING]
> Dataset files are **not** tracked by Git. Do not commit raw or processed dataset archives to this repository.

---

## Download Links

The prepared datasets are shared through Google Drive to support research reproducibility and consistent preprocessing. The pose labels follow the order **[yaw, pitch, roll]** and images are resized to **224 × 224** before landmark extraction.

| Dataset | Purpose | Prepared dataset link | Original source |
|---|---|---|---|
| 300W-LP | Protocol I training | [Google Drive](ADD_300W_LP_DRIVE_LINK) | [Original source](ADD_300W_LP_SOURCE_LINK) |
| AFLW2000 | Protocol I testing | [Google Drive](ADD_AFLW2000_DRIVE_LINK) | [Original source](ADD_AFLW2000_SOURCE_LINK) |
| BIWI | Protocol I and Protocol II evaluation | [Google Drive](ADD_BIWI_DRIVE_LINK) | [Original source](ADD_BIWI_SOURCE_LINK) |

> Users are responsible for complying with the original licenses and terms of use of each dataset. The Google Drive links are provided only to support research reproducibility and consistent preprocessing.

---

## Dataset Descriptions

| Dataset | Description |
|---|---|
| **300W-LP** | A large-pose face dataset synthesised from 300W using 3D face morphable models. Used for Protocol I training. |
| **AFLW2000** | Real-world annotated face images with a wide range of pose angles. Used for Protocol I testing. |
| **BIWI** | RGB-D video sequences of subjects with head-pose annotations. Used for both Protocol I evaluation and Protocol II subject-independent evaluation. |

---

## Evaluation Protocols

### Protocol I

The model is trained on **300W-LP** and evaluated on **AFLW2000** and **BIWI**.

### Protocol II

**BIWI** is divided at the sequence level into **16 training sequences** and **8 testing sequences** (approximately two-thirds training, one-third testing). This is a subject-independent split.

---

## Setting Up Datasets in Google Drive

Download the prepared datasets from the Google Drive links above and place them in your Google Drive following this suggested structure:

```text
MyDrive/
└── STAGNet_Datasets/
    ├── 300W_LP/
    ├── AFLW2000/
    └── BIWI/
        ├── train/
        └── test/
```

The exact BIWI folder structure must match the paths expected by the notebooks. Update `DATASET_ROOT` and `OUTPUT_ROOT` in each notebook before running.

---

## Landmark Extraction Image Size

All images are resized to **224 × 224** pixels before landmark extraction is performed.

---

## Pose Label Order

All prepared datasets use the consistent label order:

```
[yaw, pitch, roll]
```
