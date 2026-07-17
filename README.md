# STAG-Net

**STAG-Net: Structure Aware TinyML Attention over Groups for Head Pose Estimation**

STAG-Net is a lightweight landmark-based head pose estimation framework designed for resource-constrained and TinyML-oriented applications. It estimates **yaw**, **pitch**, and **roll** from facial landmarks using a pipeline that combines movement-aware landmark selection, spatial grouping of landmarks, DeepSet-based group encoding, inter-group self-attention, a compact global geometry representation, and lightweight pose regression heads.

> [!NOTE]
> This repository is being released incrementally. The current release provides the prepared dataset links and the Protocol II landmark-extraction notebooks for MediaPipe FaceMesh and FAN. The complete STAG-Net training, evaluation, model, and analysis code will be released soon.

---

## Available in the Current Release

- [x] Prepared dataset access links
- [x] Protocol II MediaPipe FaceMesh landmark-extraction notebook
- [x] Protocol II FAN landmark-extraction notebook
- [ ] STAG-Net model implementation
- [ ] Training pipeline
- [ ] Evaluation pipeline
- [ ] Protocol I experiments
- [ ] Protocol II training and testing code
- [ ] Complexity and FLOPs analysis
- [ ] Trained model checkpoints
- [ ] Complete result reproduction scripts

---

## Evaluation Protocols

### Protocol I: Cross-Dataset Evaluation

The model is trained on **300W-LP** and evaluated on **AFLW2000** and **BIWI**. This protocol measures cross-dataset generalisation from synthetic or augmented training data to real-world test sets. The related training and evaluation code is **not yet available**.

### Protocol II: Subject-Independent BIWI Evaluation

BIWI is divided at the sequence level into **16 training sequences** and **8 testing sequences**, corresponding to approximately two-thirds of the sequences for training and one-third for testing. The current release provides the two landmark-extraction notebooks for this protocol. The STAG-Net model training code for Protocol II is **not yet available**.

---

## Landmark Detectors

### MediaPipe FaceMesh

This notebook extracts facial landmarks from BIWI RGB frames using MediaPipe FaceMesh and stores the resulting landmark coordinates together with the associated pose information.

📓 [`notebooks/protocol_ii/STAGNet_Protocol_II_FaceMesh_Landmark_Extraction.ipynb`](notebooks/protocol_ii/STAGNet_Protocol_II_FaceMesh_Landmark_Extraction.ipynb)

### Face Alignment Network (FAN)

This notebook uses FAN to extract facial landmarks from BIWI RGB frames and prepares the landmark data required by the STAG-Net pipeline.

📓 [`notebooks/protocol_ii/STAGNet_Protocol_II_FAN_Landmark_Extraction.ipynb`](notebooks/protocol_ii/STAGNet_Protocol_II_FAN_Landmark_Extraction.ipynb)

---

## Datasets

The paper uses three datasets: **300W-LP**, **AFLW2000**, and **BIWI**. The authors reconstructed and standardised these datasets using a consistent pose order of `[yaw, pitch, roll]`. Images are resized to **224 × 224** before landmark extraction.

Dataset archives are not committed to Git. Prepared versions are shared through Google Drive for research reproducibility and consistent preprocessing.

| Dataset | Purpose | Prepared dataset link | Original source |
|---|---|---|---|
| 300W-LP | Protocol I training | [Google Drive](ADD_300W_LP_DRIVE_LINK) | [Original source](ADD_300W_LP_SOURCE_LINK) |
| AFLW2000 | Protocol I testing | [Google Drive](ADD_AFLW2000_DRIVE_LINK) | [Original source](ADD_AFLW2000_SOURCE_LINK) |
| BIWI | Protocol I and Protocol II evaluation | [Google Drive](ADD_BIWI_DRIVE_LINK) | [Original source](ADD_BIWI_SOURCE_LINK) |

> Users are responsible for complying with the original licenses and terms of use of each dataset. The Google Drive links are provided only to support research reproducibility and consistent preprocessing.

See [`datasets/README.md`](datasets/README.md) for additional dataset instructions.

---

## Repository Structure

```text
STAGNet/
├── README.md                          # This file
├── .gitignore
├── datasets/
│   └── README.md                      # Dataset download and setup instructions
├── notebooks/
│   └── protocol_ii/
│       ├── STAGNet_Protocol_II_FaceMesh_Landmark_Extraction.ipynb
│       └── STAGNet_Protocol_II_FAN_Landmark_Extraction.ipynb
├── assets/
│   └── README.md                      # Architecture diagrams and figures (coming soon)
└── results/
    └── README.md                      # Result tables and reproducibility scripts (coming soon)
```

---

## Running the Notebooks

Both notebooks are designed to run in **Google Colab** with datasets accessed through Google Drive.

1. Open the required notebook in Google Colab.
2. Make a copy in your Google Drive if necessary.
3. Mount Google Drive when prompted.
4. Download or access the prepared BIWI dataset using the Google Drive link in the [Datasets](#datasets) section.
5. Update `DATASET_ROOT` and `OUTPUT_ROOT` at the top of the notebook to point to your Google Drive paths.
6. Install any dependencies listed in the notebook's setup cell.
7. Run the notebook cells in order.
8. Verify that the generated landmark files have been saved in the selected output directory.

> **Note:** GPU runtime may be useful when running the FAN notebook. The MediaPipe FaceMesh notebook can typically run on CPU as well.

---

## Expected Data Convention

Each generated sample should associate:

- facial landmark coordinates;
- yaw, pitch, and roll angles;
- sample or frame identifier, when available;
- detector validity information, when available.

The exact saved array names and file format follow the corresponding notebook implementation.

---

## Important Evaluation Note

A landmark-based regressor can only process frames for which the selected detector produces a valid landmark set. Detector coverage and pose-estimation MAE should therefore be considered together, particularly for extreme head poses.

---

## Roadmap

- Release the complete STAG-Net PyTorch implementation
- Release Protocol I training and evaluation code
- Release Protocol II training and evaluation code
- Release preprocessing and landmark-selection utilities
- Release movement-aware grouping and clustering code
- Release FLOPs and parameter-count analysis
- Release trained model checkpoints
- Release scripts for reproducing the paper tables and figures
- Add TinyML deployment and conversion examples

---

## Results

The complete reproducibility scripts and detailed result tables will be added in a future release. See [`results/README.md`](results/README.md) for the current status.

---

## Citation

The citation information will be updated after the final publication details become available.

```bibtex
@article{stagnet,
  title   = {STAG-Net: Structure Aware TinyML Attention over Groups for Head Pose Estimation},
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

- **Name:** ADD_CONTACT_NAME
- **Email:** ADD_CONTACT_EMAIL

---

## Acknowledgements

We gratefully acknowledge the developers and maintainers of MediaPipe FaceMesh, Face Alignment Network (FAN), and the creators and distributors of the 300W-LP, AFLW2000, and BIWI datasets.
