"""Build BIWI FAN NPZ files for STAGNet Protocol II experiments.

Pipeline: split subjects -> detect/crop faces -> extract landmarks -> normalize -> save NPZ.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from tqdm import tqdm


# ====== Config ======
BIWI_ROOT = Path("BIWI")
OUT_PREFIX = "BIWI"

SSD_PROTOTXT = Path("./face_detector/deploy.prototxt")
SSD_MODEL = Path("./face_detector/res10_300x300_ssd_iter_140000.caffemodel")

IMG_SIZE = 224
AD = 0.25
TRAIN_RATIO = 0.70
RANDOM_SEED = 42
SSD_CONF_THRESH = 0.9
ENABLE_TRACKING = True
MAX_BBOX_JUMP_PX = 80


def load_ssd_face_detector(prototxt_path: Path, model_path: Path):
    if not prototxt_path.exists():
        raise FileNotFoundError(f"SSD prototxt not found: {prototxt_path.resolve()}")
    if not model_path.exists():
        raise FileNotFoundError(f"SSD caffemodel not found: {model_path.resolve()}")
    return cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))


def ssd_detect_faces(net, bgr_img, conf_thresh=0.95):
    h, w = bgr_img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr_img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    dets = net.forward()

    out = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([x1, y1, x2, y2, conf])
    return out


def choose_best_face(detections, img_w, img_h):
    if len(detections) == 0:
        return None
    cx_t, cy_t = img_w * 0.5, img_h * 0.5
    best, best_score = None, 1e18
    for (x1, y1, x2, y2, conf) in detections:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        score = abs(cx - cx_t) + abs(cy - cy_t)
        if score < best_score:
            best_score = score
            best = (x1, y1, x2, y2, conf)
    return best


def expand_and_clip_box(x1, y1, x2, y2, img_w, img_h, ad=0.25):
    w = x2 - x1
    h = y2 - y1
    x1n = max(int(x1 - ad * w), 0)
    y1n = max(int(y1 - ad * h), 0)
    x2n = min(int(x2 + ad * w), img_w - 1)
    y2n = min(int(y2 + ad * h), img_h - 1)
    return x1n, y1n, x2n, y2n


def crop_face_with_ssd(bgr_img, net, ad=0.1, conf_thresh=0.95, prev_box=None):
    h, w = bgr_img.shape[:2]
    dets = ssd_detect_faces(net, bgr_img, conf_thresh=conf_thresh)
    best = choose_best_face(dets, w, h)

    if best is None:
        if prev_box is None:
            return None, None
        x1, y1, x2, y2 = prev_box
        return bgr_img[y1:y2, x1:x2], prev_box

    x1, y1, x2, y2, _ = best
    x1, y1, x2, y2 = expand_and_clip_box(x1, y1, x2, y2, w, h, ad=ad)
    cur_box = (x1, y1, x2, y2)

    if ENABLE_TRACKING and prev_box is not None:
        jump = abs(cur_box[0] - prev_box[0])
        if jump >= MAX_BBOX_JUMP_PX:
            x1, y1, x2, y2 = prev_box
            cur_box = prev_box

    return bgr_img[y1:y2, x1:x2], cur_box


def read_biwi_pose_txt(pose_path: Path):
    rows = []
    for line in pose_path.read_text().strip().splitlines():
        parts = [p for p in line.strip().split(" ") if p]
        if parts:
            rows.append([float(p) for p in parts])

    arr = np.array(rows, dtype=np.float32)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected pose file format: {pose_path} -> {arr.shape}")

    return arr[:3, :3]


def rotation_to_yaw_pitch_roll_degrees(R):
    rt = R.T
    roll = -np.arctan2(rt[1, 0], rt[0, 0]) * 180.0 / np.pi
    yaw = -np.arctan2(-rt[2, 0], np.sqrt(rt[2, 1] ** 2 + rt[2, 2] ** 2)) * 180.0 / np.pi
    pitch = np.arctan2(rt[2, 1], rt[2, 2]) * 180.0 / np.pi
    return float(yaw), float(pitch), float(roll)


def list_subject_dirs(biwi_root: Path):
    subs = []
    for i in range(1, 25):
        p = biwi_root / f"{i:02d}"
        if p.exists() and p.is_dir():
            subs.append(p)
    if not subs:
        raise FileNotFoundError(f"No subject folders found under {biwi_root.resolve()}")
    return subs


def list_png_and_txt(subject_dir: Path):
    pngs = sorted([p for p in subject_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    txts = sorted([p for p in subject_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if len(pngs) != len(txts):
        png_map = {p.stem: p for p in pngs}
        txt_map = {p.stem: p for p in txts}
        common = sorted(set(png_map).intersection(txt_map))
        pngs = [png_map[k] for k in common]
        txts = [txt_map[k] for k in common]
    return pngs, txts


def split_subjects(subject_dirs, train_ratio=0.7, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(subject_dirs))
    rng.shuffle(idx)
    n_train = int(round(train_ratio * len(subject_dirs)))
    train_ids = set(idx[:n_train].tolist())
    test_ids = set(idx[n_train:].tolist())
    train_subjects = [subject_dirs[i] for i in range(len(subject_dirs)) if i in train_ids]
    test_subjects = [subject_dirs[i] for i in range(len(subject_dirs)) if i in test_ids]
    return train_subjects, test_subjects


def normalize_landmarks(landmarks_array: np.ndarray, anchor_index: int) -> np.ndarray:
    centered = []
    for i in range(landmarks_array.shape[0]):
        lm = landmarks_array[i].copy()
        lm = lm - lm[anchor_index]
        lm = np.delete(lm, anchor_index, axis=0)
        centered.append(lm)

    arr = np.asarray(centered, dtype=np.float32)
    eps = 1e-8
    for i in range(arr.shape[0]):
        for j in range(3):
            denom = arr[i, :, j].max() - arr[i, :, j].min()
            arr[i, :, j] = arr[i, :, j] / (denom + eps)
    return arr


def landmark_stats(lms: np.ndarray, name="set"):
    axes = ["x", "y", "z"]
    print(f"\n===== Landmark stats: {name} =====")
    for i, ax in enumerate(axes):
        vals = lms[:, :, i].reshape(-1)
        print(
            f"{ax.upper()} | mean={np.mean(vals):.4f}, var={np.var(vals):.4f}, "
            f"range={(np.max(vals)-np.min(vals)):.4f}, [{np.min(vals):.4f}, {np.max(vals):.4f}]"
        )


def build_split(subject_list, fan_model, ssd_net, name="split"):
    # Iterate frame pairs and keep only samples with valid pose + landmarks.
    out_lms, out_poses = [], []
    prev_box = None
    stats = defaultdict(int)

    for subject_dir in subject_list:
        pngs, txts = list_png_and_txt(subject_dir)
        print(f"[{name}] subject {subject_dir.name} frames={len(pngs)}")

        for j in tqdm(range(len(pngs))):
            img_path = pngs[j]
            pose_path = txts[j]

            try:
                R = read_biwi_pose_txt(pose_path)
                yaw, pitch, roll = rotation_to_yaw_pitch_roll_degrees(R)
                label = np.array([pitch, yaw, roll], dtype=np.float32)
            except Exception:
                stats["pose_read_fail"] += 1
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                stats["image_read_fail"] += 1
                continue

            face, prev_box = crop_face_with_ssd(
                img_bgr,
                ssd_net,
                ad=AD,
                conf_thresh=SSD_CONF_THRESH,
                prev_box=prev_box,
            )
            if face is None:
                stats["face_crop_fail"] += 1
                continue

            img_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            try:
                preds = fan_model.get_landmarks(crop_resized)
            except Exception:
                stats["detector_error"] += 1
                continue

            if preds is None or len(preds) == 0:
                stats["no_landmarks"] += 1
                continue

            lms = preds[0].astype(np.float32)
            out_lms.append(lms)
            out_poses.append(label)

    out_lms = np.asarray(out_lms, dtype=np.float32)
    out_poses = np.asarray(out_poses, dtype=np.float32)
    print(f"[{name}] kept={len(out_lms)} dropped={sum(stats.values())}")
    for k, v in sorted(stats.items()):
        print(f"[{name}] {k:16s}: {v}")
    return out_lms, out_poses


def main():
    # Subject-wise split prevents frame leakage between train and test.
    subject_dirs = list_subject_dirs(BIWI_ROOT)
    train_subjects, test_subjects = split_subjects(subject_dirs, TRAIN_RATIO, RANDOM_SEED)

    print("Train subjects:", [d.name for d in train_subjects])
    print("Test subjects :", [d.name for d in test_subjects])

    ssd_net = load_ssd_face_detector(SSD_PROTOTXT, SSD_MODEL)

    if torch.backends.mps.is_available():
        fan_device = "mps"
    elif torch.cuda.is_available():
        fan_device = "cuda"
    else:
        fan_device = "cpu"

    print("FAN device:", fan_device)
    fan_model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.THREE_D,
        device=fan_device,
        flip_input=False,
    )

    train_lms, train_poses = build_split(train_subjects, fan_model, ssd_net, name="train")
    test_lms, test_poses = build_split(test_subjects, fan_model, ssd_net, name="test")

    all_lms = np.concatenate([train_lms, test_lms], axis=0)
    all_poses = np.concatenate([train_poses, test_poses], axis=0)

    # Normalize around a stable anchor and store train/test/all bundles.
    train_lms_norm = normalize_landmarks(train_lms, anchor_index=30)
    test_lms_norm = normalize_landmarks(test_lms, anchor_index=30)
    all_lms_norm = normalize_landmarks(all_lms, anchor_index=30)

    landmark_stats(all_lms_norm, name="all")
    landmark_stats(train_lms_norm, name="train")
    landmark_stats(test_lms_norm, name="test")

    out_all = Path(f"{OUT_PREFIX}_fan_all.npz")
    out_train = Path(f"{OUT_PREFIX}_fan_train.npz")
    out_test = Path(f"{OUT_PREFIX}_fan_test.npz")

    np.savez(str(out_all), landmark=all_lms_norm, pose=all_poses)
    np.savez(str(out_train), landmark=train_lms_norm, pose=train_poses)
    np.savez(str(out_test), landmark=test_lms_norm, pose=test_poses)

    print("Saved:")
    print(" -", out_all.resolve())
    print(" -", out_train.resolve())
    print(" -", out_test.resolve())


if __name__ == "__main__":
    main()
