from __future__ import annotations

import importlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any


def ensure_package(package_name: str, import_name: str | None = None) -> None:
    module_name = import_name or package_name
    try:
        importlib.import_module(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


for package_name, import_name in [
    ("gradio", "gradio"),
    ("ultralytics", "ultralytics"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
]:
    ensure_package(package_name, import_name)

import cv2
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "patient_safety_monitoring_yolov8n.ipynb"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "gradio_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_SAMPLE_COUNT = 10
INFERENCE_IMG_SIZE = 640
MIN_PERSON_CONF = 0.20
MAX_DETECTIONS = 5

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

FrameShape = Sequence[int]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model_key: str
    display_name: str
    privacy_mode: str
    privacy_score: int
    decision_layer_path: Path
    decision_config_path: Path


MODEL_WEIGHTS = {
    "yolov8n": PROJECT_ROOT / "yolov8n.pt",
    "yolov8n_pose": PROJECT_ROOT / "yolov8n-pose.pt",
}

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "yolov8n_rgb": ExperimentConfig(
        experiment_name="yolov8n_rgb",
        model_key="yolov8n",
        display_name="YOLOv8n RGB",
        privacy_mode="full_rgb",
        privacy_score=1,
        decision_layer_path=MODELS_DIR / "yolov8n" / "yolov8n_rgb" / "decision_layer.joblib",
        decision_config_path=MODELS_DIR / "yolov8n" / "yolov8n_rgb" / "decision_layer_config.json",
    ),
    "yolov8n_blur": ExperimentConfig(
        experiment_name="yolov8n_blur",
        model_key="yolov8n",
        display_name="YOLOv8n Blurred",
        privacy_mode="blurred_rgb",
        privacy_score=3,
        decision_layer_path=MODELS_DIR / "yolov8n" / "yolov8n_blur" / "decision_layer.joblib",
        decision_config_path=MODELS_DIR / "yolov8n" / "yolov8n_blur" / "decision_layer_config.json",
    ),
    "yolov8n_pose_rgb": ExperimentConfig(
        experiment_name="yolov8n_pose_rgb",
        model_key="yolov8n_pose",
        display_name="YOLOv8n-Pose Skeleton Output",
        privacy_mode="full_rgb",
        privacy_score=4,
        decision_layer_path=MODELS_DIR / "yolov8n_pose" / "yolov8n_pose_rgb" / "decision_layer.joblib",
        decision_config_path=MODELS_DIR / "yolov8n_pose" / "yolov8n_pose_rgb" / "decision_layer_config.json",
    ),
}

EXPERIMENT_LABELS = {key: cfg.display_name for key, cfg in EXPERIMENTS.items()}


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


for path in [
    NOTEBOOK_PATH,
    REPORTS_DIR / "tables" / "experiment_split_metrics.csv",
    REPORTS_DIR / "tables" / "latency_results.csv",
]:
    require_path(path)

for config in EXPERIMENTS.values():
    require_path(config.decision_layer_path)
    require_path(config.decision_config_path)


def apply_privacy_transform_to_input(frame_bgr: np.ndarray, privacy_mode: str) -> np.ndarray:
    if privacy_mode == "full_rgb":
        return frame_bgr.copy()
    if privacy_mode == "blurred_rgb":
        return cv2.GaussianBlur(frame_bgr, (31, 31), 0)
    if privacy_mode == "pixelated_rgb":
        height, width = frame_bgr.shape[:2]
        small = cv2.resize(
            frame_bgr,
            (max(width // 12, 1), max(height // 12, 1)),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return frame_bgr.copy()


def get_primary_person_detection(result: Any) -> dict[str, Any] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    person_indices = np.where(cls == 0)[0]
    if len(person_indices) == 0:
        return None
    candidate_xyxy = xyxy[person_indices]
    candidate_areas = (candidate_xyxy[:, 2] - candidate_xyxy[:, 0]) * (candidate_xyxy[:, 3] - candidate_xyxy[:, 1])
    best_local_index = int(np.argmax(candidate_areas))
    best_index = int(person_indices[best_local_index])
    return {
        "det_index": best_index,
        "xyxy": xyxy[best_index],
        "conf": float(conf[best_index]),
        "cls": int(cls[best_index]),
    }


def safe_mean_point(keypoints: np.ndarray, indices: list[int], conf_threshold: float = 0.30) -> np.ndarray | None:
    valid_points = [keypoints[index, :2] for index in indices if keypoints[index, 2] >= conf_threshold]
    if not valid_points:
        return None
    return np.mean(valid_points, axis=0)


def extract_detection_frame_features(result: Any, frame_shape: FrameShape) -> dict[str, Any]:
    frame_height, frame_width = frame_shape[:2]
    detection = get_primary_person_detection(result)
    features = {
        "detected": 0,
        "detection_conf": 0.0,
        "bbox_width_norm": np.nan,
        "bbox_height_norm": np.nan,
        "bbox_area_norm": np.nan,
        "bbox_aspect_ratio": np.nan,
        "center_y_norm": np.nan,
        "bottom_y_norm": np.nan,
    }
    if detection is None:
        return features
    x1, y1, x2, y2 = detection["xyxy"]
    width = max(float(x2 - x1), 1e-6)
    height = max(float(y2 - y1), 1e-6)
    area = width * height
    center_y = (y1 + y2) / 2.0
    features.update(
        {
            "detected": 1,
            "detection_conf": float(detection["conf"]),
            "bbox_width_norm": width / frame_width,
            "bbox_height_norm": height / frame_height,
            "bbox_area_norm": area / (frame_width * frame_height),
            "bbox_aspect_ratio": width / height,
            "center_y_norm": center_y / frame_height,
            "bottom_y_norm": y2 / frame_height,
        }
    )
    return features


def extract_pose_frame_features(result: Any, frame_shape: FrameShape) -> dict[str, Any]:
    features = extract_detection_frame_features(result, frame_shape)
    features.update(
        {
            "keypoint_count_visible": np.nan,
            "horizontal_body_ratio": np.nan,
            "torso_span_norm": np.nan,
            "head_to_hip_span_norm": np.nan,
            "shoulder_width_norm": np.nan,
        }
    )
    detection = get_primary_person_detection(result)
    keypoints = getattr(result, "keypoints", None)
    if detection is None or keypoints is None or keypoints.data is None:
        return features
    det_index = detection["det_index"]
    if det_index >= len(keypoints.data):
        return features
    keypoint_array = keypoints.data[det_index].cpu().numpy()
    visible_mask = keypoint_array[:, 2] >= 0.30
    visible_points = keypoint_array[visible_mask, :2]
    if len(visible_points) == 0:
        features["keypoint_count_visible"] = 0
        return features
    body_width = float(visible_points[:, 0].max() - visible_points[:, 0].min())
    body_height = float(visible_points[:, 1].max() - visible_points[:, 1].min())
    shoulder_center = safe_mean_point(keypoint_array, [5, 6])
    hip_center = safe_mean_point(keypoint_array, [11, 12])
    head_center = safe_mean_point(keypoint_array, [0, 1, 2, 3, 4])
    features["keypoint_count_visible"] = int(visible_mask.sum())
    features["horizontal_body_ratio"] = body_width / max(body_height, 1e-6)
    if shoulder_center is not None and hip_center is not None and not pd.isna(features.get("bbox_height_norm")):
        bbox_height_pixels = max(float(features["bbox_height_norm"]) * frame_shape[0], 1e-6)
        features["torso_span_norm"] = abs(float(hip_center[1] - shoulder_center[1])) / bbox_height_pixels
    if head_center is not None and hip_center is not None and not pd.isna(features.get("bbox_height_norm")):
        bbox_height_pixels = max(float(features["bbox_height_norm"]) * frame_shape[0], 1e-6)
        features["head_to_hip_span_norm"] = abs(float(hip_center[1] - head_center[1])) / bbox_height_pixels
    if (
        keypoint_array[5, 2] >= 0.30
        and keypoint_array[6, 2] >= 0.30
        and not pd.isna(features.get("bbox_width_norm"))
    ):
        bbox_width_pixels = max(float(features["bbox_width_norm"]) * frame_shape[1], 1e-6)
        features["shoulder_width_norm"] = abs(float(keypoint_array[5, 0] - keypoint_array[6, 0])) / bbox_width_pixels
    return features


def aggregate_numeric_features(frame_df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for column in columns:
        if column not in frame_df.columns:
            aggregated[f"{column}_mean"] = np.nan
            aggregated[f"{column}_max"] = np.nan
            aggregated[f"{column}_min"] = np.nan
            aggregated[f"{column}_std"] = np.nan
            continue
        series = pd.to_numeric(frame_df[column], errors="coerce")
        if series.notna().any():
            aggregated[f"{column}_mean"] = float(series.mean())
            aggregated[f"{column}_max"] = float(series.max())
            aggregated[f"{column}_min"] = float(series.min())
            aggregated[f"{column}_std"] = float(series.std(ddof=0)) if len(series.dropna()) > 1 else 0.0
        else:
            aggregated[f"{column}_mean"] = np.nan
            aggregated[f"{column}_max"] = np.nan
            aggregated[f"{column}_min"] = np.nan
            aggregated[f"{column}_std"] = np.nan
    return aggregated


def aggregate_clip_features(
    frame_df: pd.DataFrame,
    clip_row: dict[str, Any],
    experiment_cfg: ExperimentConfig,
) -> dict[str, Any]:
    numeric_columns = [
        "detected",
        "detection_conf",
        "bbox_width_norm",
        "bbox_height_norm",
        "bbox_area_norm",
        "bbox_aspect_ratio",
        "center_y_norm",
        "bottom_y_norm",
        "keypoint_count_visible",
        "horizontal_body_ratio",
        "torso_span_norm",
        "head_to_hip_span_norm",
        "shoulder_width_norm",
    ]
    detected_series = pd.to_numeric(frame_df.get("detected", pd.Series(dtype=float)), errors="coerce").fillna(0)
    aspect_ratio_series = pd.to_numeric(frame_df.get("bbox_aspect_ratio", pd.Series(dtype=float)), errors="coerce")
    torso_series = pd.to_numeric(frame_df.get("torso_span_norm", pd.Series(dtype=float)), errors="coerce")
    horizontal_series = pd.to_numeric(frame_df.get("horizontal_body_ratio", pd.Series(dtype=float)), errors="coerce")
    bottom_series = pd.to_numeric(frame_df.get("bottom_y_norm", pd.Series(dtype=float)), errors="coerce")

    aggregated = {
        "clip_id": clip_row["clip_id"],
        "subject": clip_row["subject"],
        "split": clip_row["split"],
        "binary_label": clip_row["binary_label"],
        "target": 0,
        "video_path": clip_row["video_path"],
        "experiment_name": experiment_cfg.experiment_name,
        "display_name": experiment_cfg.display_name,
        "model_key": experiment_cfg.model_key,
        "privacy_mode": experiment_cfg.privacy_mode,
        "privacy_score": experiment_cfg.privacy_score,
        "frames_sampled": int(len(frame_df)),
        "detected_ratio": float(detected_series.mean()) if len(frame_df) else 0.0,
        "wide_bbox_ratio": float((aspect_ratio_series.fillna(-1) > 0.90).mean()) if len(frame_df) else 0.0,
        "low_torso_ratio": float((torso_series.fillna(999) < 0.35).mean()) if len(frame_df) else 0.0,
        "horizontal_pose_ratio": float((horizontal_series.fillna(-1) > 1.00).mean()) if len(frame_df) else 0.0,
        "near_floor_ratio": float((bottom_series.fillna(-1) > 0.80).mean()) if len(frame_df) else 0.0,
    }
    aggregated.update(aggregate_numeric_features(frame_df, numeric_columns))
    return aggregated


def render_masked_person_frame(frame_bgr: np.ndarray, result: Any) -> np.ndarray:
    detection = get_primary_person_detection(result)
    canvas = np.zeros_like(frame_bgr)
    if detection is None:
        return canvas
    x1, y1, x2, y2 = detection["xyxy"].astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_bgr.shape[1]), min(y2, frame_bgr.shape[0])
    canvas[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
    return canvas


def render_skeleton_canvas(frame_shape: FrameShape, result: Any, conf_threshold: float = 0.30) -> np.ndarray:
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    detection = get_primary_person_detection(result)
    keypoints = getattr(result, "keypoints", None)
    if detection is None or keypoints is None or keypoints.data is None:
        return canvas
    det_index = detection["det_index"]
    if det_index >= len(keypoints.data):
        return canvas
    keypoint_array = keypoints.data[det_index].cpu().numpy()
    for start_index, end_index in SKELETON_EDGES:
        if keypoint_array[start_index, 2] >= conf_threshold and keypoint_array[end_index, 2] >= conf_threshold:
            start_point = tuple(keypoint_array[start_index, :2].astype(int))
            end_point = tuple(keypoint_array[end_index, :2].astype(int))
            cv2.line(canvas, start_point, end_point, (0, 255, 255), 3)
    for keypoint in keypoint_array:
        if keypoint[2] >= conf_threshold:
            cv2.circle(canvas, tuple(keypoint[:2].astype(int)), 4, (0, 255, 0), -1)
    return canvas


def overlay_prediction_banner(frame_bgr: np.ndarray, label: str, probability: float, threshold: float, frame_note: str) -> np.ndarray:
    annotated = frame_bgr.copy()
    color = (0, 0, 255) if label == "Fall" else (0, 180, 0)
    cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 95), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 95), color, 2)
    cv2.putText(annotated, f"Prediction: {label}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"Fall probability: {probability:.3f} | threshold: {threshold:.3f}",
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(annotated, frame_note, (24, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return annotated


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe or "uploaded_video"


def choose_sample_indices(video_path: str | Path, sample_count: int) -> tuple[list[int], float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if frame_count <= 0:
        raise RuntimeError("The uploaded video does not contain readable frames.")
    actual_sample_count = min(max(sample_count, 1), frame_count)
    frame_indices = np.unique(np.linspace(0, frame_count - 1, num=actual_sample_count, dtype=int)).tolist()
    return frame_indices, fps, frame_count


@lru_cache(maxsize=4)
def load_decision_layer(experiment_name: str) -> tuple[Any, dict[str, Any]]:
    cfg = EXPERIMENTS[experiment_name]
    pipeline = joblib.load(cfg.decision_layer_path)
    decision_config = json.loads(cfg.decision_config_path.read_text(encoding="utf-8"))
    return pipeline, decision_config


@lru_cache(maxsize=2)
def load_yolo_model(model_key: str) -> YOLO:
    return YOLO(str(MODEL_WEIGHTS[model_key]))


@lru_cache(maxsize=1)
def load_reference_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    split_metrics = pd.read_csv(REPORTS_DIR / "tables" / "experiment_split_metrics.csv")
    latency_metrics = pd.read_csv(REPORTS_DIR / "tables" / "latency_results.csv")
    return split_metrics, latency_metrics


def build_reference_tables(experiment_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_metrics, latency_metrics = load_reference_metrics()
    experiment_cfg = EXPERIMENTS[experiment_name]
    split_subset = (
        split_metrics.loc[
            split_metrics["experiment_name"] == experiment_name,
            ["split", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "threshold", "n_samples"],
        ]
        .copy()
        .round(4)
    )
    latency_subset = (
        latency_metrics.loc[
            latency_metrics["model_key"] == experiment_cfg.model_key,
            ["format", "device", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "fps"],
        ]
        .copy()
        .round(3)
    )
    return split_subset, latency_subset


def build_live_summary(
    experiment_cfg: ExperimentConfig,
    clip_features: dict[str, Any],
    probability: float,
    threshold: float,
    processed_frames: int,
    total_frames: int,
    fps: float,
    latency_values_ms: list[float],
) -> pd.DataFrame:
    mean_latency = float(np.mean(latency_values_ms)) if latency_values_ms else np.nan
    processing_fps = float(1000.0 / mean_latency) if latency_values_ms and mean_latency > 0 else np.nan
    last_latency = float(latency_values_ms[-1]) if latency_values_ms else np.nan
    prediction = "Fall" if probability >= threshold else "No Fall"
    rows = [
        {"metric": "display_name", "value": experiment_cfg.display_name},
        {"metric": "prediction", "value": prediction},
        {"metric": "fall_probability", "value": round(probability, 4)},
        {"metric": "decision_threshold", "value": round(threshold, 4)},
        {"metric": "decision_margin", "value": round(probability - threshold, 4)},
        {"metric": "frames_processed", "value": f"{processed_frames}/{total_frames}"},
        {"metric": "video_fps", "value": round(fps, 3) if not math.isnan(fps) else "unknown"},
        {"metric": "last_frame_latency_ms", "value": round(last_latency, 3) if not math.isnan(last_latency) else "n/a"},
        {"metric": "mean_frame_latency_ms", "value": round(mean_latency, 3) if not math.isnan(mean_latency) else "n/a"},
        {"metric": "mean_processing_fps", "value": round(processing_fps, 3) if not math.isnan(processing_fps) else "n/a"},
        {"metric": "detected_ratio", "value": round(float(clip_features.get("detected_ratio", np.nan)), 4)},
        {"metric": "near_floor_ratio", "value": round(float(clip_features.get("near_floor_ratio", np.nan)), 4)},
        {"metric": "horizontal_pose_ratio", "value": round(float(clip_features.get("horizontal_pose_ratio", np.nan)), 4)},
        {"metric": "wide_bbox_ratio", "value": round(float(clip_features.get("wide_bbox_ratio", np.nan)), 4)},
        {"metric": "low_torso_ratio", "value": round(float(clip_features.get("low_torso_ratio", np.nan)), 4)},
        {"metric": "privacy_mode", "value": experiment_cfg.privacy_mode},
        {"metric": "privacy_score", "value": experiment_cfg.privacy_score},
    ]
    return pd.DataFrame(rows)


def build_probability_plot(probability_history: list[float], threshold: float):
    fig, ax = plt.subplots(figsize=(6, 3))
    if probability_history:
        x_values = np.arange(1, len(probability_history) + 1)
        ax.plot(x_values, probability_history, marker="o", linewidth=2, color="#b1440e")
        ax.scatter(x_values[-1], probability_history[-1], s=80, color="#1f5c99")
    ax.axhline(threshold, linestyle="--", linewidth=1.5, color="#444444", label=f"threshold={threshold:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Processed Sampled Frame")
    ax.set_ylabel("Fall Probability")
    ax.set_title("Running Clip Probability")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def frame_metrics_view(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(
            columns=[
                "frame_index",
                "timestamp_seconds",
                "detected",
                "detection_conf",
                "bbox_aspect_ratio",
                "bottom_y_norm",
                "horizontal_body_ratio",
                "torso_span_norm",
            ]
        )
    columns = [
        "frame_index",
        "timestamp_seconds",
        "detected",
        "detection_conf",
        "bbox_aspect_ratio",
        "bottom_y_norm",
        "horizontal_body_ratio",
        "torso_span_norm",
    ]
    available_columns = [column for column in columns if column in frame_df.columns]
    return frame_df[available_columns].copy().round(4)


def render_result_frame(
    experiment_cfg: ExperimentConfig,
    prepared_frame: np.ndarray,
    result: Any,
    probability: float,
    threshold: float,
    frame_note: str,
) -> np.ndarray:
    if experiment_cfg.model_key == "yolov8n_pose":
        base_frame = render_skeleton_canvas(prepared_frame.shape, result)
    else:
        base_frame = result.plot()
    if base_frame is None or not isinstance(base_frame, np.ndarray):
        base_frame = prepared_frame.copy()
    label = "Fall" if probability >= threshold else "No Fall"
    return overlay_prediction_banner(base_frame, label, probability, threshold, frame_note)


def save_preview_video(frames_rgb: list[np.ndarray], stem: str, fps: float) -> str | None:
    if not frames_rgb:
        return None
    output_path = OUTPUT_DIR / f"{sanitize_name(stem)}_annotated.mp4"
    height, width = frames_rgb[0].shape[:2]
    video_fps = fps if fps and fps > 0 else 15.0
    # video_fps = float(min(max(video_fps, 2.0), 8.0))
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc")
    writer = cv2.VideoWriter(str(output_path), int(fourcc_fn(*"mp4v")), video_fps, (width, height))
    if not writer.isOpened():
        return None
    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return str(output_path)


def status_message(experiment_cfg: ExperimentConfig, video_path: str, processed_frames: int, total_frames: int, probability: float, threshold: float) -> str:
    label = "Fall" if probability >= threshold else "No Fall"
    return (
        f"### {experiment_cfg.display_name}\n"
        f"- Video: `{Path(video_path).name}`\n"
        f"- Progress: `{processed_frames}/{total_frames}` sampled frames processed\n"
        f"- Current prediction: `{label}`\n"
        f"- Fall probability: `{probability:.4f}`\n"
        f"- Decision threshold: `{threshold:.4f}`"
    )


def normalize_video_path(video_value: Any) -> str:
    if isinstance(video_value, Path):
        return str(video_value)
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, tuple) and video_value:
        first = video_value[0]
        if isinstance(first, Path):
            return str(first)
        if isinstance(first, str):
            return first
    raise gr.Error("Unsupported video input format returned by Gradio.")


def run_inference(
    video_path: Any,
    experiment_name: str,
    sample_count: int,
    progress=gr.Progress(track_tqdm=False),
):
    if not video_path:
        raise gr.Error("Upload or record a video first.")
    resolved_video_path = normalize_video_path(video_path)
    if experiment_name not in EXPERIMENTS:
        raise gr.Error(f"Unknown experiment: {experiment_name}")

    experiment_cfg = EXPERIMENTS[experiment_name]
    pipeline, decision_config = load_decision_layer(experiment_name)
    model = load_yolo_model(experiment_cfg.model_key)
    threshold = float(decision_config["best_threshold"])
    feature_columns = list(decision_config["feature_columns"])
    reference_split_df, reference_latency_df = build_reference_tables(experiment_name)

    sample_indices, video_fps, total_video_frames = choose_sample_indices(resolved_video_path, sample_count)
    sample_index_set = set(sample_indices)
    clip_row = {
        "clip_id": sanitize_name(Path(resolved_video_path).stem),
        "subject": "uploaded",
        "split": "inference",
        "binary_label": "unknown",
        "video_path": resolved_video_path,
    }

    capture = cv2.VideoCapture(str(resolved_video_path))
    if not capture.isOpened():
        raise gr.Error(f"Could not open uploaded video: {resolved_video_path}")

    processed_sample_count = 0
    frame_records: list[dict[str, Any]] = []
    annotated_frames_rgb: list[np.ndarray] = []
    probability_history: list[float] = []
    latency_values_ms: list[float] = []
    latest_preview_rgb = np.zeros((360, 640, 3), dtype=np.uint8)

    try:
        for frame_index in range(total_video_frames):
            success, frame_bgr = capture.read()
            if not success or frame_bgr is None:
                break
            if frame_index not in sample_index_set:
                continue

            frame_timestamp = (frame_index / video_fps) if video_fps > 0 else np.nan
            prepared_frame = apply_privacy_transform_to_input(frame_bgr, experiment_cfg.privacy_mode)
            start_time = perf_counter()
            result = model.predict(
                source=[prepared_frame],
                conf=MIN_PERSON_CONF,
                imgsz=INFERENCE_IMG_SIZE,
                max_det=MAX_DETECTIONS,
                verbose=False,
                device="cpu",
            )[0]
            elapsed_ms = (perf_counter() - start_time) * 1000.0
            latency_values_ms.append(elapsed_ms)

            feature_extractor = extract_pose_frame_features if experiment_cfg.model_key == "yolov8n_pose" else extract_detection_frame_features
            frame_features = feature_extractor(result, prepared_frame.shape)
            frame_features.update(
                {
                    "clip_id": clip_row["clip_id"],
                    "subject": clip_row["subject"],
                    "split": clip_row["split"],
                    "binary_label": clip_row["binary_label"],
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_timestamp,
                    "experiment_name": experiment_cfg.experiment_name,
                    "model_key": experiment_cfg.model_key,
                    "privacy_mode": experiment_cfg.privacy_mode,
                }
            )
            frame_records.append(frame_features)
            frame_df = pd.DataFrame(frame_records)
            clip_features = aggregate_clip_features(frame_df, clip_row, experiment_cfg)
            clip_feature_df = pd.DataFrame([clip_features], columns=feature_columns)
            probability = float(pipeline.predict_proba(clip_feature_df)[0, 1])
            probability_history.append(probability)

            processed_sample_count += 1
            frame_note = f"sample {processed_sample_count}/{len(sample_indices)} | frame {frame_index + 1}/{total_video_frames}"
            preview_bgr = render_result_frame(experiment_cfg, prepared_frame, result, probability, threshold, frame_note)
            latest_preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            annotated_frames_rgb.append(latest_preview_rgb)

            live_summary_df = build_live_summary(
                experiment_cfg=experiment_cfg,
                clip_features=clip_features,
                probability=probability,
                threshold=threshold,
                processed_frames=processed_sample_count,
                total_frames=len(sample_indices),
                fps=video_fps,
                latency_values_ms=latency_values_ms,
            )
            progress(processed_sample_count / max(len(sample_indices), 1), desc=f"Processing sampled frame {processed_sample_count}/{len(sample_indices)}")
            yield (
                status_message(experiment_cfg, resolved_video_path, processed_sample_count, len(sample_indices), probability, threshold),
                reference_split_df,
                reference_latency_df,
                live_summary_df,
                build_probability_plot(probability_history, threshold),
                frame_metrics_view(frame_df),
                latest_preview_rgb,
                None,
            )
    finally:
        capture.release()

    if not frame_records:
        raise gr.Error("No sampled frames were processed from the uploaded video.")

    final_frame_df = pd.DataFrame(frame_records)
    final_clip_features = aggregate_clip_features(final_frame_df, clip_row, experiment_cfg)
    final_clip_feature_df = pd.DataFrame([final_clip_features], columns=feature_columns)
    final_probability = float(pipeline.predict_proba(final_clip_feature_df)[0, 1])
    final_video_path = save_preview_video(annotated_frames_rgb, clip_row["clip_id"], video_fps)
    final_live_summary_df = build_live_summary(
        experiment_cfg=experiment_cfg,
        clip_features=final_clip_features,
        probability=final_probability,
        threshold=threshold,
        processed_frames=len(sample_indices),
        total_frames=len(sample_indices),
        fps=video_fps,
        latency_values_ms=latency_values_ms,
    )

    yield (
        status_message(experiment_cfg, resolved_video_path, len(sample_indices), len(sample_indices), final_probability, threshold),
        reference_split_df,
        reference_latency_df,
        final_live_summary_df,
        build_probability_plot(probability_history, threshold),
        frame_metrics_view(final_frame_df),
        latest_preview_rgb,
        final_video_path,
    )


def refresh_reference_metrics(experiment_name: str):
    if experiment_name not in EXPERIMENTS:
        return pd.DataFrame(), pd.DataFrame()
    return build_reference_tables(experiment_name)


def build_interface() -> gr.Blocks:
    default_experiment = "yolov8n_pose_rgb"
    default_reference_split_df, default_reference_latency_df = build_reference_tables(default_experiment)

    with gr.Blocks(title="Patient Safety Monitoring Inference") as demo:
        gr.Markdown(
            """
            # Patient Safety Monitoring Inference
            Upload a video or record a short clip, choose one of the trained project variants, and run clip-level inference.
            The interface shows the stored project evaluation metrics for the selected model and updates live inference metrics as sampled frames are processed.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")
                experiment_input = gr.Dropdown(
                    label="Model Variant",
                    choices=[(label, key) for key, label in EXPERIMENT_LABELS.items()],
                    value=default_experiment,
                )
                sample_count_input = gr.Slider(label="Sampled Frames Per Video", minimum=4, maximum=32, step=1, value=FRAME_SAMPLE_COUNT)
                run_button = gr.Button("Run Inference", variant="primary")
                clear_button = gr.Button("Clear")

            with gr.Column(scale=1):
                status_output = gr.Markdown(label="Status")
                preview_output = gr.Image(label="Live Preview", type="numpy")
                video_output = gr.Video(label="Annotated Sampled-Frame Video")

        with gr.Row():
            reference_split_output = gr.Dataframe(label="Stored Evaluation Metrics", value=default_reference_split_df, interactive=False)
            reference_latency_output = gr.Dataframe(label="Stored Latency Benchmarks", value=default_reference_latency_df, interactive=False)

        with gr.Row():
            live_summary_output = gr.Dataframe(label="Live Inference Metrics", interactive=False)
            probability_plot_output = gr.Plot(label="Running Probability")

        frame_metrics_output = gr.Dataframe(label="Frame-Level Metrics", interactive=False)

        experiment_input.change(
            fn=refresh_reference_metrics,
            inputs=experiment_input,
            outputs=[reference_split_output, reference_latency_output],
        )

        run_button.click(
            fn=run_inference,
            inputs=[video_input, experiment_input, sample_count_input],
            outputs=[
                status_output,
                reference_split_output,
                reference_latency_output,
                live_summary_output,
                probability_plot_output,
                frame_metrics_output,
                preview_output,
                video_output,
            ],
        )

        clear_button.click(
            fn=lambda: ("", default_reference_split_df, default_reference_latency_df, pd.DataFrame(), None, pd.DataFrame(), None, None),
            inputs=None,
            outputs=[
                status_output,
                reference_split_output,
                reference_latency_output,
                live_summary_output,
                probability_plot_output,
                frame_metrics_output,
                preview_output,
                video_output,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.queue().launch(share=True, show_error=True)