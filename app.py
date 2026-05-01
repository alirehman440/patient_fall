
import json
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import gradio as gr
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- PATHS ----------------
ROOT_DIR = Path.cwd()
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_DIR = ROOT_DIR / "gradio_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- CONSTANTS ----------------
MIN_PERSON_CONF = 0.20
INFERENCE_IMG_SIZE = 640
MAX_DETECTIONS = 5

# ---------------- EXPERIMENT CONFIG ----------------
class ExperimentConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "yolov8n_rgb": ExperimentConfig(
        model_key="yolov8n",
        privacy_mode="full_rgb",
        display_name="YOLOv8n RGB",
        decision_layer_path=MODELS_DIR / "yolov8n/yolov8n_rgb/decision_layer.joblib",
        decision_config_path=MODELS_DIR / "yolov8n/yolov8n_rgb/decision_layer_config.json",
    ),
    "yolov8n_blur": ExperimentConfig(
        model_key="yolov8n",
        privacy_mode="blurred_rgb",
        display_name="YOLOv8n Blur",
        decision_layer_path=MODELS_DIR / "yolov8n/yolov8n_blur/decision_layer.joblib",
        decision_config_path=MODELS_DIR / "yolov8n/yolov8n_blur/decision_layer_config.json",
    ),
    "yolov8n_pose_rgb": ExperimentConfig(
        model_key="yolov8n_pose",
        privacy_mode="full_rgb",
        display_name="YOLOv8n Pose",
        decision_layer_path=MODELS_DIR / "yolov8n_pose/yolov8n_pose_rgb/decision_layer.joblib",
        decision_config_path=MODELS_DIR / "yolov8n_pose/yolov8n_pose_rgb/decision_layer_config.json",
    ),
}

MODEL_WEIGHTS = {
    "yolov8n": ROOT_DIR / "yolov8n.pt",
    "yolov8n_pose": ROOT_DIR / "yolov8n-pose.pt",
}

# ---------------- LOAD CACHE ----------------
MODEL_CACHE = {}
PIPELINE_CACHE = {}


def load_artifacts():
    for k, v in MODEL_WEIGHTS.items():
        if v.exists():
            MODEL_CACHE[k] = YOLO(str(v))

    for name, cfg in EXPERIMENTS.items():
        if cfg.decision_layer_path.exists():
            PIPELINE_CACHE[name] = {
                "pipeline": joblib.load(cfg.decision_layer_path),
                "config": json.loads(cfg.decision_config_path.read_text())
            }


load_artifacts()

# ---------------- HELPERS ----------------
def apply_privacy(frame, mode):
    if mode == "blurred_rgb":
        return cv2.GaussianBlur(frame, (31, 31), 0)
    return frame


def get_person(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    persons = np.where(cls == 0)[0]
    if len(persons) == 0:
        return None

    return xyxy[persons[0]]


def extract_features(result, frame_shape):
    h, w = frame_shape[:2]
    det = get_person(result)

    base = {
        "detected": 0,
        "bbox_area_norm": 0,
        "bbox_aspect_ratio": 0,
    }

    if det is None:
        return base

    x1, y1, x2, y2 = det
    bw, bh = x2 - x1, y2 - y1

    base.update({
        "detected": 1,
        "bbox_area_norm": (bw * bh) / (h * w),
        "bbox_aspect_ratio": bw / (bh + 1e-6),
    })

    return base


# ---------------- MAIN INFERENCE ----------------
def predict(video, experiment):
    if video is None:
        return "No video", None

    cfg = EXPERIMENTS[experiment]
    model = MODEL_CACHE.get(cfg.model_key)
    pipe = PIPELINE_CACHE.get(experiment)

    if model is None or pipe is None:
        return "Model not loaded", None

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Cannot read video", None

    frame = apply_privacy(frame, cfg.privacy_mode)

    result = model.predict(frame, conf=MIN_PERSON_CONF, imgsz=INFERENCE_IMG_SIZE)[0]
    features = extract_features(result, frame.shape)

    df = pd.DataFrame([features])
    features = df[pipe["config"]["feature_columns"]]

    prob = pipe["pipeline"].predict_proba(features)[0][1]
    label = "FALL" if prob > pipe["config"]["best_threshold"] else "NO FALL"

    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return f"{label} ({prob:.2f})", annotated


# ---------------- UI ----------------
with gr.Blocks() as app:
    gr.Markdown("# Patient Fall Detection System")

    with gr.Row():
        video_input = gr.Video()
        model_input = gr.Dropdown(
            choices=list(EXPERIMENTS.keys()),
            value="yolov8n_pose_rgb"
        )

    btn = gr.Button("Run")

    out_text = gr.Textbox()
    out_img = gr.Image()

    btn.click(predict, [video_input, model_input], [out_text, out_img])


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.queue().launch(share=True, show_error=True)
