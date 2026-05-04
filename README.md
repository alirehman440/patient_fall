# Edge AI for Patient Safety Monitoring in Hospital Rooms

This project evaluates lightweight YOLOv8n-based approaches for patient fall and safety monitoring under edge deployment constraints. The focus is not only on detection quality, but also on inference efficiency and privacy-preserving output modes.

## Project Scope

The project uses the GMDCSA-24 video dataset as a practical stand-in for patient safety monitoring research. It compares three main variants:

- `yolov8n_rgb`: standard RGB inference baseline
- `yolov8n_blur`: blurred-frame privacy baseline
- `yolov8n_pose_rgb`: pose/keypoint-based pipeline built on `yolov8n-pose`

The study benchmarks:

- classification performance on a subject-aware split
- latency on local CPU
- export readiness for PyTorch, ONNX, and OpenVINO
- privacy-oriented output modes such as blurred frames, masked-person views, and skeleton-style rendering

## Dataset

Primary dataset:

- [GMDCSA-24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos#gmdcsa24-a-dataset-for-human-fall-detection-in-videos): downloaded from the official GitHub repository.

Dataset summary from the completed run:

- Total clips: `160`
- Fall clips: `79`
- ADL clips: `81`
- Subjects: `4`
- Median clip duration: `7.0` seconds
- Median FPS: `29.825`

Subject-aware split used:

- Train: Subject 1 and Subject 2
- Validation: Subject 3
- Test: Subject 4

This split avoids leakage across subjects, but the dataset is still limited in size and is not an actual hospital-room dataset.


## Approach

The notebook is organized as a full end-to-end pipeline:

1. Environment setup and folder creation
2. Dataset download with skip logic if the dataset already exists
3. EDA and metadata extraction
4. Subject-aware train/validation/test split
5. YOLOv8n and YOLOv8n-pose loading
6. Clip-level feature extraction with caching
7. Lightweight decision-layer training and threshold tuning
8. Evaluation, confusion matrices, ROC/PR plots, and comparison dashboards
9. Export to ONNX and OpenVINO
10. Latency benchmarking
11. Privacy output generation

The pipeline uses pretrained YOLO backbones and then trains a lightweight decision layer on extracted clip features. This keeps the project practical for a single-notebook workflow.

## Key Results

Test-set results from the completed run:

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | Mean CPU Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n RGB | 0.6757 | 0.6471 | 0.6471 | 0.6471 | 0.8059 | 64.33 ms |
| YOLOv8n Blurred | 0.7297 | 0.6842 | 0.7647 | 0.7222 | 0.7824 | 64.33 ms |
| YOLOv8n-Pose | 0.8378 | 0.7619 | 0.9412 | 0.8421 | 0.9382 | 39.27 ms |

Main interpretation:

- `yolov8n_pose_rgb` produced the strongest overall result
- it achieved the best recall and F1, which is important for fall-event sensitivity
- it also had the best measured local CPU latency in this run
- the blurred baseline outperformed the plain RGB baseline on this split, but that result should be treated cautiously because the dataset is small

## Privacy Interpretation

The project includes several privacy modes:

- `full_rgb`: original frame retained
- `blurred_rgb`: full-frame blur
- `masked_person`: keep subject while removing background
- `skeleton_only`: render only pose keypoints on a blank canvas

The pose-based pipeline is the most promising route for privacy-preserving monitoring because the downstream retained representation can be reduced to skeletal information rather than storing full video frames.



Latency findings from the local CPU benchmark:

- YOLOv8n PT: `64.33 ms`
- YOLOv8n ONNX: `56.92 ms`
- YOLOv8n OpenVINO: `68.04 ms`
- YOLOv8n-pose PT: `39.27 ms`
- YOLOv8n-pose ONNX: `57.78 ms`
- YOLOv8n-pose OpenVINO: `77.10 ms`

The practical lesson is that export format should be benchmarked on the target hardware. It should not be assumed that OpenVINO or ONNX will always be faster.

## Project Outputs

Important generated outputs include:

- Metrics table: `reports/tables/experiment_split_metrics.csv`
- Latency table: `reports/tables/latency_results.csv`
- Trade-off summary: `reports/summaries/accuracy_latency_privacy_tradeoff.csv`
- Final comparison plot: `visuals/comparisons/final_tradeoff_dashboard.png`
- Privacy examples: `visuals/privacy/privacy_modes_Subject_4_FALL_01.png`


## How To Run

### Quick Start

To open and run the project interface:

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install the required packages (see Requirements section below).

2. **Run the interface**:
   ```bash
   python interface.py
   ```
   This will open the project interface where you can interact with the patient fall detection system.


## Requirements

The project uses standard Python tooling plus a few ML and CV packages, including:

- `ultralytics`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `tqdm`
- `psutil`

If packages are missing when running `interface.py`, the script will guide you to install any missing dependencies. You can also install all dependencies at once using:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib seaborn scikit-learn joblib tqdm psutil
```

