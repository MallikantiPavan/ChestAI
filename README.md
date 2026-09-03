# ChestAI — Chest X-Ray Multi-Label Classification with DenseNet-ECA

A deep learning pipeline for **multi-label chest X-ray pathology classification** using a **DenseNet-121 backbone enhanced with Efficient Channel Attention (ECA)**. The project includes model training/evaluation scripts, Grad-CAM visual explainability, and a full-stack web application (FastAPI + vanilla JS) for real-time inference.

> **⚠️ Disclaimer:** This tool is intended for **research and educational purposes only**. It is not a substitute for clinical diagnosis or the judgment of a qualified radiologist.

---

## Table of Contents

- [Features](#features)
- [Supported Pathologies](#supported-pathologies)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Testing / Inference](#testing--inference)
  - [Evaluation](#evaluation)
  - [Grad-CAM Visualization](#grad-cam-visualization)
  - [Web Application](#web-application)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Features

- **DenseNet-121 + ECA** attention for improved channel-wise feature recalibration
- **Multi-label binary classification** across 5 thoracic pathologies
- **Grad-CAM heatmaps** overlaid on the original X-ray for visual explainability
- **Full-stack web app** — drag-and-drop X-ray upload, real-time predictions, severity labels, PDF export, analysis history, and dark/light theme toggle
- **Configurable training** via a single JSON config file (optimizer, learning rate schedule, augmentation, pooling strategy, etc.)
- **Evaluation tooling** — ROC curve plots, confusion matrix generation, and per-class AUC/accuracy metrics
- **TensorBoard integration** for live training/validation monitoring
- **Multi-GPU support** via PyTorch `DataParallel`

---

## Supported Pathologies

| # | Pathology         |
|---|-------------------|
| 1 | Atelectasis       |
| 2 | Cardiomegaly      |
| 3 | Consolidation     |
| 4 | Edema             |
| 5 | Pleural Effusion  |

---

## Architecture Overview

```
Input (224×224 grayscale)
        │
        ▼
  DenseNet-121 Backbone (ImageNet pre-trained)
        │
        ▼
  ECA Attention Module  ←  Adaptive 1D conv on channel descriptors
        │
        ▼
  Global Pooling (AVG + MAX)
        │
        ▼
  BatchNorm → Dropout → 1×1 Conv Classifier (×5 heads)
        │
        ▼
  Sigmoid → Per-class probabilities
```

The **Efficient Channel Attention (ECA)** module replaces heavier SE-style blocks with a lightweight 1D convolution over the channel dimension after global average pooling, keeping the parameter overhead minimal while still boosting representational power.

Multiple attention variants are included in the codebase and can be toggled via config:

| Key   | Module                          |
|-------|---------------------------------|
| `ECA` | Efficient Channel Attention     |
| `CAM` | Channel Attention Module (SE)   |
| `SAM` | Spatial Attention Module         |
| `FPA` | Feature Pyramid Attention        |

---

## Project Structure

```
classification/
├── backend/
│   └── app.py                  # FastAPI server — /predict and /health endpoints
├── bin/
│   ├── train.py                # Training loop with validation & checkpointing
│   ├── test.py                 # Batch inference → prediction CSV
│   ├── roc.py                  # Per-class ROC curve plotting
│   ├── confusion_matrix.py     # Confusion matrix + per-class metrics
│   ├── confusion_metric_1.py   # Extended confusion matrix with visualization
│   └── heatmap.py              # Batch Grad-CAM heatmap generation
├── config/
│   └── example.json            # Training/model configuration
├── data/
│   ├── dataset.py              # ImageDataset (PyTorch Dataset, CheXpert-style CSV)
│   ├── imgaug.py               # Data augmentation transforms
│   └── utils.py                # Image preprocessing (resize, normalize, pad)
├── frontend/
│   ├── index.html              # Single-page web UI
│   ├── script.js               # Client-side logic (upload, predict, history, PDF)
│   └── style.css               # Full styling with dark/light theme support
├── model/
│   ├── backbone/
│   │   └── densenet.py         # DenseNet-121 implementation
│   ├── attention_map.py        # ECA, CA, SA, FPA attention modules
│   ├── classifier.py           # Main Classifier nn.Module
│   ├── global_pool.py          # AVG, MAX, AVG_MAX, LSE pooling strategies
│   └── utils.py                # Optimizer factory, normalization helpers
├── utils/
│   ├── heatmaper.py            # Heatmap overlay utilities
│   └── misc.py                 # LR schedule helper
├── grad_cam.py                 # GradCAM class + preprocessing/overlay helpers
├── grad_cam_folder.py          # Batch Grad-CAM over a directory of images
├── grad_cam_single_image.py    # Single-image Grad-CAM script
├── run_gradcam.py              # CLI entry-point for Grad-CAM generation
├── chekpoint.py                # Checkpoint inspection utility
├── requirements.txt            # Python dependencies
└── run_code.txt                # Example CLI commands reference
```

---

## Prerequisites

- **Python** 3.8+
- **CUDA**-capable GPU (recommended; CPU inference is supported but slow)
- **pip** or **conda** for dependency management

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/chestai-densenet-eca.git
   cd chestai-densenet-eca/classification
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Additionally, install **FastAPI** and **Uvicorn** for the web application:

   ```bash
   pip install fastapi uvicorn python-multipart
   ```

---

## Configuration

All training and model parameters are controlled through a single JSON config file. See [`config/example.json`](config/example.json) for the full reference.

| Parameter            | Description                                               | Default        |
|----------------------|-----------------------------------------------------------|----------------|
| `backbone`           | CNN backbone architecture                                 | `densenet121`  |
| `attention_map`      | Attention module (`ECA`, `CAM`, `SAM`, `FPA`, `None`)     | `ECA`          |
| `global_pool`        | Pooling strategy (`AVG`, `MAX`, `AVG_MAX`, `AVG_MAX_LSE`) | `AVG_MAX`      |
| `optimizer`          | Optimizer (`Adam`, `SGD`)                                 | `Adam`         |
| `lr`                 | Initial learning rate                                     | `0.0001`       |
| `epoch`              | Number of training epochs                                 | `10`           |
| `train_batch_size`   | Training batch size                                       | `32`           |
| `criterion`          | Loss function (`BCE`)                                     | `BCE`          |
| `best_target`        | Metric to track for best checkpoint (`auc`, `acc`, `loss`)| `auc`          |
| `width` / `height`   | Input image dimensions                                    | `224`          |
| `use_equalizeHist`   | Apply histogram equalization to inputs                    | `true`         |
| `train_csv`          | Path to training labels CSV (CheXpert format)             | —              |
| `dev_csv`            | Path to validation labels CSV                             | —              |

---

## Usage

### Training

```bash
python bin/train.py config/example.json ./checkpoints \
    --device_ids "0" \
    --num_workers 8 \
    --logtofile True
```

- Checkpoints are saved as `best1.ckpt`, `best2.ckpt`, etc. (top-k by AUC)
- A running `train.ckpt` is saved every epoch for resume support
- TensorBoard logs are written to the save path

**Resume training:**

```bash
python bin/train.py config/example.json ./checkpoints \
    --device_ids "0" \
    --resume 1
```

**Monitor with TensorBoard:**

```bash
tensorboard --logdir ./checkpoints
```

### Testing / Inference

```bash
python bin/test.py \
    --model_path ./checkpoints/ \
    --in_csv_path /path/to/test_labels.csv \
    --out_csv_path ./results/predictions.csv \
    --device_ids "0"
```

This generates a CSV with per-image sigmoid probabilities for each pathology.

### Evaluation

**ROC Curves:**

```bash
python bin/roc.py dev_eval \
    --pred_csv_path ./results/predictions.csv \
    --true_csv_path /path/to/test_labels.csv \
    --plot_path ./results/
```

**Confusion Matrix:**

```bash
python bin/confusion_metric_1.py dev_eval \
    --pred_csv_path ./results/predictions.csv \
    --true_csv_path /path/to/test_labels.csv \
    --out_path ./results/ \
    --prob_thred 0.5
```

Outputs include:
- Per-class confusion matrix PNG
- CSV with TN, FP, FN, TP, accuracy, and AUC per pathology

### Grad-CAM Visualization

**Single image:**

```bash
python grad_cam_single_image.py \
    --image_path /path/to/xray.jpg \
    --model_path ./checkpoints/ \
    --class_idx 0 \
    --save_path ./output/gradcam_result.jpg
```

**Batch (folder):**

```bash
python grad_cam_folder.py \
    --image_dir /path/to/images/ \
    --model_path ./checkpoints/ \
    --save_dir ./output/gradcam/
```

### Web Application

1. **Start the backend server:**

   ```bash
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open the frontend:**

   Open `frontend/index.html` in your browser, or serve it with any static file server:

   ```bash
   # Using Python's built-in server
   cd frontend
   python -m http.server 5500
   ```

3. **Use the app:**
   - Upload a chest X-ray (JPEG/PNG)
   - Adjust confidence threshold and heatmap opacity
   - Click **Analyze X-Ray** to get predictions with Grad-CAM overlays
   - Export results as PDF or share via link
   - View analysis history in the History tab

---

## API Reference

### `GET /health`

Returns the current server status, device info, and model name.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda:0",
  "model": "DenseNet-ECA",
  "timestamp": "2024-01-01T00:00:00"
}
```

### `POST /predict`

Upload a chest X-ray image for multi-label classification with Grad-CAM.

**Request:** `multipart/form-data` with a `file` field (JPEG or PNG).

**Response:**
```json
{
  "results": [
    {
      "class": "Cardiomegaly",
      "prob": 0.8732,
      "pred": 1,
      "severity": "High",
      "image": "<base64-encoded Grad-CAM overlay>"
    }
  ],
  "original": "<base64-encoded original image>",
  "top_class": "Cardiomegaly",
  "top_prob": 0.8732,
  "top_severity": "High",
  "timestamp": "2024-01-01T00:00:00",
  "model_info": {
    "name": "DenseNet-ECA",
    "threshold": 0.5,
    "device": "cuda:0"
  }
}
```

**Severity levels:**
| Probability  | Severity |
|-------------|----------|
| ≥ 0.75      | High     |
| ≥ 0.50      | Moderate |
| ≥ 0.25      | Low      |
| < 0.25      | Normal   |

---

## Tech Stack

| Layer      | Technology                              |
|------------|----------------------------------------|
| Model      | PyTorch, DenseNet-121, ECA Attention   |
| Training   | BCE Loss, Adam, LR Step Scheduler     |
| Logging    | TensorboardX, Python logging           |
| Backend    | FastAPI, Uvicorn                       |
| Frontend   | HTML5, Vanilla JS, CSS (dark/light)    |
| Evaluation | scikit-learn, matplotlib               |
| Data       | OpenCV, PIL, NumPy                     |

---

## Acknowledgements

- **CheXpert** dataset by Stanford ML Group — [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **ECA-Net** — Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" (CVPR 2020)
- **DenseNet** — Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)
- **Grad-CAM** — Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)

---

## License

This project is provided as-is for academic and research purposes. Please check with the repository owner for specific licensing terms.
