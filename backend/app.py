import os
import json
import base64
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from easydict import EasyDict as edict
from datetime import datetime

from model.classifier import Classifier
from grad_cam import GradCAM, preprocess_image, overlay_heatmap, combine_images

CFG_PATH  = r"D:\densenet_eca\final_label_eca_adam_224\classification\config\example.json"
CKPT_PATH = r"D:\densenet_eca\final_label_eca_adam_224\best1.ckpt"

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

THRESHOLD = 0.5

with open(CFG_PATH) as f:
    cfg = edict(json.load(f))

model = Classifier(cfg)
ckpt  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
model.eval()

gradcam = GradCAM(model)

app = FastAPI(title="Chest X-Ray AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


def severity_label(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    elif prob >= 0.50:
        return "Moderate"
    elif prob >= 0.25:
        return "Low"
    else:
        return "Normal"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model": "DenseNet-ECA",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_img    = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Could not decode image. Please upload a valid JPEG/PNG."}

    temp_path = "temp_upload.jpg"
    cv2.imwrite(temp_path, img)

    image_tensor, image_color = preprocess_image(temp_path, cfg)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits, _ = model(image_tensor)

    logits = torch.stack(logits)
    probs  = torch.sigmoid(logits).cpu().numpy().flatten()
    preds  = (probs >= THRESHOLD).astype(int)

    thumb = cv2.resize(image_color, (224, 224))
    original_b64 = to_base64(thumb)

    results = []
    for i, cls in enumerate(CLASS_NAMES):
        cam     = gradcam.generate(image_tensor, i)
        overlay = overlay_heatmap(image_color, cam)

        cam_resized = cv2.resize(overlay, (448, 224))

        results.append({
            "class":    cls,
            "prob":     round(float(probs[i]), 4),
            "pred":     int(preds[i]),
            "severity": severity_label(float(probs[i])),
            "image":    to_base64(cam_resized),   
        })
    results.sort(key=lambda x: x["prob"], reverse=True)

    top = results[0]

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "results":       results,
        "original":      original_b64,
        "top_class":     top["class"],
        "top_prob":      top["prob"],
        "top_severity":  top["severity"],
        "timestamp":     datetime.utcnow().isoformat(),
        "model_info": {
            "name":       "DenseNet-ECA",
            "threshold":  THRESHOLD,
            "device":     str(device),
        }
    }
