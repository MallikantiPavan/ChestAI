import os
import cv2
import torch
import json
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

from model.classifier import Classifier
from grad_cam import GradCAM, overlay_heatmap, combine_images, preprocess_image

CFG_PATH = r"D:\densenet_eca\final_label_eca_adam_224\classification\config\example.json"
CKPT_PATH = r"D:\densenet_eca\final_label_eca_adam_224\best1.ckpt"
GT_CSV = r"D:\densenet_eca\dense_eca\demo\grad_Cam_images_path.csv"
SAVE_DIR = r"D:\densenet_eca\dense_eca\demo\images"

THRESHOLD = 0.5

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

os.makedirs(SAVE_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = edict(json.load(f))

model = Classifier(cfg)

ckpt = torch.load(CKPT_PATH, map_location="cuda:0",weights_only=False)
state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda:0")
model = model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad = True

gradcam = GradCAM(model)

df = pd.read_csv(GT_CSV)

def clean_label(val):
    if val in [1, 1.0, "1.0"]:
        return 1
    elif val in [0, 0.0, "0.0"]:
        return 0
    else:
        return None  

def get_result(pred, gt):
    if gt is None:
        return "IGNORED"
    if pred == 1 and gt == 1:
        return "TP"
    elif pred == 1 and gt == 0:
        return "FP"
    elif pred == 0 and gt == 1:
        return "FN"
    else:
        return "TN"

for _, row in df.iterrows():

    img_path = row["Path"]

    if not os.path.exists(img_path):
        print("Missing:", img_path)
        continue

    print("Processing:", img_path)

    image_tensor, image_color = preprocess_image(img_path, cfg)
    image_tensor = image_tensor.to(device)

    logits, _ = model(image_tensor)
    logits = torch.stack(logits)

    probs = torch.sigmoid(logits).cpu().detach().numpy().flatten()
    preds = (probs >= THRESHOLD).astype(int)

    for i, cls_name in enumerate(CLASS_NAMES):

        gt = clean_label(row[cls_name])
        pred = preds[i]

        result = get_result(pred, gt)

        if result == "IGNORED":
            continue

        cam = gradcam.generate(image_tensor, i)
        overlay = overlay_heatmap(image_color, cam)
        final = combine_images(image_color, overlay)

        text = f"{cls_name} | {result} | P:{probs[i]:.2f} | GT:{gt}"

        cv2.putText(
            final,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_name = f"{base}_{cls_name}.jpg"
        save_path = os.path.join(SAVE_DIR, save_name)

        cv2.imwrite(save_path, final)

print("DONE")