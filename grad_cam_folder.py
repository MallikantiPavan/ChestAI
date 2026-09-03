import os
import cv2
import torch
import json
import numpy as np
from easydict import EasyDict as edict

from model.classifier import Classifier
from grad_cam import GradCAM, overlay_heatmap, combine_images, preprocess_image

CFG_PATH = r"D:\densenet_eca\final_label_eca_adam_224\classification\config\example.json"
CKPT_PATH = r"D:\densenet_eca\final_label_eca_adam_224\best1.ckpt"
FOLDER_PATH = r"D:\densenet_eca\Dataset_original_test_val\Testing_images_wrt_disease\Atelectasis\TP"
# FOLDER_PATH = r"D:\densenet_eca\Dataset_original_test_val\Testing_images_wrt_disease\Atelectasis\TP"
# FOLDER_PATH = r"D:\densenet_eca\Dataset_original_test_val\Testing_images_wrt_disease\Cardimelogy\TP"
# FOLDER_PATH = r"D:\densenet_eca\Dataset_original_test_val\Testing_images_wrt_disease\Conslidation\TP"
# FOLDER_PATH = r"D:\densenet_eca\Dataset_original_test_val\Testing_images_wrt_disease\Edema\TP"
# FOLDER_PATH = r"D:\densenet_eca\Dataset_original_test_val\Testing_images_wrt_disease\Pleural Effusion\TP"

SAVE_DIR = r"D:\densenet_eca\dense_eca\grad_Cam_images\Atelectesis"
# SAVE_DIR = r"D:\densenet_eca\dense_eca\grad_Cam_images\Cardimology"

# SAVE_DIR = r"D:\densenet_eca\dense_eca\grad_Cam_images\Consolidation"
# SAVE_DIR = r"D:\densenet_eca\dense_eca\grad_Cam_images\Edema"
# SAVE_DIR = r"D:\densenet_eca\dense_eca\grad_Cam_images\pleural_effusion"



THRESHOLD = 0.5

CLASS_NAMES = [
    "Atelectasis",
    # "Cardiomegaly",
    # "Consolidation",
    # "Edema",
    # "Pleural Effusion"
]

os.makedirs(SAVE_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = edict(json.load(f))

model = Classifier(cfg)

ckpt = torch.load(CKPT_PATH, map_location="cuda:0", weights_only=False)
state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda:0")
model = model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad = True

gradcam = GradCAM(model)

image_files = [
    f for f in os.listdir(FOLDER_PATH)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

for img_name in image_files:

    image_path = os.path.join(FOLDER_PATH, img_name)

    print("Processing:", img_name)

    image_tensor, image_color = preprocess_image(image_path, cfg)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits, _ = model(image_tensor)

    logits = torch.stack(logits)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs >= THRESHOLD).astype(int)

    base_name = os.path.splitext(img_name)[0]

    for i, cls_name in enumerate(CLASS_NAMES):
        cam = gradcam.generate(image_tensor, i)

        overlay = overlay_heatmap(image_color, cam)
        final = combine_images(image_color, overlay)

        label = f"{cls_name} | P:{probs[i]:.2f} | Pred:{preds[i]}"

        cv2.putText(
            final,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        save_path = os.path.join(
            SAVE_DIR,
            f"{base_name}_{cls_name}.jpg"
        )

        cv2.imwrite(save_path, final)

print("DONE — all images processed")