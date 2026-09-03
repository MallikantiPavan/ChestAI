import torch

ckpt = torch.load(
    r"D:\densenet_eca\final_label_224_densenet\best1.ckpt",
    map_location="cpu",
    weights_only=False   # 👈 IMPORTANT
)

# ckpt = torch.load(
#     r"D:\densenet_eca\final_label_eca_adam_224\best1.ckpt",
#     map_location="cpu",
#     weights_only=False   # 👈 IMPORTANT
# )
for k in ckpt['state_dict'].keys():
    if "attention_map" in k:
        print(k)

for k, v in ckpt['state_dict'].items():
    if "attention_map" in k and v.dtype.is_floating_point:
        print(k, v.abs().mean().item())

groups = {
    "ECA": [],
    "CAM": [],
    "SAM": [],
    "FPA": []
}

for k, v in ckpt['state_dict'].items():
    if not v.dtype.is_floating_point:
        continue

    if "eca_attention" in k:
        groups["ECA"].append(v.abs().mean().item())

    elif "channel_attention" in k:
        groups["CAM"].append(v.abs().mean().item())

    elif "spatial_attention" in k:
        groups["SAM"].append(v.abs().mean().item())

    elif "pyramid_attention" in k:
        groups["FPA"].append(v.abs().mean().item())

for name, values in groups.items():
    if len(values) > 0:
        print(name, sum(values)/len(values))