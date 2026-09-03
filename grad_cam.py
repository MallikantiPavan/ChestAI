import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        target_layer = self.model.backbone.features[-1]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):

        logits, _ = self.model(input_tensor)

        logits = torch.stack(logits)  

        self.model.zero_grad()

        score = logits[class_idx]

        if score.dim() > 0:
            score = score.sum()

        score.backward(retain_graph=True)

        grads = self.gradients[0].cpu().data.numpy()     
        acts = self.activations[0].cpu().data.numpy()    

        weights = np.mean(grads, axis=(1, 2))  

        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-8)

        return cam


def overlay_heatmap(image, cam, alpha=0.4):
    h, w = cam.shape

    image = cv2.resize(image, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay

def combine_images(original, cam_overlay):
    h, w = cam_overlay.shape[:2]

    original = cv2.resize(original, (w, h))

    combined = np.hstack((original, cam_overlay))

    return combined

def preprocess_image(image_path, cfg):
    from data.utils import transform

    image_gray = cv2.imread(image_path, 0)
    assert image_gray is not None, f"Invalid image: {image_path}"

    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    image = transform(image_gray, cfg)
    image = torch.from_numpy(image).unsqueeze(0)

    return image, image_color


def run_gradcam(model, device, image_path, cfg, class_idx, save_path):
    model.eval()

    for p in model.parameters():
        p.requires_grad = True

    gradcam = GradCAM(model)

    image_tensor, image_color = preprocess_image(image_path, cfg)

    image_tensor = image_tensor.to(device)

    cam = gradcam.generate(image_tensor, class_idx)

    overlay = overlay_heatmap(image_color, cam)

    final = combine_images(image_color, overlay)

    cv2.imwrite(save_path, final)