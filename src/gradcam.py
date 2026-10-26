import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN architectures."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample CAM to input image resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create RGB heatmap overlay on top of MRI slice."""
    img_rgb = np.array(img_pil.convert("RGB"))
    h, w, _ = img_rgb.shape

    # Resize CAM to match image dimensions if needed
    if cam.shape != (h, w):
        cam = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1.0 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def compute_cam_mask_hit(cam: np.ndarray, mask: np.ndarray) -> bool:
    """Check if the peak Grad-CAM activation pixel falls inside the ground-truth tumor mask."""
    if mask is None or mask.sum() == 0:
        return False
        
    if cam.shape != mask.shape:
        cam = cv2.resize(cam, (mask.shape[1], mask.shape[0]))

    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(mask[peak_y, peak_x] > 0)
