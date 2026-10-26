import os
import sys
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from src.models import create_model
from src.gradcam import GradCAM, overlay_cam_on_image
from src.data import IDX_TO_CLASS


def main():
    parser = argparse.ArgumentParser(description="NeuroLens — Single MRI Slice Prediction CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to 2D brain MRI slice (PNG/JPG)")
    parser.add_argument("--model", type=str, default="outputs/models/resnet50_patient_best.pt", help="Path to trained model checkpoint")
    parser.add_argument("--model_arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "simplecnn"], help="Architecture name")
    parser.add_argument("--save_cam", type=str, default=None, help="Optional output path to save Grad-CAM overlay")
    parser.add_argument("--json", action="store_true", help="Output JSON results to stdout")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return

    if not os.path.exists(args.model):
        print(f"[ERROR] Model checkpoint not found: {args.model}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.json:
        print(f"\n[NeuroLens] Loading {args.model_arch.upper()} weights from {args.model}...")

    model = create_model(args.model_arch, num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()

    raw_pil = Image.open(args.image).convert("RGB")

    eval_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = eval_tf(raw_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    pred_class = IDX_TO_CLASS[pred_idx]

    cam_saved_path = None
    if args.save_cam:
        target_layer = model.layer4[-1] if hasattr(model, "layer4") else (model.conv_head if hasattr(model, "conv_head") else list(model.children())[0][-1])
        cam_gen = GradCAM(model, target_layer)
        cam = cam_gen.generate(input_tensor, target_class=pred_idx)
        overlay = overlay_cam_on_image(raw_pil, cam, alpha=0.55)

        os.makedirs(os.path.dirname(args.save_cam) if os.path.dirname(args.save_cam) else ".", exist_ok=True)
        Image.fromarray(overlay).save(args.save_cam)
        cam_saved_path = args.save_cam

    if args.json:
        import json
        res = {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                cname: float(probs[idx]) for idx, cname in IDX_TO_CLASS.items()
            },
            "cam_path": cam_saved_path
        }
        print(json.dumps(res))
    else:
        print("\n" + "=" * 55)
        print("           NEUROLENS PREDICTION RESULTS")
        print("=" * 55)
        print(f"Predicted Diagnosis:  >>> {pred_class.upper()} <<<")
        print(f"Prediction Confidence: {probs[pred_idx] * 100:.2f}%\n")
        print("Class Confidence Distribution:")
        for idx, cname in IDX_TO_CLASS.items():
            bar = "#" * int(probs[idx] * 25)
            print(f"  {cname.capitalize():11s} : {probs[idx] * 100:6.2f}%  {bar}")
        print("=" * 55)
        print("DISCLAIMER: NeuroLens is a research & educational project.")
        print("This is NOT a certified clinical diagnostic tool.")
        print("=" * 55 + "\n")
        if cam_saved_path:
            print(f"Grad-CAM overlay saved to: {cam_saved_path}")


if __name__ == "__main__":
    main()
