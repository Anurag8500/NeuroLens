import os
import sys
import uuid
import json
import csv
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
sys.path.append(os.path.dirname(__file__))

from src.models import create_model
from src.gradcam import GradCAM, overlay_cam_on_image
from src.data import IDX_TO_CLASS

app = FastAPI(title="NeuroLens — Brain Tumour Classification & XAI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "outputs", "uploads")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "processed", "images")
MASKS_DIR = os.path.join(BASE_DIR, "data", "processed", "masks")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Cache loaded models in memory for instant inference
MODEL_CACHE = {}

def get_loaded_model(arch: str = "resnet50"):
    ckpt_name = "resnet50_patient_best.pt" if arch == "resnet50" else "simplecnn_patient_best.pt"
    ckpt_path = os.path.join(BASE_DIR, "outputs", "models", ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    if arch not in MODEL_CACHE:
        print(f"Loading {arch.upper()} weights from {ckpt_path} onto {DEVICE}...")
        model = create_model(arch, num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        MODEL_CACHE[arch] = model

    return MODEL_CACHE[arch]

EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLINICAL_PROFILES = {
    "glioma": {
        "title": "Glioma (Intra-axial Glial Neoplasm)",
        "prevalence": "Most common primary intra-axial malignant brain tumor in adults (~75-80% of malignant cases).",
        "features": "Characterized by invasive, infiltrative borders, heterogeneity with hyperintense signal on T2/FLAIR, and non-uniform ring-enhancement on T1-CE due to blood-brain barrier disruption.",
        "management": "Maximal safe surgical resection followed by radiation therapy and Temozolomide (Stupp Protocol). Molecular testing for IDH1/2 mutation and 1p/19q codeletion is critical for grading (WHO Grade 2-4)."
    },
    "meningioma": {
        "title": "Meningioma (Extra-axial Arachnoid Neoplasm)",
        "prevalence": "Most prevalent overall primary brain tumor (~35-40%), predominant in adults aged 40-70.",
        "features": "Originates from arachnoid cap cells of the meninges. Characterized by well-demarcated extra-axial location, broad dural attachment with a classic 'dural tail sign' and intense homogenous contrast enhancement on T1-CE.",
        "management": "Majority are benign (WHO Grade 1). Asymptomatic lesions are monitored via serial neuroimaging. Symptomatic or progressive lesions undergo Simpson-graded microsurgical resection or stereotactic radiosurgery (SRS)."
    },
    "pituitary": {
        "title": "Pituitary Neuroendocrine Tumor (PitNET / Adenoma)",
        "prevalence": "Accounts for 10-15% of intracranial neoplasms, localized to the sella turcica.",
        "features": "Originates from the anterior pituitary gland. Microadenomas (<10mm) or macroadenomas (>10mm) frequently present with suprasellar extension, causing compression of the optic chiasm (bitemporal hemianopsia). Displays delayed enhancement relative to normal gland parenchyma on dynamic T1-CE.",
        "management": "Endocrine hormone workup (prolactin, GH, ACTH). Prolactinomas respond favorably to dopamine agonists (cabergoline). Compressive macroadenomas are primarily resected via endoscopic transsphenoidal surgery (TSS)."
    }
}

# Mount static asset folders
app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
app.mount("/slices", StaticFiles(directory=IMAGES_DIR), name="slices")
if os.path.exists(MASKS_DIR):
    app.mount("/masks", StaticFiles(directory=MASKS_DIR), name="masks")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")



@app.get("/api/status")
def get_status():
    models_available = []
    models_dir = os.path.join(BASE_DIR, "outputs", "models")
    if os.path.exists(models_dir):
        models_available = [f for f in os.listdir(models_dir) if f.endswith(".pt")]

    return {
        "status": "online",
        "project": "NeuroLens",
        "description": "Brain Tumour Classification Using Deep Learning and MRI Analysis",
        "split_protocol": "Strict Patient-Disjoint Partitioning (Zero Patient Leakage)",
        "device": str(DEVICE),
        "dataset": {
            "total_slices": 3064,
            "unique_patients": 233,
            "classes": {"glioma": 1426, "pituitary": 930, "meningioma": 708},
            "split": {"train_slices": 2188, "val_slices": 436, "test_slices": 440}
        },
        "available_models": models_available
    }


@app.get("/api/samples")
def get_samples():
    samples = [
        {"slice_id": "0946", "label": 2, "class_name": "pituitary", "pid": "107743", "title": "Pituitary Macroadenoma — Patient #107743 (Slice 0946)"},
        {"slice_id": "0837", "label": 1, "class_name": "glioma", "pid": "100412", "title": "Glioma Infiltration — Patient #100412 (Slice 0837)"},
        {"slice_id": "0001", "label": 0, "class_name": "meningioma", "pid": "100360", "title": "Meningioma Dural Lesion — Patient #100360 (Slice 0001)"},
        {"slice_id": "0722", "label": 1, "class_name": "glioma", "pid": "102075", "title": "Glioma Contrast Lesion — Patient #102075 (Slice 0722)"},
        {"slice_id": "0006", "label": 0, "class_name": "meningioma", "pid": "100572", "title": "Meningioma Parasagittal — Patient #100572 (Slice 0006)"}
    ]
    return samples


@app.get("/api/metrics")
def get_metrics():
    csv_path = os.path.join(BASE_DIR, "outputs", "results_summary.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Results summary CSV not found")

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    return records


@app.post("/api/predict")
async def predict_mri(
    slice_id: str = Form(None),
    model_arch: str = Form("resnet50"),
    mri_file: UploadFile = File(None)
):
    image_path = None
    is_uploaded = False

    if mri_file is not None and mri_file.filename:
        file_ext = os.path.splitext(mri_file.filename)[1] or ".png"
        saved_name = f"upload_{uuid.uuid4().hex[:8]}{file_ext}"
        image_path = os.path.join(UPLOADS_DIR, saved_name)
        contents = await mri_file.read()
        with open(image_path, "wb") as f:
            f.write(contents)
        is_uploaded = True
    elif slice_id:
        sid = int(slice_id)
        fname = f"{sid:04d}.png"
        image_path = os.path.join(IMAGES_DIR, fname)

    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Invalid MRI slice input. Please select a sample or upload a valid image.")

    try:
        model = get_loaded_model(model_arch)
        raw_pil = Image.open(image_path).convert("RGB")
        input_tensor = EVAL_TRANSFORM(raw_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_class = IDX_TO_CLASS[pred_idx]

        # Generate Grad-CAM activation map
        target_layer = model.layer4[-1] if hasattr(model, "layer4") else list(model.children())[0][-1]
        cam_gen = GradCAM(model, target_layer)
        cam = cam_gen.generate(input_tensor, target_class=pred_idx)
        overlay = overlay_cam_on_image(raw_pil, cam, alpha=0.55)

        cam_fname = f"cam_{uuid.uuid4().hex[:8]}.png"
        cam_save_path = os.path.join(FIGURES_DIR, cam_fname)
        Image.fromarray(overlay).save(cam_save_path)

        src_url = f"/uploads/{os.path.basename(image_path)}" if is_uploaded else f"/slices/{os.path.basename(image_path)}"
        cam_url = f"/figures/{cam_fname}"

        mask_url = None
        if slice_id:
            sid = int(slice_id)
            mask_fname = f"{sid:04d}.png"
            if os.path.exists(os.path.join(MASKS_DIR, mask_fname)):
                mask_url = f"/masks/{mask_fname}"

        profile = CLINICAL_PROFILES.get(pred_class, {})

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                cname: float(probs[idx]) for idx, cname in IDX_TO_CLASS.items()
            },
            "source_image_url": src_url,
            "cam_url": cam_url,
            "mask_url": mask_url,
            "clinical_profile": profile,
            "model_used": model_arch.upper(),
            "split_guarantee": "Patient-Disjoint Partition (Zero Data Leakage)"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Mount the public static files (index.html, style.css, app.js)
if os.path.exists(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 65)
    print(f"  NeuroLens Clinical MRI Classification & XAI Studio")
    print(f"  Server URL: http://localhost:{port}")
    print(f"  Interactive API Docs: http://localhost:{port}/docs")
    print("=" * 65 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
