# inference_engine.py

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model_loader import load_models
from download_models import ensure_models_local

# ===================================================================
# MODEL PATHS
# ===================================================================

BASE_DIR = Path(__file__).resolve().parent

# Force ".backend-models" (Render + your desired structure)
MODEL_DIR_NAME = os.environ.get("MODEL_DIR", ".backend-models").strip() or ".backend-models"
MODEL_DIR = (BASE_DIR / MODEL_DIR_NAME)

ROOM_MODEL_PATH = str(MODEL_DIR / "FINAL MVP ROOM TYPE MODEL.pth")
CONDITION_MODEL_PATH = str(MODEL_DIR / "FINAL MVP CONDITION MODEL.pth")
VALIDATOR_MODEL_PATH = str(MODEL_DIR / "FINAL MVP VALIDATOR MODEL.pth")

# ===================================================================
# LOAD MODELS ONCE — LAZY INIT
# ===================================================================

_room_model = None
_condition_model = None
_validator_model = None

def _assert_exists(p: str):
    if not os.path.exists(p):
        raise RuntimeError(f"Model file missing on disk: {p}")

def _get_models():
    global _room_model, _condition_model, _validator_model

    # Ensure models exist locally (download from S3 on Render if missing)
    ensure_models_local(MODEL_DIR)

    # Hard assert so we never try torch.load() on missing files
    _assert_exists(ROOM_MODEL_PATH)
    _assert_exists(CONDITION_MODEL_PATH)
    _assert_exists(VALIDATOR_MODEL_PATH)

    if _room_model is None or _condition_model is None or _validator_model is None:
        _room_model, _condition_model, _validator_model = load_models(
            ROOM_MODEL_PATH,
            CONDITION_MODEL_PATH,
            VALIDATOR_MODEL_PATH,
        )
    return _room_model, _condition_model, _validator_model

# ===================================================================
# TRANSFORM
# ===================================================================

img_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# ===================================================================
# VALIDATOR
# ===================================================================

def classify_validity(img_path, device: str = "cpu"):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return "invalid", 0.0

    tensor_img = img_tf(img).unsqueeze(0).to(device)

    room_model, condition_model, validator_model = _get_models()

    with torch.no_grad():
        logits = validator_model(tensor_img)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)

    label = validator_model.classes[idx.item()]
    return label, float(conf.item())

# ===================================================================
# SINGLE IMAGE CLASSIFICATION
# ===================================================================

def classify_image(img_path, device: str = "cpu") -> dict:
    validity, valid_conf = classify_validity(img_path, device)

    if validity == "invalid" or valid_conf < 0.60:
        return {
            "image_path": img_path,
            "valid": False,
            "valid_conf": valid_conf,
            "room_type": "invalid",
            "room_conf": 0.0,
            "condition": "invalid",
            "condition_conf": 0.0,
        }

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return {
            "image_path": img_path,
            "valid": False,
            "valid_conf": valid_conf,
            "room_type": "error",
            "room_conf": 0.0,
            "condition": "error",
            "condition_conf": 0.0,
        }

    tensor_img = img_tf(img).unsqueeze(0).to(device)

    room_model, condition_model, validator_model = _get_models()

    # ROOM TYPE
    with torch.no_grad():
        room_logits = room_model(tensor_img)
        room_probs = F.softmax(room_logits, dim=1)
        room_conf, room_idx = torch.max(room_probs, 1)

    room_label = room_model.classes[room_idx.item()]
    room_conf = round(room_conf.item(), 4)

    # CONDITION
    with torch.no_grad():
        cond_logits = condition_model(tensor_img)
        cond_probs = F.softmax(cond_logits, dim=1)
        cond_conf, cond_idx = torch.max(cond_probs, 1)

    cond_label = condition_model.classes[cond_idx.item()]
    cond_conf = round(cond_conf.item(), 4)

    return {
        "image_path": img_path,
        "valid": True,
        "valid_conf": valid_conf,
        "room_type": room_label,
        "room_conf": room_conf,
        "condition": cond_label,
        "condition_conf": cond_conf,
    }

# ===================================================================
# MULTI-IMAGE ENTRY POINT
# ===================================================================

def classify_images(image_paths, device: str = "cpu") -> list[dict]:
    results: list[dict] = []
    for img_path in image_paths:
        results.append(classify_image(img_path, device))
    return results
