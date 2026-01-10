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

# Render-safe writable default:
# - If MODEL_DIR is set, use it
# - Else use /tmp/.backend-models
MODEL_DIR_NAME = os.environ.get("MODEL_DIR", "/tmp/.backend-models").strip() or "/tmp/.backend-models"
MODEL_DIR = Path(MODEL_DIR_NAME)

ROOM_MODEL_PATH = str(MODEL_DIR / "FINAL MVP ROOM TYPE MODEL.pth")
CONDITION_MODEL_PATH = str(MODEL_DIR / "FINAL MVP CONDITION MODEL.pth")
VALIDATOR_MODEL_PATH = str(MODEL_DIR / "FINAL MVP VALIDATOR MODEL.pth")

# ===================================================================
# LOAD MODELS ONCE — LAZY INIT
# ===================================================================

_room_model = None
_condition_model = None
_validator_model = None


def _ensure_models_or_raise() -> None:
    """
    Ensures model files exist locally. If they do not exist after ensure_models_local(),
    raise a clear error so jobs fail with the REAL reason (download didn't happen),
    instead of a confusing torch file-not-found later.
    """
    # Ensure models exist locally (download from S3 on Render if missing)
    ensure_models_local(MODEL_DIR)

    # Hard check: files must exist after ensure_models_local()
    missing = []
    if not Path(ROOM_MODEL_PATH).exists():
        missing.append(Path(ROOM_MODEL_PATH).name)
    if not Path(CONDITION_MODEL_PATH).exists():
        missing.append(Path(CONDITION_MODEL_PATH).name)
    if not Path(VALIDATOR_MODEL_PATH).exists():
        missing.append(Path(VALIDATOR_MODEL_PATH).name)

    if missing:
        # Include dir listing to make debugging on Render obvious
        try:
            present = sorted([p.name for p in MODEL_DIR.glob("*") if p.is_file()])
        except Exception:
            present = []
        raise FileNotFoundError(
            f"Model files missing in '{str(MODEL_DIR)}': {missing}. "
            f"Present files: {present}. "
            f"ensure_models_local() did not download/copy the required .pth files."
        )


def _get_models():
    global _room_model, _condition_model, _validator_model

    # Ensure models exist locally (download from S3 on Render if missing)
    _ensure_models_or_raise()

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
        out = classify_image(img_path, device)
        results.append(out)
    return results
