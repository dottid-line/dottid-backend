import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import List, Dict, Any

# ============================================================
# PATHS
# ============================================================
IMAGE_FOLDER = r"C:\FINAL MVP MODEL\Test comp rating"
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", IMAGE_FOLDER)

VALIDATOR_MODEL_PATH = r"C:\FINAL MVP MODEL\FINAL MVP VALIDATOR MODEL.pth"
ROOMTYPE_MODEL_PATH  = r"C:\FINAL MVP MODEL\FINAL MVP ROOM TYPE MODEL.pth"
CONDITION_MODEL_PATH = r"C:\FINAL MVP MODEL\FINAL MVP CONDITION MODEL.pth"

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# TRANSFORMS
# ============================================================
validator_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

room_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

condition_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================================================
# LABEL MAPS
# ============================================================
ROOM_LABELS = ["bathroom", "bedroom", "dining_room", "kitchen", "living_room"]
COND_LABELS = ["fullrehab", "fullyupdated", "needsrehab", "solidcondition"]

COND_SCORE = {
    "fullyupdated": 0,
    "solidcondition": 1,
    "needsrehab": 2,
    "fullrehab": 3
}

# ============================================================
# MODEL LOADING (LAZY + SINGLETON)
# ============================================================
_models = None

def _load_validator():
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    m.load_state_dict(torch.load(VALIDATOR_MODEL_PATH, map_location=device))
    m.eval().to(device)
    return m

def _load_roomtype():
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 5)
    m.load_state_dict(torch.load(ROOMTYPE_MODEL_PATH, map_location=device))
    m.eval().to(device)
    return m

def _load_condition():
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 4)
    m.load_state_dict(torch.load(CONDITION_MODEL_PATH, map_location=device))
    m.eval().to(device)
    return m

def get_models():
    global _models
    if _models is None:
        _models = {
            "validator": _load_validator(),
            "room": _load_roomtype(),
            "condition": _load_condition(),
        }
    return _models

# ============================================================
# BATCH PREDICT
# ============================================================
def _batch_predict(model, tensors: List[torch.Tensor], batch_size: int = 64) -> List[int]:
    preds: List[int] = []
    with torch.inference_mode():
        for i in range(0, len(tensors), batch_size):
            x = torch.stack(tensors[i:i+batch_size]).to(device, non_blocking=True)
            y = model(x)
            preds.extend(torch.argmax(y, 1).detach().cpu().tolist())
    return preds

# ============================================================
# SCORING (IN-MEMORY) + TIMING
# ============================================================
def score_pil_images(images: List[Image.Image]) -> Dict[str, Any]:
    """
    Rules preserved:
      - require >=7 interior images (existing logic)
      - room-type first; skip if no kitchen (existing logic)
      - kitchen score is average across kitchen images (existing logic)
    New rule added:
      - REQUIRE >=3 interior images AND >=1 kitchen image to score
        If not met: skip_reason = "INSUFFICIENT_KITCHEN_IMAGES"
    """
    timing = {}

    if not images:
        return {"final_score": None, "condition": None, "skip_reason": "NO_IMAGES", "timing": timing}

    m = get_models()
    validator_model = m["validator"]
    room_model = m["room"]
    condition_model = m["condition"]

    # -------- VALIDATOR --------
    t0 = time.perf_counter()
    v_tensors = [validator_tf(img) for img in images]
    t1 = time.perf_counter()
    v_preds = _batch_predict(validator_model, v_tensors, batch_size=64)
    t2 = time.perf_counter()

    timing["validator_tf_s"] = round(t1 - t0, 3)
    timing["validator_fw_s"] = round(t2 - t1, 3)

    interior = [img for img, p in zip(images, v_preds) if p == 1]

    if len(interior) < 7:
        return {
            "final_score": None,
            "condition": None,
            "skip_reason": "INSUFFICIENT_INTERIOR_IMAGES",
            "timing": timing,
            "interior_count": len(interior),
        }

    # -------- ROOM TYPE (EARLY KITCHEN RULE) --------
    t3 = time.perf_counter()
    r_tensors = [room_tf(img) for img in interior]
    t4 = time.perf_counter()
    r_preds = _batch_predict(room_model, r_tensors, batch_size=64)
    t5 = time.perf_counter()

    timing["room_tf_s"] = round(t4 - t3, 3)
    timing["room_fw_s"] = round(t5 - t4, 3)

    rooms = [ROOM_LABELS[i] for i in r_preds]

    if not any(r == "kitchen" for r in rooms):
        return {
            "final_score": None,
            "condition": None,
            "skip_reason": "NO_KITCHEN",
            "timing": timing,
            "interior_count": len(interior),
        }

    # -------- NEW RULE HERE --------
    kitchen_count = sum(1 for r in rooms if r == "kitchen")
    if kitchen_count < 1 or len(interior) < 3:
        return {
            "final_score": None,
            "condition": None,
            "skip_reason": "INSUFFICIENT_KITCHEN_IMAGES",
            "timing": timing,
            "interior_count": len(interior),
            "kitchen_count": kitchen_count,
        }

    # -------- CONDITION --------
    t6 = time.perf_counter()
    c_tensors = [condition_tf(img) for img in interior]
    t7 = time.perf_counter()
    c_preds = _batch_predict(condition_model, c_tensors, batch_size=64)
    t8 = time.perf_counter()

    timing["cond_tf_s"] = round(t7 - t6, 3)
    timing["cond_fw_s"] = round(t8 - t7, 3)

    conds = [COND_LABELS[i] for i in c_preds]
    scores = [COND_SCORE[c] for c in conds]

    kitchen_scores = []
    bath_scores = []
    other_scores = []

    for r, s in zip(rooms, scores):
        if r == "kitchen":
            kitchen_scores.append(s)
        elif r == "bathroom":
            bath_scores.append(s)
        else:
            other_scores.append(s)

    if not kitchen_scores:
        return {
            "final_score": None,
            "condition": None,
            "skip_reason": "NO_KITCHEN",
            "timing": timing,
            "interior_count": len(interior),
        }

    # kitchen is average
    kitchen_score = float(np.mean(kitchen_scores))
    bath_mean = float(np.mean(bath_scores)) if bath_scores else None
    other_mean = float(np.mean(other_scores)) if other_scores else None

    if bath_mean is not None and other_mean is not None:
        final_score = (0.60 * kitchen_score) + (0.30 * bath_mean) + (0.10 * other_mean)
    elif bath_mean is None and other_mean is not None:
        final_score = (0.70 * kitchen_score) + (0.30 * other_mean)
    elif bath_mean is not None and other_mean is None:
        final_score = (0.70 * kitchen_score) + (0.30 * bath_mean)
    else:
        final_score = kitchen_score

    final_score = round(float(final_score), 3)

    if final_score <= 0.6:
        label = "FULLY UPDATED / TURNKEY"
    elif final_score <= 1.35:
        label = "SOLID"
    else:
        label = "NEEDS REHAB"

    return {
        "final_score": final_score,
        "condition": label,
        "skip_reason": None,
        "kitchen_score": round(kitchen_score, 3),
        "bath_score": (round(bath_mean, 3) if bath_mean is not None else None),
        "other_score": (round(other_mean, 3) if other_mean is not None else None),
        "interior_count": len(interior),
        "kitchen_count": kitchen_count,
        "timing": timing,
    }

# ============================================================
# FILESYSTEM WRAPPER (BACKWARDS COMPAT)
# ============================================================
def load_images_from_folder(folder: str) -> List[Image.Image]:
    imgs = []
    try:
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                p = os.path.join(folder, f)
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except:
                    pass
    except:
        pass
    return imgs

def score_folder(folder: str) -> Dict[str, Any]:
    imgs = load_images_from_folder(folder)
    return score_pil_images(imgs)

# ============================================================
# CLI ENTRY
# ============================================================
if __name__ == "__main__":
    print("Using device:", device)
    res = score_folder(IMAGE_FOLDER)

    if res.get("skip_reason"):
        print(f"SKIP_REASON: {res['skip_reason']}")
        sys.exit(0)

    print("\n=== CONDITION RESULT ===")
    print(f"Kitchen score: {res.get('kitchen_score')}")
    print(f"Bathroom score: {res.get('bath_score') if res.get('bath_score') is not None else 'N/A'}")
    print(f"Other rooms: {res.get('other_score') if res.get('other_score') is not None else 'N/A'}")
    print(f"\nFINAL SCORE: {res.get('final_score')}")
    print(f"Condition: {res.get('condition')}")
    print(f"\nTiming: {res.get('timing')}")

