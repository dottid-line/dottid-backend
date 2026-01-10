raise RuntimeError("DEPLOY_PROOF: model_loader is updated and being imported")
# model_loader.py
# Loads Validator, Room Type, and Condition ConvNeXt models with FIXED class order

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# NEW: force model download before torch.load
from download_models import ensure_models_local

# ============================================================
# FIXED CLASS ORDERS (must match TRAINING ORDER exactly)
# ============================================================

# Room-type classes (unchanged)
ROOM_CLASSES = ["bathroom", "bedroom", "dining_room", "kitchen", "living_room"]

# CONDITION CLASSES – RESTORED TO TRAINING ORDER
# This order must match how the model was trained (ImageFolder alphabetical):
#   0 -> fullrehab
#   1 -> fullyupdated
#   2 -> needsrehab
#   3 -> solidcondition
CONDITION_CLASSES = ["fullrehab", "fullyupdated", "needsrehab", "solidcondition"]

# Validator classes (unchanged)
VALIDATOR_CLASSES = ["invalid", "valid"]   # update only if your training order differs

# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================================================
# BASE MODEL LOADER (ConvNeXt Tiny)
# ============================================================

def load_convnext_model(model_path, num_classes):
    """
    Loads a ConvNeXt Tiny model and replaces the classifier layer.
    """

    # --- NEW: FORCE DOWNLOAD BEFORE LOADING ---
    model_path_p = Path(model_path)
    ensure_models_local(model_path_p.parent)

    if not model_path_p.exists():
        raise FileNotFoundError(
            f"Model file still missing after ensure_models_local(): {model_path}"
        )
    # -----------------------------------------

    model = models.convnext_tiny(weights=None)  # No pretrained weights
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    return model

# ============================================================
# VALIDATOR MODEL LOADER
# ============================================================

def load_validator_model(model_path):
    """
    Loads the validator model (valid vs invalid image detector).
    """
    return load_convnext_model(model_path, len(VALIDATOR_CLASSES))

# ============================================================
# PUBLIC FUNCTION: load ALL 3 models
# ============================================================

def load_models(room_model_path, condition_model_path, validator_model_path):
    """
    Returns:
      - room_model
      - condition_model
      - validator_model
    """

    # Room-type model
    room_model = load_convnext_model(room_model_path, len(ROOM_CLASSES))
    room_model.classes = ROOM_CLASSES

    # Condition model – uses CONDITION_CLASSES in the EXACT training order above
    condition_model = load_convnext_model(condition_model_path, len(CONDITION_CLASSES))
    condition_model.classes = CONDITION_CLASSES

    # Validator model
    validator_model = load_validator_model(validator_model_path)
    validator_model.classes = VALIDATOR_CLASSES

    return room_model, condition_model, validator_model

# ============================================================
# PUBLIC FUNCTION: preprocess one image
# ============================================================

def load_image_as_tensor(path):
    """
    Loads an image and converts it into a 1x3x224x224 tensor.
    """
    img = Image.open(path).convert("RGB")
    tensor = IMAGE_TF(img).unsqueeze(0)  # Add batch dimension
    return tensor

# ============================================================
# PUBLIC FUNCTION: get prediction + confidence
# ============================================================

def predict(model, tensor, classes):
    """
    Returns predicted_class, confidence_score
    """

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    conf, idx = torch.max(probs, dim=0)

    return classes[idx], float(conf.item())
