import json
import os
from functools import lru_cache

from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "image_classifier.pt")
IMAGE_METADATA_PATH = os.path.join(MODELS_DIR, "image_model_metadata.json")

DEFAULT_IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.6
RISK_CONFIDENCE_THRESHOLD = 0.6


def _torch_imports():
    try:
        import torch
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Image prediction requires torch, torchvision, and pillow. "
            "Install backend requirements first."
        ) from exc

    return torch, models, transforms


def _build_model(num_classes):
    torch, models, _ = _torch_imports()
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def image_model_status():
    metadata = None
    if os.path.exists(IMAGE_METADATA_PATH):
        with open(IMAGE_METADATA_PATH) as f:
            raw_metadata = json.load(f)
            metadata = {
                "model_name": raw_metadata.get("model_name"),
                "num_classes": raw_metadata.get("num_classes"),
                "num_images": raw_metadata.get("num_images"),
                "image_size": raw_metadata.get("image_size"),
                "best_validation_accuracy": raw_metadata.get("best_validation_accuracy"),
                "test_metrics": raw_metadata.get("test_metrics"),
                "species_metrics": raw_metadata.get("species_metrics"),
                "edibility_metrics": raw_metadata.get("edibility_metrics"),
                "risk_metrics": raw_metadata.get("risk_metrics"),
                "training_strategy": raw_metadata.get("training_strategy"),
            }

    return {
        "available": os.path.exists(IMAGE_MODEL_PATH),
        "model_path": IMAGE_MODEL_PATH,
        "metadata": metadata,
    }


@lru_cache(maxsize=1)
def load_image_model():
    if not os.path.exists(IMAGE_MODEL_PATH):
        return None

    torch, _, transforms = _torch_imports()
    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location="cpu", weights_only=False)

    idx_to_class = {
        int(idx): label for idx, label in checkpoint["idx_to_class"].items()
    }
    model = _build_model(len(idx_to_class))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    edibility_head = None
    idx_to_edibility = {
        int(idx): label
        for idx, label in checkpoint.get("idx_to_edibility", {}).items()
    }
    if checkpoint.get("edibility_head_state_dict") and idx_to_edibility:
        in_features = model.classifier[1].in_features
        torch, _, _ = _torch_imports()
        edibility_head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.35),
            torch.nn.Linear(in_features, len(idx_to_edibility)),
        )
        edibility_head.load_state_dict(checkpoint["edibility_head_state_dict"])
        edibility_head.eval()

    risk_head = None
    idx_to_risk = {
        int(idx): label
        for idx, label in checkpoint.get("idx_to_risk", {}).items()
    }
    if checkpoint.get("risk_head_state_dict") and idx_to_risk:
        in_features = model.classifier[1].in_features
        torch, _, _ = _torch_imports()
        risk_head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.35),
            torch.nn.Linear(in_features, len(idx_to_risk)),
        )
        risk_head.load_state_dict(checkpoint["risk_head_state_dict"])
        risk_head.eval()

    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return {
        "model": model,
        "transform": transform,
        "idx_to_class": idx_to_class,
        "edibility_map": checkpoint.get("edibility_map", {}),
        "edibility_head": edibility_head,
        "idx_to_edibility": idx_to_edibility,
        "risk_head": risk_head,
        "idx_to_risk": idx_to_risk,
        "risk_edible_threshold": float(checkpoint.get("risk_edible_threshold", 0.75)),
        "image_size": image_size,
    }


def _display_species(label):
    return label.replace("_", " ").replace("-", " ").strip()


def predict_image(file_storage, top_k=3):
    loaded = load_image_model()
    if loaded is None:
        raise FileNotFoundError(
            "Image model has not been trained yet. Run "
            "`python train_image_model.py --data-dir <image_dataset>` first."
        )

    torch, _, _ = _torch_imports()

    image = Image.open(file_storage.stream).convert("RGB")
    tensor = loaded["transform"](image).unsqueeze(0)

    with torch.no_grad():
        features = loaded["model"].features(tensor)
        features = loaded["model"].avgpool(features)
        features = torch.flatten(features, 1)
        logits = loaded["model"].classifier(features)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_count = min(max(1, int(top_k)), len(loaded["idx_to_class"]))
    scores, indices = torch.topk(probabilities, top_count)

    predictions = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        species_key = loaded["idx_to_class"][int(idx)]
        predictions.append({
            "species_key": species_key,
            "species": _display_species(species_key),
            "confidence": round(float(score), 4),
            "edibility": loaded["edibility_map"].get(species_key, "unknown"),
        })

    best = predictions[0]
    low_confidence = best["confidence"] < LOW_CONFIDENCE_THRESHOLD
    edibility = best["edibility"]
    edibility_confidence = best["confidence"]
    edibility_source = "species_mapping"

    if loaded["edibility_head"] is not None:
        with torch.no_grad():
            edibility_logits = loaded["edibility_head"](features)
            edibility_probabilities = torch.softmax(edibility_logits, dim=1)[0]
            edibility_score, edibility_idx = torch.max(edibility_probabilities, dim=0)
        edibility = loaded["idx_to_edibility"][int(edibility_idx)]
        edibility_confidence = round(float(edibility_score), 4)
        edibility_source = "direct_image_head"

    if low_confidence and edibility_confidence < LOW_CONFIDENCE_THRESHOLD:
        edibility = "unknown-risk"

    risk_label = "uncertain"
    risk_confidence = 0.0
    risk_probabilities = {}
    risk_subtype = None

    if loaded["risk_head"] is not None:
        with torch.no_grad():
            risk_logits = loaded["risk_head"](features)
            risk_scores = torch.softmax(risk_logits, dim=1)[0]

        risk_probabilities = {
            loaded["idx_to_risk"][idx]: round(float(score), 4)
            for idx, score in enumerate(risk_scores.tolist())
        }
        edible_probability = risk_probabilities.get("edible", 0.0)
        avoid_probability = risk_probabilities.get("avoid", 0.0)

        if edible_probability >= loaded["risk_edible_threshold"]:
            risk_label = "edible"
            risk_confidence = edible_probability
        elif avoid_probability >= RISK_CONFIDENCE_THRESHOLD:
            risk_label = "avoid"
            risk_confidence = avoid_probability
        else:
            risk_label = "uncertain"
            risk_confidence = max(edible_probability, avoid_probability)
    else:
        if edibility == "edible" and not low_confidence:
            risk_label = "edible"
            risk_confidence = edibility_confidence
        elif edibility in ("poisonous", "deadly", "conditionally_edible"):
            risk_label = "avoid"
            risk_confidence = edibility_confidence

    if risk_label == "avoid" and not low_confidence:
        risk_subtype = best["edibility"]

    return {
        "risk_label": risk_label,
        "risk_confidence": round(float(risk_confidence), 4),
        "risk_probabilities": risk_probabilities,
        "risk_subtype": risk_subtype,
        "species_suggestions": predictions,
        "species_key": best["species_key"],
        "species": best["species"],
        "species_confidence": best["confidence"],
        "edibility": edibility,
        "edibility_confidence": edibility_confidence,
        "edibility_source": edibility_source,
        "low_confidence": low_confidence,
        "top_predictions": predictions,
        "warning": (
            "Do not consume wild mushrooms based only on this application. "
            "Consult a qualified expert before eating any foraged mushroom."
        ),
    }
