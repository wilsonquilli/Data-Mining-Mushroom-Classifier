import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "image_dataset")
DEFAULT_EDIBILITY_MAP = os.path.join(BASE_DIR, "edibility_map.json")
MODEL_PATH = os.path.join(MODELS_DIR, "image_classifier.pt")
METADATA_PATH = os.path.join(MODELS_DIR, "image_model_metadata.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EDIBILITY_RISK_ORDER = {
    "unknown": 0,
    "edible": 1,
    "conditionally_edible": 2,
    "conditionally edible": 2,
    "poisonous": 3,
    "deadly": 4,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SampleDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_edibility_map(path, classes):
    if not path or not os.path.exists(path):
        return {class_name: "unknown" for class_name in classes}

    with open(path) as f:
        mapping = json.load(f)

    return {
        class_name: mapping.get(class_name, mapping.get(class_name.replace("_", " "), "unknown"))
        for class_name in classes
    }


def normalize_edibility(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def pick_riskier_edibility(current, candidate):
    current = normalize_edibility(current or "unknown")
    candidate = normalize_edibility(candidate or "unknown")
    if EDIBILITY_RISK_ORDER.get(candidate, 0) > EDIBILITY_RISK_ORDER.get(current, 0):
        return candidate
    return current


def discover_image_dataset(data_dir):
    """
    Supports both layouts:
      image_dataset/<species>/*.jpg
      mushroom_dataset/<edibility>/<species>/*.jpg
    """
    samples_by_class = {}
    discovered_edibility = {}

    for current_dir, _, filenames in os.walk(data_dir):
        image_files = [
            os.path.join(current_dir, filename)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not image_files:
            continue

        species = os.path.basename(current_dir)
        parent = os.path.basename(os.path.dirname(current_dir))
        edibility = parent if parent in EDIBILITY_RISK_ORDER else "unknown"

        samples_by_class.setdefault(species, []).extend(image_files)
        discovered_edibility[species] = pick_riskier_edibility(
            discovered_edibility.get(species, "unknown"),
            edibility,
        )

    class_names = sorted(samples_by_class)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    samples = [
        (path, class_to_idx[class_name])
        for class_name in class_names
        for path in sorted(samples_by_class[class_name])
    ]

    return samples, class_names, class_to_idx, discovered_edibility


def split_samples(samples, seed):
    labels = [label for _, label in samples]
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [label for _, label in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_samples, val_samples, test_samples


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(18),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def build_model(num_classes, freeze_backbone=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, num_classes),
    )
    return model


def run_epoch(model, dataloader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)

    return {
        "loss": total_loss / max(1, seen),
        "accuracy": correct / max(1, seen),
    }


def evaluate(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_predictions = []
    top3_correct = 0
    seen = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            top_k = min(3, logits.shape[1])
            top3 = logits.topk(top_k, dim=1).indices

            top3_correct += top3.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            seen += images.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return {
        "top1_accuracy": sum(
            int(pred == label) for pred, label in zip(all_predictions, all_labels)
        ) / max(1, len(all_labels)),
        "top3_accuracy": top3_correct / max(1, seen),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def train(args):
    set_seed(args.seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Image dataset not found: {args.data_dir}\n"
            "Expected folder layout: image_dataset/<species_name>/*.jpg"
        )

    samples, class_names, class_to_idx, discovered_edibility = discover_image_dataset(args.data_dir)
    if len(class_names) < 2:
        raise ValueError("Need at least two species folders to train an image classifier.")

    class_counts = Counter(label for _, label in samples)
    too_small = [
        class_names[label]
        for label, count in class_counts.items()
        if count < 4
    ]
    if too_small:
        raise ValueError(
            "Each species needs at least 4 images for stratified train/validation/test "
            f"splits. Too few images for: {too_small}"
        )

    train_samples, val_samples, test_samples = split_samples(samples, args.seed)
    edibility_map = load_edibility_map(args.edibility_map, class_names)
    edibility_map = {
        class_name: pick_riskier_edibility(
            discovered_edibility.get(class_name, "unknown"),
            edibility_map.get(class_name, "unknown"),
        )
        for class_name in class_names
    }

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = SampleDataset(train_samples, class_to_idx, train_transform)
    val_dataset = SampleDataset(val_samples, class_to_idx, eval_transform)
    test_dataset = SampleDataset(test_samples, class_to_idx, eval_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), freeze_backbone=True).to(device)

    class_counts = Counter(label for _, label in train_samples)
    class_weights = [
        len(train_samples) / max(1, len(class_names) * class_counts[idx])
        for idx in range(len(class_names))
    ]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_accuracy = 0.0
    best_state = None
    patience_remaining = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate * 0.1,
                weight_decay=args.weight_decay,
            )

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        print(
            f"Epoch {epoch:02d}: "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_remaining = args.patience
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, class_names)

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    checkpoint = {
        "model_name": "efficientnet_b0",
        "image_size": args.image_size,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "edibility_map": edibility_map,
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, MODEL_PATH)

    metadata = {
        "model_name": "efficientnet_b0",
        "dataset_dir": os.path.abspath(args.data_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_images": len(samples),
        "splits": {
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
        },
        "best_validation_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "history": history,
        "edibility_map": edibility_map,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved image model to {MODEL_PATH}")
    print(f"Saved image metadata to {METADATA_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet mushroom image classifier."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--edibility-map", default=DEFAULT_EDIBILITY_MAP)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
