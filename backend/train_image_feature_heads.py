import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from train_image_model import (
    METADATA_PATH,
    MODEL_PATH,
    SampleDataset,
    build_transforms,
    discover_image_dataset,
    load_edibility_map,
    pick_riskier_edibility,
    set_seed,
    split_samples,
)

RISK_AVOID = "avoid"
RISK_EDIBLE = "edible"
RISK_LABELS = [RISK_AVOID, RISK_EDIBLE]
TARGET_UNSAFE_RECALL = 0.90


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_feature_extractor():
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()
    model.eval()
    return model


def extract_features(model, dataloader, device):
    all_features = []
    all_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images).cpu()
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def train_head(features, labels, num_classes, epochs, learning_rate, weight_decay):
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_weights = labels.numel() / torch.clamp(num_classes * class_counts, min=1)

    head = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(features.shape[1], num_classes),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            with torch.no_grad():
                accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            print(f"head epoch {epoch:03d}: loss={loss.item():.4f} acc={accuracy:.4f}")

    return head


def evaluate_head(head, features, labels, top_k=3):
    head.eval()
    with torch.no_grad():
        logits = head(features)
        predictions = logits.argmax(dim=1)
        top_count = min(top_k, logits.shape[1])
        top_predictions = logits.topk(top_count, dim=1).indices

    top1 = (predictions == labels).float().mean().item()
    topk = top_predictions.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()
    return {
        "top1_accuracy": top1,
        f"top{top_count}_accuracy": topk,
    }


def build_edibility_labels(samples, idx_to_species, edibility_to_idx, edibility_map):
    labels = []
    for _, species_idx in samples:
        species = idx_to_species[int(species_idx)]
        edibility = edibility_map.get(species, "unknown")
        labels.append(edibility_to_idx[edibility])
    return torch.tensor(labels, dtype=torch.long)


def build_risk_labels(samples, idx_to_species, edibility_map):
    labels = []
    for _, species_idx in samples:
        species = idx_to_species[int(species_idx)]
        edibility = edibility_map.get(species, "unknown")
        labels.append(1 if edibility == RISK_EDIBLE else 0)
    return torch.tensor(labels, dtype=torch.long)


def calibrate_edible_threshold(head, features, labels, target_unsafe_recall):
    head.eval()
    with torch.no_grad():
        probabilities = torch.softmax(head(features), dim=1)
        edible_probabilities = probabilities[:, 1]

    candidates = []
    for threshold in torch.linspace(0.5, 0.98, steps=49).tolist():
        predicted_edible = edible_probabilities >= threshold
        actual_avoid = labels == 0
        actual_edible = labels == 1

        true_avoid = ((~predicted_edible) & actual_avoid).sum().item()
        false_safe = (predicted_edible & actual_avoid).sum().item()
        true_edible = (predicted_edible & actual_edible).sum().item()
        false_avoid = ((~predicted_edible) & actual_edible).sum().item()

        unsafe_total = max(1, actual_avoid.sum().item())
        predicted_edible_total = max(1, predicted_edible.sum().item())
        unsafe_recall = true_avoid / unsafe_total
        edible_precision = true_edible / predicted_edible_total
        accuracy = (true_avoid + true_edible) / max(1, labels.numel())

        if unsafe_recall >= target_unsafe_recall:
            candidates.append({
                "threshold": float(threshold),
                "accuracy": accuracy,
                "unsafe_recall": unsafe_recall,
                "edible_precision": edible_precision,
                "false_safe": false_safe,
                "false_avoid": false_avoid,
            })

    if not candidates:
        return 0.98

    best = max(
        candidates,
        key=lambda item: (
            item["accuracy"],
            item["edible_precision"],
            -item["false_safe"],
        ),
    )
    return best["threshold"]


def evaluate_risk_head(head, features, labels, edible_threshold):
    head.eval()
    with torch.no_grad():
        probabilities = torch.softmax(head(features), dim=1)
        edible_probabilities = probabilities[:, 1]

    predicted_edible = edible_probabilities >= edible_threshold
    actual_avoid = labels == 0
    actual_edible = labels == 1

    true_avoid = ((~predicted_edible) & actual_avoid).sum().item()
    false_safe = (predicted_edible & actual_avoid).sum().item()
    false_avoid = ((~predicted_edible) & actual_edible).sum().item()
    true_edible = (predicted_edible & actual_edible).sum().item()

    unsafe_total = max(1, actual_avoid.sum().item())
    predicted_edible_total = max(1, predicted_edible.sum().item())
    total = max(1, labels.numel())

    return {
        "accuracy": (true_avoid + true_edible) / total,
        "unsafe_recall": true_avoid / unsafe_total,
        "edible_precision": true_edible / predicted_edible_total,
        "false_safe_count": false_safe,
        "false_safe_rate": false_safe / unsafe_total,
        "confusion_matrix": [
            [true_avoid, false_safe],
            [false_avoid, true_edible],
        ],
        "edible_threshold": edible_threshold,
    }


def save_model(
    args,
    species_head,
    edibility_head,
    risk_head,
    class_to_idx,
    edibility_to_idx,
    risk_to_idx,
    edibility_map,
    risk_threshold,
):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier = species_head

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    idx_to_edibility = {idx: label for label, idx in edibility_to_idx.items()}
    idx_to_risk = {idx: label for label, idx in risk_to_idx.items()}

    checkpoint = {
        "model_name": "efficientnet_b0_feature_heads",
        "image_size": args.image_size,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "edibility_map": edibility_map,
        "edibility_to_idx": edibility_to_idx,
        "idx_to_edibility": idx_to_edibility,
        "edibility_head_state_dict": edibility_head.state_dict(),
        "risk_to_idx": risk_to_idx,
        "idx_to_risk": idx_to_risk,
        "risk_head_state_dict": risk_head.state_dict(),
        "risk_edible_threshold": risk_threshold,
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, MODEL_PATH)


def train(args):
    set_seed(args.seed)

    samples, class_names, class_to_idx, discovered_edibility = discover_image_dataset(args.data_dir)
    if len(class_names) < 2:
        raise ValueError("Need at least two species folders to train an image classifier.")

    train_samples, val_samples, test_samples = split_samples(samples, args.seed)
    edibility_map = load_edibility_map(args.edibility_map, class_names)
    edibility_map = {
        class_name: pick_riskier_edibility(
            discovered_edibility.get(class_name, "unknown"),
            edibility_map.get(class_name, "unknown"),
        )
        for class_name in class_names
    }

    _, eval_transform = build_transforms(args.image_size)
    train_dataset = SampleDataset(train_samples, class_to_idx, eval_transform)
    val_dataset = SampleDataset(val_samples, class_to_idx, eval_transform)
    test_dataset = SampleDataset(test_samples, class_to_idx, eval_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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

    device = choose_device()
    print(f"Using device: {device}")
    extractor = build_feature_extractor()

    print("Extracting train features...")
    train_features, train_labels = extract_features(extractor, train_loader, device)
    print("Extracting validation features...")
    val_features, val_labels = extract_features(extractor, val_loader, device)
    print("Extracting test features...")
    test_features, test_labels = extract_features(extractor, test_loader, device)

    print("Training species head...")
    species_head = train_head(
        train_features,
        train_labels,
        len(class_names),
        args.head_epochs,
        args.learning_rate,
        args.weight_decay,
    )

    idx_to_species = {idx: class_name for class_name, idx in class_to_idx.items()}
    edibility_classes = sorted(set(edibility_map.values()))
    edibility_to_idx = {label: idx for idx, label in enumerate(edibility_classes)}
    train_edibility_labels = build_edibility_labels(
        train_samples,
        idx_to_species,
        edibility_to_idx,
        edibility_map,
    )
    val_edibility_labels = build_edibility_labels(
        val_samples,
        idx_to_species,
        edibility_to_idx,
        edibility_map,
    )
    test_edibility_labels = build_edibility_labels(
        test_samples,
        idx_to_species,
        edibility_to_idx,
        edibility_map,
    )

    print("Training edibility head...")
    edibility_head = train_head(
        train_features,
        train_edibility_labels,
        len(edibility_classes),
        args.head_epochs,
        args.learning_rate,
        args.weight_decay,
    )

    risk_to_idx = {label: idx for idx, label in enumerate(RISK_LABELS)}
    train_risk_labels = build_risk_labels(train_samples, idx_to_species, edibility_map)
    val_risk_labels = build_risk_labels(val_samples, idx_to_species, edibility_map)
    test_risk_labels = build_risk_labels(test_samples, idx_to_species, edibility_map)

    print("Training safety risk head...")
    risk_head = train_head(
        train_features,
        train_risk_labels,
        len(RISK_LABELS),
        args.risk_epochs,
        args.risk_learning_rate,
        args.weight_decay,
    )

    risk_threshold = calibrate_edible_threshold(
        risk_head,
        val_features,
        val_risk_labels,
        args.target_unsafe_recall,
    )

    species_val = evaluate_head(species_head, val_features, val_labels)
    species_test = evaluate_head(species_head, test_features, test_labels)
    edibility_val = evaluate_head(edibility_head, val_features, val_edibility_labels)
    edibility_test = evaluate_head(edibility_head, test_features, test_edibility_labels)
    risk_val = evaluate_risk_head(risk_head, val_features, val_risk_labels, risk_threshold)
    risk_test = evaluate_risk_head(risk_head, test_features, test_risk_labels, risk_threshold)

    save_model(
        args,
        species_head,
        edibility_head,
        risk_head,
        class_to_idx,
        edibility_to_idx,
        risk_to_idx,
        edibility_map,
        risk_threshold,
    )

    metadata = {
        "model_name": "efficientnet_b0_feature_heads",
        "dataset_dir": os.path.abspath(args.data_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_images": len(samples),
        "image_size": args.image_size,
        "splits": {
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
        },
        "species_metrics": {
            "validation": species_val,
            "test": species_test,
        },
        "edibility_metrics": {
            "validation": edibility_val,
            "test": edibility_test,
            "classes": edibility_classes,
        },
        "risk_metrics": {
            "validation": risk_val,
            "test": risk_test,
            "classes": RISK_LABELS,
            "target_unsafe_recall": args.target_unsafe_recall,
            "edible_threshold": risk_threshold,
        },
        "best_validation_accuracy": species_val["top1_accuracy"],
        "test_metrics": {
            "top1_accuracy": species_test["top1_accuracy"],
            "top3_accuracy": species_test["top3_accuracy"],
        },
        "edibility_map": edibility_map,
        "training_strategy": (
            "ImageNet EfficientNet-B0 feature extractor with trained species "
            "suggestion, edibility subtype, and calibrated safety risk heads."
        ),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Species validation: {species_val}")
    print(f"Species test: {species_test}")
    print(f"Edibility validation: {edibility_val}")
    print(f"Edibility test: {edibility_test}")
    print(f"Risk validation: {risk_val}")
    print(f"Risk test: {risk_test}")
    print(f"Saved image model to {MODEL_PATH}")
    print(f"Saved image metadata to {METADATA_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fast high-quality image classifier heads on EfficientNet features."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--edibility-map", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=300)
    parser.add_argument("--risk-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--risk-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-unsafe-recall", type=float, default=TARGET_UNSAFE_RECALL)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
