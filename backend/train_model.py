from __future__ import annotations

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "mushroom")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_ENCODERS_PATH  = os.path.join(MODELS_DIR, "feature_encoders.joblib")
TARGET_ENCODER_PATH    = os.path.join(MODELS_DIR, "target_encoder.joblib")
RF_MODEL_PATH          = os.path.join(MODELS_DIR, "random_forest.joblib")
DT_MODEL_PATH          = os.path.join(MODELS_DIR, "decision_tree.joblib")
METADATA_PATH          = os.path.join(MODELS_DIR, "metadata.json")
FEATURE_MAPPING_PATH   = os.path.join(MODELS_DIR, "feature_mapping.json")
DATASET_PATH           = os.path.join(MODELS_DIR, "mushrooms.csv")
RAW_DATASET_PATH       = os.path.join(DATA_DIR, "agaricus-lepiota.data")
COLUMN_NAMES = [
    "classification",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]
FEATURE_LABELS = {
    "cap-shape": {
        "b": "Bell", "c": "Conical", "x": "Convex",
        "f": "Flat", "k": "Knobbed", "s": "Sunken"
    },
    "cap-surface": {
        "f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"
    },
    "cap-color": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray",
        "r": "Green", "p": "Pink", "u": "Purple", "e": "Red",
        "w": "White", "y": "Yellow"
    },
    "bruises": {"t": "Bruises", "f": "No Bruises"},
    "odor": {
        "a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy",
        "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"
    },
    "gill-attachment": {
        "a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"
    },
    "gill-spacing": {"c": "Close", "w": "Crowded", "d": "Distant"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {
        "k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate",
        "g": "Gray", "r": "Green", "o": "Orange", "p": "Pink",
        "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"
    },
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {
        "b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal",
        "z": "Rhizomorphs", "r": "Rooted", "?": "Missing"
    },
    "stalk-surface-above-ring": {
        "f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"
    },
    "stalk-surface-below-ring": {
        "f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"
    },
    "stalk-color-above-ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray",
        "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"
    },
    "stalk-color-below-ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray",
        "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"
    },
    "veil-type": {"p": "Partial", "u": "Universal"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {
        "c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large",
        "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone"
    },
    "spore-print-color": {
        "k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate",
        "r": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"
    },
    "population": {
        "a": "Abundant", "c": "Clustered", "n": "Numerous",
        "s": "Scattered", "v": "Several", "y": "Solitary"
    },
    "habitat": {
        "g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths",
        "u": "Urban", "w": "Waste", "d": "Woods"
    },
}

def load_dataset():
    if not os.path.exists(RAW_DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATASET_PATH}. "
            "Expected the mushroom dataset inside backend/mushroom."
        )

    print(f"Loading local mushroom dataset from {RAW_DATASET_PATH}...")
    df = pd.read_csv(RAW_DATASET_PATH, header=None, names=COLUMN_NAMES)
    df = df.fillna("?")

    X = df.drop(columns=["classification"]).copy()
    y = df["classification"].copy()

    print(f"  Loaded {len(X)} rows, {X.shape[1]} features from local dataset.")
    return X, y

def apply_dataset_noise(
    X: pd.DataFrame,
    noise_rate: float = 0.0,
    mask_rate: float = 0.0,
    drop_features: list[str] | None = None,
    seed: int = 42,
):
    """
    Adds controlled noise to reduce the overly clean nature of the UCI dataset.
    Defaults keep the original dataset unchanged.
    """
    X_noisy = X.copy()
    rng = np.random.default_rng(seed)

    if drop_features:
        existing = [feature for feature in drop_features if feature in X_noisy.columns]
        if existing:
            X_noisy = X_noisy.drop(columns=existing)
            print(f"Dropped features for robustness experiment: {existing}")

    if noise_rate > 0:
        print(f"Applying categorical replacement noise at rate {noise_rate:.3f}...")
        for column in X_noisy.columns:
            values = X_noisy[column].dropna().astype(str).unique()
            if len(values) <= 1:
                continue
            mask = rng.random(len(X_noisy)) < noise_rate
            replacement_values = rng.choice(values, size=mask.sum(), replace=True)
            X_noisy.loc[mask, column] = replacement_values

    if mask_rate > 0:
        print(f"Applying missing-value masking at rate {mask_rate:.3f}...")
        for column in X_noisy.columns:
            mask = rng.random(len(X_noisy)) < mask_rate
            X_noisy.loc[mask, column] = "?"

    return X_noisy

def preprocess(X: pd.DataFrame, y: pd.Series):
    print("Preprocessing...")

    X = X.fillna("?")

    feature_encoders: dict[str, LabelEncoder] = {}
    X_enc = X.copy()
    for col in X.columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        feature_encoders[col] = le

    target_encoder = LabelEncoder()
    y_enc = target_encoder.fit_transform(y.astype(str))

    print(f"  Classes: {list(target_encoder.classes_)}") 
    return X_enc, y_enc, feature_encoders, target_encoder

def train_and_evaluate(X_enc, y_enc, feature_names):
    print("Splitting data 80/20 …")
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            min_samples_split=2,
            random_state=42,
        ),
    }

    results = {}
    trained_models = {}

    for name, clf in models.items():
        print(f"\nTraining {name} …")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cv   = cross_val_score(clf, X_enc, y_enc, cv=5, scoring="accuracy")
        cm   = confusion_matrix(y_test, y_pred).tolist()

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  5-Fold CV: {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(y_test, y_pred, target_names=["edible", "poisonous"]))

        importances = clf.feature_importances_
        feat_imp = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        results[name] = {
            "accuracy":          round(acc, 6),
            "precision":         round(prec, 6),
            "recall":            round(rec, 6),
            "f1_score":          round(f1, 6),
            "cross_val_mean":    round(float(cv.mean()), 6),
            "cross_val_std":     round(float(cv.std()), 6),
            "confusion_matrix":  cm,
            "feature_importance": [
                {"feature": f, "importance": round(float(i), 6)}
                for f, i in feat_imp
            ],
        }
        trained_models[name] = clf

    return trained_models, results, X_train, X_test, y_train, y_test

def build_readable_dataset(X_raw: pd.DataFrame, y_raw: pd.Series):
    """
    Returns a list of dicts with human-readable values so the API
    can serve the edible / poisonous mushroom lists directly.
    """
    records = []
    for idx, (_, row) in enumerate(X_raw.iterrows()):
        label = y_raw.iloc[idx]   
        rec = {
            "id": idx,
            "classification": "edible" if label == "e" else "poisonous",
        }
        for col in X_raw.columns:
            val = str(row[col]) if not pd.isna(row[col]) else "?"
            mapping = FEATURE_LABELS.get(col, {})
            rec[col] = mapping.get(val, val)
        records.append(rec)
    return records

def build_feature_mapping(X_raw: pd.DataFrame):
    mapping = {}
    for col in X_raw.columns:
        unique_vals = X_raw[col].fillna("?").unique().tolist()
        label_map = FEATURE_LABELS.get(col, {})
        options = []
        for v in sorted(unique_vals):
            options.append({
                "value": str(v),
                "label": label_map.get(str(v), str(v))
            })
        mapping[col] = options
    return mapping

def main(args):
    X_raw, y_raw = load_dataset()
    X_raw = apply_dataset_noise(
        X_raw,
        noise_rate=args.noise_rate,
        mask_rate=args.mask_rate,
        drop_features=args.drop_features,
        seed=args.seed,
    )

    df_full = X_raw.copy()
    df_full["poisonous"] = y_raw.values
    df_full.to_csv(DATASET_PATH, index=False)
    print(f"\nSaved raw dataset → {DATASET_PATH}")

    X_enc, y_enc, feature_encoders, target_encoder = preprocess(X_raw, y_raw)

    trained_models, results, X_train, X_test, y_train, y_test = train_and_evaluate(
        X_enc, y_enc, list(X_raw.columns)
    )

    joblib.dump(feature_encoders, FEATURE_ENCODERS_PATH)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)
    joblib.dump(trained_models["random_forest"], RF_MODEL_PATH)
    joblib.dump(trained_models["decision_tree"], DT_MODEL_PATH)
    print(f"\nModels saved to {MODELS_DIR}/")

    metadata = {
        "feature_names": list(X_raw.columns),
        "target_classes": list(target_encoder.classes_),
        "total_samples": len(X_raw),
        "edible_count": int((y_raw == "e").sum()),
        "poisonous_count": int((y_raw == "p").sum()),
        "dataset_source": RAW_DATASET_PATH,
        "noise_config": {
            "noise_rate": args.noise_rate,
            "mask_rate": args.mask_rate,
            "drop_features": args.drop_features,
            "seed": args.seed,
        },
        "models": results,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved → {METADATA_PATH}")

    feature_mapping = build_feature_mapping(X_raw)
    with open(FEATURE_MAPPING_PATH, "w") as f:
        json.dump(feature_mapping, f, indent=2)
    print(f"Feature mapping saved → {FEATURE_MAPPING_PATH}")

    records = build_readable_dataset(X_raw, y_raw)
    readable_path = os.path.join(MODELS_DIR, "mushrooms_readable.json")
    with open(readable_path, "w") as f:
        json.dump(records, f)
    print(f"Readable dataset saved → {readable_path}")

    print("\n✅ Training complete. You can now start the Flask server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular mushroom classifiers.")
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.0,
        help="Fraction of categorical cells to replace with another valid value.",
    )
    parser.add_argument(
        "--mask-rate",
        type=float,
        default=0.0,
        help="Fraction of categorical cells to replace with '?'.",
    )
    parser.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        help="Feature names to remove for robustness experiments, e.g. odor.",
    )
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
