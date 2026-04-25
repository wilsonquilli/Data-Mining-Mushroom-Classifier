import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from image_model import image_model_status, predict_image
    IMAGE_PIPELINE_ERROR = None
except Exception as e:
    image_model_status = None
    predict_image = None
    IMAGE_PIPELINE_ERROR = str(e)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_artefacts():
    required = [
        "feature_encoders.joblib",
        "target_encoder.joblib",
        "random_forest.joblib",
        "decision_tree.joblib",
        "metadata.json",
        "feature_mapping.json",
        "mushrooms_readable.json",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}. "
            "Please run `python train_model.py` first."
        )

    artefacts = {
        "feature_encoders": joblib.load(os.path.join(MODELS_DIR, "feature_encoders.joblib")),
        "target_encoder":   joblib.load(os.path.join(MODELS_DIR, "target_encoder.joblib")),
        "random_forest":    joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "decision_tree":    joblib.load(os.path.join(MODELS_DIR, "decision_tree.joblib")),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        artefacts["metadata"] = json.load(f)
    with open(os.path.join(MODELS_DIR, "feature_mapping.json")) as f:
        artefacts["feature_mapping"] = json.load(f)
    with open(os.path.join(MODELS_DIR, "mushrooms_readable.json")) as f:
        artefacts["mushrooms"] = json.load(f)

    return artefacts

try:
    A = load_artefacts()
    print("✅ All model artefacts loaded.")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    A = {}

DISPLAY_FIELDS = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-size",
    "gill-color",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

def encode_input(feature_dict: dict) -> pd.DataFrame:
    """
    Accepts a dict like {"cap-shape": "x", "odor": "n", ...}
    and returns a single-row DataFrame ready for clf.predict().
    Missing features fall back to the most-frequent class seen during training.
    """
    feature_encoders: dict = A["feature_encoders"]
    feature_names: list    = A["metadata"]["feature_names"]
    row = []
    for feat in feature_names:
        le = feature_encoders[feat]
        val = feature_dict.get(feat, le.classes_[0])  
        val = str(val)
        if val not in le.classes_:
            val = le.classes_[0]
        row.append(le.transform([val])[0])
    return pd.DataFrame([row], columns=feature_names)

def paginate_records(records, page, per_page):
    total = len(records)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "data": records[start:end],
    }

def unique_mushroom_profiles(records):
    grouped = {}

    for mushroom in records:
        signature = tuple((field, mushroom[field]) for field in DISPLAY_FIELDS)
        if signature not in grouped:
            grouped[signature] = {
                **{field: mushroom[field] for field in DISPLAY_FIELDS},
                "classification": mushroom["classification"],
                "example_id": mushroom["id"],
                "occurrence_count": 0,
                "source_ids": [],
            }

        grouped[signature]["occurrence_count"] += 1
        grouped[signature]["source_ids"].append(mushroom["id"])

    return sorted(
        grouped.values(),
        key=lambda item: (-item["occurrence_count"], item["example_id"]),
    )

def filtered_mushroom_response(classification=None):
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    page = max(1, int(request.args.get("page", 1)))
    per_page = min(200, int(request.args.get("per_page", 50)))
    unique = request.args.get("unique", "false").lower() == "true"

    mushrooms = A["mushrooms"]
    if classification:
        mushrooms = [m for m in mushrooms if m["classification"] == classification]

    if unique:
        unique_profiles = unique_mushroom_profiles(mushrooms)
        payload = paginate_records(unique_profiles, page, per_page)
        payload["unique_profiles"] = True
        payload["raw_total"] = len(mushrooms)
        return jsonify(payload)

    payload = paginate_records(mushrooms, page, per_page)
    payload["unique_profiles"] = False
    return jsonify(payload)

@app.route("/api/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok", "models_loaded": bool(A)})

@app.route("/api/mushrooms", methods=["GET"])
def get_all_mushrooms():
    """
    Returns paginated list of all mushrooms.
    Query params:
      page     (int, default 1)
      per_page (int, default 50)
    """
    return filtered_mushroom_response()


@app.route("/api/mushrooms/edible", methods=["GET"])
def get_edible():
    """
    Returns paginated list of edible mushrooms.
    Query params:
      page     (int, default 1)
      per_page (int, default 50)
    """
    return filtered_mushroom_response("edible")

@app.route("/api/mushrooms/poisonous", methods=["GET"])
def get_poisonous():
    """
    Returns paginated list of poisonous mushrooms.
    Query params:
      page     (int, default 1)
      per_page (int, default 50)
    """
    return filtered_mushroom_response("poisonous")

@app.route("/api/mushrooms/<int:mushroom_id>", methods=["GET"])
def get_mushroom(mushroom_id):
    """Returns a single mushroom record by its id."""
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    if mushroom_id < 0 or mushroom_id >= len(A["mushrooms"]):
        return jsonify({"error": "Mushroom not found"}), 404

    return jsonify(A["mushrooms"][mushroom_id])

@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Classify a mushroom given its feature values.

    Request body (JSON):
    {
        "features": {
            "cap-shape": "x",
            "cap-surface": "s",
            "cap-color": "n",
            "bruises": "t",
            "odor": "p",
            ... (any subset of the 22 features)
        },
        "model": "random_forest"   // or "decision_tree" (optional, default rf)
    }

    Response:
    {
        "classification": "poisonous",
        "confidence": {
            "edible": 0.03,
            "poisonous": 0.97
        },
        "model_used": "random_forest",
        "features_used": { ... }
    }
    """
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    body = request.get_json(silent=True)
    if not body or "features" not in body:
        return jsonify({"error": "Request body must contain a 'features' object"}), 400

    model_name = body.get("model", "random_forest")
    if model_name not in ("random_forest", "decision_tree"):
        return jsonify({"error": "model must be 'random_forest' or 'decision_tree'"}), 400

    clf = A[model_name]
    features = body["features"]

    try:
        X = encode_input(features)
    except Exception as e:
        return jsonify({"error": f"Encoding error: {str(e)}"}), 400

    pred_encoded   = clf.predict(X)[0]
    pred_proba     = clf.predict_proba(X)[0]
    target_encoder = A["target_encoder"]
    classes        = target_encoder.classes_   # ['e', 'p']

    label_map  = {"e": "edible", "p": "poisonous"}
    prediction = label_map[target_encoder.inverse_transform([pred_encoded])[0]]

    confidence = {
        label_map[cls]: round(float(prob), 4)
        for cls, prob in zip(classes, pred_proba)
    }

    return jsonify({
        "classification": prediction,
        "confidence":     confidence,
        "model_used":     model_name,
        "features_used":  features,
    })

@app.route("/api/classify/both", methods=["POST"])
def classify_both():
    """
    Runs the same feature set through both models and returns
    a side-by-side comparison. Useful for showing users the
    difference between RF and DT predictions.

    Request body: same as /api/classify (no 'model' key needed).
    """
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    body = request.get_json(silent=True)
    if not body or "features" not in body:
        return jsonify({"error": "Request body must contain a 'features' object"}), 400

    features = body["features"]
    try:
        X = encode_input(features)
    except Exception as e:
        return jsonify({"error": f"Encoding error: {str(e)}"}), 400

    target_encoder = A["target_encoder"]
    classes        = target_encoder.classes_
    label_map      = {"e": "edible", "p": "poisonous"}

    comparisons = {}
    for model_name in ("random_forest", "decision_tree"):
        clf          = A[model_name]
        pred_encoded = clf.predict(X)[0]
        pred_proba   = clf.predict_proba(X)[0]
        prediction   = label_map[target_encoder.inverse_transform([pred_encoded])[0]]
        confidence   = {
            label_map[cls]: round(float(prob), 4)
            for cls, prob in zip(classes, pred_proba)
        }
        comparisons[model_name] = {
            "classification": prediction,
            "confidence":     confidence,
        }

    rf_pred = comparisons["random_forest"]["classification"]
    dt_pred = comparisons["decision_tree"]["classification"]
    comparisons["models_agree"] = rf_pred == dt_pred

    return jsonify(comparisons)

@app.route("/api/image-model/status", methods=["GET"])
def get_image_model_status():
    """Returns whether the computer-vision model is trained and loadable."""
    if IMAGE_PIPELINE_ERROR:
        return jsonify({
            "available": False,
            "error": IMAGE_PIPELINE_ERROR,
        }), 503

    return jsonify(image_model_status())

@app.route("/api/predict-image", methods=["POST"])
def predict_uploaded_image():
    """
    Predict mushroom species and edibility from an uploaded image.

    Expects multipart/form-data with a file field named "image".
    """
    if IMAGE_PIPELINE_ERROR:
        return jsonify({
            "error": "Image prediction dependencies are not available",
            "details": IMAGE_PIPELINE_ERROR,
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "Upload an image file using the 'image' field"}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "Uploaded image is missing a filename"}), 400

    top_k = int(request.form.get("top_k", 3))
    try:
        return jsonify(predict_image(image_file, top_k=top_k))
    except FileNotFoundError as e:
        return jsonify({
            "error": "Image model not trained",
            "details": str(e),
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Image prediction failed",
            "details": str(e),
        }), 400

@app.route("/api/model/stats", methods=["GET"])
def model_stats():
    """
    Returns accuracy, precision, recall, F1, CV scores, and
    confusion matrix for both trained models.
    """
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    meta = A["metadata"]
    return jsonify({
        "dataset": {
            "total_samples":   meta["total_samples"],
            "edible_count":    meta["edible_count"],
            "poisonous_count": meta["poisonous_count"],
        },
        "models": meta["models"],
    })

@app.route("/api/model/feature-importance", methods=["GET"])
def feature_importance():
    """
    Returns the top-N most important features for each model.
    Query param: top (int, default 10)
    """
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    top = int(request.args.get("top", 10))
    result = {}
    for model_name in ("random_forest", "decision_tree"):
        fi = A["metadata"]["models"][model_name]["feature_importance"]
        result[model_name] = fi[:top]

    return jsonify(result)

@app.route("/api/features", methods=["GET"])
def get_features():
    """
    Returns all features and their possible values so the
    React frontend can dynamically render the classification form.

    Response shape:
    {
        "features": {
            "cap-shape": [
                {"value": "b", "label": "Bell"},
                ...
            ],
            ...
        }
    }
    """
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    return jsonify({"features": A["feature_mapping"]})

@app.route("/api/stats", methods=["GET"])
def stats():
    """High-level dataset and model summary for the homepage."""
    if not A:
        return jsonify({"error": "Models not loaded"}), 503

    meta = A["metadata"]
    return jsonify({
        "total_mushrooms": meta["total_samples"],
        "edible":          meta["edible_count"],
        "poisonous":       meta["poisonous_count"],
        "total_features":  len(meta["feature_names"]),
        "rf_accuracy":     meta["models"]["random_forest"]["accuracy"],
        "dt_accuracy":     meta["models"]["decision_tree"]["accuracy"],
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
