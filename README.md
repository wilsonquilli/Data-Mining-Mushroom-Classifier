# Mushroom Classifier

Full-stack mushroom classification project with a Flask backend and React frontend.
The app supports two machine-learning workflows:

- A tabular data-mining classifier trained on the UCI Mushroom dataset.
- A safety-first image classifier that accepts a mushroom photo and predicts whether the mushroom should be treated as edible, avoid, or uncertain.

This project is a decision-support/demo tool. It must not be used as the only basis for eating wild mushrooms.

## Features

- Upload a mushroom photo and receive a safety-first result.
- Show the #1 species candidate and other possible matches as secondary context.
- Classify tabular mushroom features with Random Forest and Decision Tree models.
- Browse edible and poisonous mushroom profiles from the UCI dataset.
- Expose all model results through a REST API.

## Tech Stack

- Backend: Python, Flask, scikit-learn, PyTorch, torchvision
- Frontend: React, Vite, Tailwind CSS
- Data mining models: Random Forest, Decision Tree
- Image model: EfficientNet-B0 feature extractor with trained classifier heads

## Project Structure

```text
backend/
  app.py                         Flask API
  train_model.py                 UCI tabular training pipeline
  train_image_model.py           CNN fine-tuning training pipeline
  train_image_feature_heads.py   Safety-first image-head training pipeline
  image_model.py                 Image model loading and inference
  mushroom/                      UCI mushroom dataset files
  models/                        Saved model artifacts

frontend/
  src/App.jsx                    Main React app and routes
  src/api.js                     API client helpers
  src/pages/edible.jsx           Edible profile browser
  src/pages/poisonous.jsx        Poisonous profile browser
```

## Current Model Results

### Tabular UCI Model

The UCI model was retrained with controlled noise to reduce overfitting:

- Categorical replacement noise: `3%`
- Missing-value masking: `5%`

Current saved metrics:

- Random Forest accuracy: `99.69%`
- Decision Tree accuracy: `98.46%`

### Image Safety Model

The image model was trained on a Kaggle mushroom image dataset with:

- `8,781` images
- `247` species
- Four source labels: `edible`, `conditionally_edible`, `poisonous`, `deadly`

For safety, the app maps labels into:

- `edible`
- `avoid` = `conditionally_edible`, `poisonous`, `deadly`
- `uncertain` when confidence is too low

Current saved risk-head metrics:

- Risk accuracy: `80.4%`
- Unsafe recall: `96.3%`
- Edible precision: `79.8%`
- False-safe rate: `3.66%`
- Confusion matrix: `[[922, 35], [223, 138]]`

Species recognition is secondary because the image dataset has many species with few examples. Current species metrics:

- Top-1 species accuracy: `50.2%`
- Top-3 species accuracy: `69.8%`

## Setup

### Backend

```bash
cd backend
python3 -m pip install -r requirements.txt
python3 app.py
```

The Flask backend runs at:

```text
http://127.0.0.1:5000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://127.0.0.1:5173
```

If that port is busy, Vite will print another local URL such as `http://127.0.0.1:5174`.

## API Endpoints

```text
GET  /api/health
GET  /api/stats
GET  /api/features
GET  /api/model/stats
GET  /api/model/feature-importance
GET  /api/mushrooms
GET  /api/mushrooms/edible
GET  /api/mushrooms/poisonous
POST /api/classify
POST /api/classify/both
GET  /api/image-model/status
POST /api/predict-image
```

`POST /api/predict-image` expects multipart form data with an `image` field.

Example image response:

```json
{
  "risk_label": "avoid",
  "risk_confidence": 0.92,
  "risk_subtype": "poisonous",
  "species_suggestions": [
    {
      "species": "Amanita abrupta",
      "confidence": 0.83,
      "edibility": "poisonous"
    }
  ],
  "warning": "Do not consume wild mushrooms based only on this application. Consult a qualified expert before eating any foraged mushroom."
}
```

## Training

### Retrain the UCI Tabular Models

```bash
cd backend
python3 train_model.py --noise-rate 0.03 --mask-rate 0.05
```

Optional robustness experiment:

```bash
python3 train_model.py --noise-rate 0.03 --mask-rate 0.05 --drop-features odor
```

### Train the Safety-First Image Model

The image training script supports datasets arranged as:

```text
mushroom_dataset/
  edible/
    Species_name/
      image.png
  poisonous/
    Species_name/
      image.png
  deadly/
    Species_name/
      image.png
  conditionally_edible/
    Species_name/
      image.png
```

Run:

```bash
cd backend
python3 train_image_feature_heads.py \
  --data-dir /path/to/mushroom_dataset \
  --image-size 224 \
  --batch-size 64 \
  --head-epochs 300 \
  --risk-epochs 300 \
  --target-unsafe-recall 0.90
```

This saves:

```text
backend/models/image_classifier.pt
backend/models/image_model_metadata.json
```

The training output reports:

- risk accuracy
- unsafe recall
- edible precision
- false-safe count and rate
- confusion matrix
- top-1 and top-3 species accuracy

## Safety Notes

- The app should never be used as the only source for deciding whether to eat a wild mushroom.
- `avoid` and `uncertain` should both be treated as do-not-eat results.
- Exact species prediction is shown only as a possible match, not as a guarantee.
- The safest project metric is unsafe recall, not overall accuracy.

## Team

| Name | Contribution |
|------|--------------|
| Wilson Quilli | Frontend with React and final report |
| Mostafa Amer | Backend with Python and model training |
| Mohamed Abdalla | Backend/frontend integration and final presentation |
