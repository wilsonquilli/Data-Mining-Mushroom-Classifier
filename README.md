# Mushroom Classifier

## Abstract
The Mushroom Classifier is a full-stack web application designed to help users determine whether a mushroom is safe to eat or poisonous. Built with a Python backend and a React frontend, the application leverages data mining techniques, specifically the Random Forest and Decision Tree algorithms, to deliver accurate, reliable mushroom classifications.  

The dataset powering the classifier is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom) and contains over 8,000 instances described by 22 categorical features, including cap shape, cap surface, odor, gill size, stalk shape, and spore print color. These features serve as the foundation for applying supervised learning, where the goal is to predict whether a given mushroom is edible or poisonous based on its physical characteristics.  

A REST API connects the Python backend to the React frontend, enabling the application to display classified mushroom data across dedicated edible and poisonous pages. Accurate mushroom identification is a critical real-world problem, as misidentification can lead to severe health consequences, making precise and dependable classification essential.  

## Introduction
The Mushroom Classifier application classifies mushrooms as either edible and safe for consumption or poisonous and dangerous. The full-stack application uses:

- **Backend:** Python  
- **Frontend:** React  

It leverages Data Mining algorithms such as **Random Forest** and **Decision Trees** to achieve the most accurate results for users. An API connects the backend to the frontend, enabling the application to display poisonous and edible mushrooms on dedicated pages.  

This project frames mushroom classification as a **supervised learning problem**, using categorical attributes describing physical characteristics to predict whether a mushroom is edible or poisonous. The Mushroom dataset from the UCI Machine Learning Repository provides over 8,000 instances and 22 categorical features such as odor, gill size, stalk shape, and spore print color, which form the basis for applying classification-based data mining techniques.  

## Problem Statement
It can be difficult for non-experts to distinguish between edible and poisonous mushrooms. Eating the wrong type can result in serious health consequences. While trained specialists may know how to identify mushrooms, many people do not.  

**Mushroom Classifier** solves this problem by using machine learning to predict whether a mushroom is edible or poisonous based on its physical features. By using data mining algorithms, the application provides a simple, reliable, and safe decision-support tool for users.  

## Goal & Objectives
The goal of the Mushroom Classifier is to develop a **reliable and user-friendly machine learning application** that accurately predicts whether a mushroom is edible or poisonous. Key objectives include:

- Implementing **Decision Tree** and **Random Forest** algorithms for classification.
- Providing a **safe and accessible tool** for individuals without expert knowledge of mushrooms.
- Connecting a **Python backend** with a **React frontend** through a REST API to display results in an intuitive interface.

## Dataset
- **Source:** [UCI Machine Learning Repository – Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)  
- **Size:** 8,124 instances  
- **Features:** 22 categorical features such as cap-shape, cap-surface, odor, gill-size, stalk-shape, and spore-print-color  
- **Task:** Classify mushrooms as edible or poisonous based on their physical characteristics

## Team Members & Contributions
| Name | Contribution |
|------|--------------|
| **Wilson Quilli** | Frontend with React and wrote the final report |
| **Mostafa Amer** | Backend with Python |
| **Mohamed Abdalla** | Integrated backend and frontend; created the final presentation |

## Technologies Used
- Python (backend)
- React (frontend)
- REST API
- Random Forest & Decision Tree Algorithms
- UCI Mushroom Dataset

## Image-Based Mushroom Identification

The original UCI mushroom dataset is tabular, not image-based. It is useful for
the data mining portion of the project, but it cannot train a photo upload
classifier. The project now includes a separate computer-vision pipeline for
image datasets.

### Image Dataset Layout

Place mushroom images in species folders:

```text
backend/image_dataset/
  Agaricus_bisporus/
    image_001.jpg
    image_002.jpg
  Amanita_muscaria/
    image_001.jpg
  Amanita_phalloides/
    image_001.jpg
```

Create an edibility map at `backend/edibility_map.json`. You can start from
`backend/edibility_map.example.json`.

```json
{
  "Agaricus_bisporus": "edible",
  "Amanita_muscaria": "poisonous",
  "Amanita_phalloides": "deadly"
}
```

### Train the Image Model

```bash
cd backend
pip install -r requirements.txt
python train_image_model.py --data-dir image_dataset --edibility-map edibility_map.json
```

The training script uses transfer learning with EfficientNet-B0, ImageNet
pretrained weights, stratified train/validation/test splits, class-weighted
loss, data augmentation, fine-tuning, and early stopping. It saves:

```text
backend/models/image_classifier.pt
backend/models/image_model_metadata.json
```

### Image Prediction API

After training, the Flask backend exposes:

```text
GET  /api/image-model/status
POST /api/predict-image
```

`POST /api/predict-image` expects multipart form data with an `image` field and
returns predicted species, confidence, edibility, top predictions, and a safety
warning.

The current image pipeline is safety-first. It trains a calibrated risk head
that returns:

- `risk_label`: `edible`, `avoid`, or `uncertain`
- `risk_confidence`: confidence for the safety decision
- `species_suggestions`: optional top species matches
- `warning`: a reminder not to consume wild mushrooms based only on the app

For the practical project demo, `conditionally_edible`, `poisonous`, and
`deadly` are treated as `avoid`. Species identification remains a secondary
suggestion because the image dataset has many species with relatively few
examples.

To train the safety-first model:

```bash
cd backend
python train_image_feature_heads.py \
  --data-dir /path/to/mushroom_dataset \
  --image-size 224 \
  --batch-size 64 \
  --head-epochs 300 \
  --risk-epochs 300 \
  --target-unsafe-recall 0.90
```

The training output reports risk accuracy, unsafe recall, edible precision,
false-safe count/rate, and a binary confusion matrix. The app should prioritize
unsafe recall and false-safe rate over exact species accuracy.

### Overfitting Controls

For the tabular UCI model, the project can reduce overfitting by masking some
features, adding controlled categorical noise, removing overly predictive
features such as odor in experiments, and relying on cross-validation instead of
only a single train/test split.

For the image model, the project uses augmentation, dropout, weight decay,
class-weighted loss, early stopping, and a holdout test split.
