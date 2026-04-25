const BASE = import.meta.env.VITE_API_BASE_URL || "/api";

async function fetchJson(path, options) {
  const response = await fetch(`${BASE}${path}`, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || `Request failed with status ${response.status}`);
  }

  return data;
}

/**
 * Fetch a page of edible mushrooms.
 * @param {number} page     1-based page number
 * @param {number} perPage  Items per page (max 200)
 */
export const getEdible = (page = 1, perPage = 50, unique = false) =>
  fetchJson(`/mushrooms/edible?page=${page}&per_page=${perPage}&unique=${unique}`);

/**
 * Fetch a page of poisonous mushrooms.
 */
export const getPoisonous = (page = 1, perPage = 50, unique = false) =>
  fetchJson(`/mushrooms/poisonous?page=${page}&per_page=${perPage}&unique=${unique}`);

/**
 * Fetch all mushrooms (paginated).
 */
export const getAllMushrooms = (page = 1, perPage = 50) =>
  fetchJson(`/mushrooms?page=${page}&per_page=${perPage}`);

/**
 * Fetch a single mushroom by numeric id.
 */
export const getMushroomById = (id) =>
  fetchJson(`/mushrooms/${id}`);

/**
 * Classify a mushroom using a single model.
 *
 * @param {Object} features   e.g. { "cap-shape": "x", "odor": "n", ... }
 * @param {string} model      "random_forest" | "decision_tree"
 *
 * @returns {{ classification: string, confidence: Object, model_used: string }}
 */
export const classify = (features, model = "random_forest") =>
  fetchJson(`/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features, model }),
  });

/**
 * Run features through both models and compare results.
 *
 * @returns {{
 *   random_forest: { classification, confidence },
 *   decision_tree:  { classification, confidence },
 *   models_agree: boolean
 * }}
 */
export const classifyBoth = (features) =>
  fetchJson(`/classify/both`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });

/**
 * Fetch accuracy, precision, recall, F1, CV scores, and confusion
 * matrices for both models.
 */
export const getModelStats = () =>
  fetchJson(`/model/stats`);

/**
 * Fetch top-N most important features.
 * @param {number} top  Number of features to return (default 10)
 */
export const getFeatureImportance = (top = 10) =>
  fetchJson(`/model/feature-importance?top=${top}`);

/**
 * Fetch all feature names and their possible values with human-readable labels.
 * Use this to dynamically render the classification form.
 *
 * @returns {{ features: { [featureName]: Array<{ value, label }> } }}
 */
export const getFeatures = () =>
  fetchJson(`/features`);

/**
 * Fetch high-level dataset + model summary for the homepage.
 *
 * @returns {{
 *   total_mushrooms, edible, poisonous,
 *   total_features, rf_accuracy, dt_accuracy
 * }}
 */
export const getStats = () =>
  fetchJson(`/stats`);

/**
 * Fetch computer-vision model availability and training metadata.
 */
export const getImageModelStatus = () =>
  fetchJson(`/image-model/status`);

/**
 * Upload a mushroom image and get species + edibility predictions.
 *
 * @param {File} imageFile
 * @param {number} topK
 */
export const predictImage = (imageFile, topK = 3) => {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("top_k", String(topK));

  return fetchJson(`/predict-image`, {
    method: "POST",
    body: formData,
  });
};
