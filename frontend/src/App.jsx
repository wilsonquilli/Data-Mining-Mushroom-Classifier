import { useEffect, useState } from "react";
import Navbar from "./components/navbar";
import Footer from "./components/footer";
import MushroomLogo from "./assets/mushroom_logo.png";
import { Routes, Route, Link } from "react-router-dom";
import Edible from "./pages/edible";
import Poisonous from "./pages/poisonous";
import ScrollToTop from "./components/scrolltotop";
import {
  classifyBoth,
  getFeatureImportance,
  getFeatures,
  getImageModelStatus,
  getStats,
  predictImage,
} from "./api";

const FEATURE_PICK_ORDER = [
  "odor",
  "cap-shape",
  "cap-surface",
  "cap-color",
  "bruises",
  "gill-size",
];

function StatCard({ label, value }) {
  return (
    <div className="rounded-3xl border border-stone-200 bg-white px-5 py-6 shadow-sm">
      <p className="text-sm uppercase tracking-[0.2em] text-stone-500">{label}</p>
      <p className="mt-3 font-patua text-3xl text-stone-900">{value}</p>
    </div>
  );
}

function FeatureSelect({ featureName, options, value, onChange }) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-semibold capitalize text-stone-700">
        {featureName.replaceAll("-", " ")}
      </span>
      <select
        value={value}
        onChange={(event) => onChange(featureName, event.target.value)}
        className="w-full rounded-2xl border border-stone-300 bg-white px-4 py-3 text-sm text-stone-800 outline-none transition focus:border-red-500"
      >
        <option value="">Select an option</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function Home() {
  const [stats, setStats] = useState(null);
  const [features, setFeatures] = useState({});
  const [featureImportance, setFeatureImportance] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState({});
  const [classification, setClassification] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [imagePrediction, setImagePrediction] = useState(null);
  const [imageModelStatus, setImageModelStatus] = useState(null);
  const [loadingHome, setLoadingHome] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [submittingImage, setSubmittingImage] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadHomepageData() {
      setLoadingHome(true);
      setError("");

      try {
        const [statsResponse, featuresResponse, featureImportanceResponse] = await Promise.all([
          getStats(),
          getFeatures(),
          getFeatureImportance(5),
        ]);

        if (!cancelled) {
          setStats(statsResponse);
          setFeatures(featuresResponse.features);
          setFeatureImportance(featureImportanceResponse);

          const defaults = {};
          FEATURE_PICK_ORDER.forEach((featureName) => {
            const firstOption = featuresResponse.features[featureName]?.[0];
            if (firstOption) {
              defaults[featureName] = firstOption.value;
            }
          });
          setSelectedFeatures(defaults);

          try {
            const statusResponse = await getImageModelStatus();
            setImageModelStatus(statusResponse);
          } catch (statusError) {
            setImageModelStatus({
              available: false,
              error: statusError.message,
            });
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingHome(false);
        }
      }
    }

    loadHomepageData();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!imageFile) {
      setImagePreview("");
      return undefined;
    }

    const previewUrl = URL.createObjectURL(imageFile);
    setImagePreview(previewUrl);

    return () => URL.revokeObjectURL(previewUrl);
  }, [imageFile]);

  function handleFeatureChange(featureName, value) {
    setSelectedFeatures((current) => ({
      ...current,
      [featureName]: value,
    }));
  }

  async function handleClassify(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      const activeFeatures = Object.fromEntries(
        Object.entries(selectedFeatures).filter(([, value]) => value)
      );
      const result = await classifyBoth(activeFeatures);
      setClassification(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  function handleImageChange(event) {
    const file = event.target.files?.[0];
    setImageFile(file || null);
    setImagePrediction(null);
  }

  async function handleImagePredict(event) {
    event.preventDefault();
    if (!imageFile) {
      setError("Choose a mushroom image before running image identification.");
      return;
    }

    setSubmittingImage(true);
    setError("");

    try {
      const result = await predictImage(imageFile, 3);
      setImagePrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmittingImage(false);
    }
  }

  const topRandomForestFeature = featureImportance?.random_forest?.[0];
  const topDecisionTreeFeature = featureImportance?.decision_tree?.[0];
  const imageModelReady = imageModelStatus?.available;
  const riskLabel = imagePrediction?.risk_label || imagePrediction?.edibility;
  const riskIsUncertain = riskLabel === "uncertain" || riskLabel === "unknown-risk";
  const speciesSuggestions = imagePrediction?.species_suggestions || imagePrediction?.top_predictions || [];
  const topSpeciesCandidate = speciesSuggestions[0];

  return (
    <div className="bg-[linear-gradient(180deg,#fffdf7_0%,#fff7ef_38%,#ffffff_100%)] text-gray-900">

      <section className="max-w-5xl mx-auto px-6 pt-20 pb-16 text-center">
        <div className="flex justify-center mb-6">
          <img
            src={MushroomLogo}
            alt="Mushroom Logo"
            className="h-16 w-auto"
          />
        </div>

        <h1 className="text-4xl md:text-5xl font-bold font-patua mb-6 leading-tight">
          Mushroom Classifier: A Data Mining Project
          <span className="text-red-500">.</span>
        </h1>

        <p className="text-gray-600 text-lg max-w-2xl mx-auto mb-8">
          The Mushroom Classifier application predicts whether mushrooms are edible or poisonous using advanced Data Mining techniques such as Random Forest and Decision Trees.
          The backend mines the mushroom dataset stored inside the project and exposes the results for the frontend to display.
        </p>

        <div className="flex justify-center gap-4">
          <Link
            to="/edible"
            className="px-6 py-3 rounded-full bg-red-500 text-white font-medium shadow-md hover:bg-red-600 transition"
          >
            Explore Edible Mushrooms
          </Link>

          <Link
            to="/poisonous"
            className="px-6 py-3 rounded-full border border-gray-300 text-gray-700 font-medium hover:border-red-500 hover:text-red-500 transition"
          >
            Explore Poisonous Mushrooms
          </Link>
        </div>
      </section>

      <section className="max-w-5xl mx-auto px-6 pb-12">
        {error && (
          <div className="mb-6 rounded-3xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-5">
          <StatCard
            label="Dataset Rows"
            value={loadingHome ? "..." : stats?.total_mushrooms ?? 0}
          />
          <StatCard
            label="Edible"
            value={loadingHome ? "..." : stats?.edible ?? 0}
          />
          <StatCard
            label="Poisonous"
            value={loadingHome ? "..." : stats?.poisonous ?? 0}
          />
          <StatCard
            label="RF Accuracy"
            value={loadingHome ? "..." : `${Math.round((stats?.rf_accuracy ?? 0) * 100)}%`}
          />
          <StatCard
            label="DT Accuracy"
            value={loadingHome ? "..." : `${Math.round((stats?.dt_accuracy ?? 0) * 100)}%`}
          />
        </div>
      </section>

      <section className="max-w-5xl mx-auto px-6 pb-16">
        <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <form
            onSubmit={handleImagePredict}
            className="rounded-[2rem] border border-stone-200 bg-white p-6 shadow-sm"
          >
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.24em] text-stone-500">
                  Image Identification
                </p>
                <h2 className="mt-2 font-patua text-3xl text-stone-900">
                  Upload a mushroom photo.
                </h2>
              </div>
              <span
                className={`shrink-0 rounded-full px-3 py-1 text-xs font-semibold ${
                  imageModelReady
                    ? "bg-green-100 text-green-700"
                    : "bg-amber-100 text-amber-700"
                }`}
              >
                {imageModelReady ? "Model Ready" : "Needs Training"}
              </span>
            </div>

            <label className="mt-6 flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-stone-300 bg-stone-50 p-4 text-center transition hover:border-red-400">
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="Selected mushroom preview"
                  className="h-52 w-full rounded-2xl object-cover"
                />
              ) : (
                <>
                  <span className="font-patua text-xl text-stone-800">
                    Choose Image
                  </span>
                  <span className="mt-2 text-sm text-stone-500">
                    JPG, PNG, or WEBP mushroom photo
                  </span>
                </>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="sr-only"
              />
            </label>

            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={!imageFile || submittingImage || !imageModelReady}
                className="rounded-full bg-stone-950 px-6 py-3 text-sm font-semibold text-white transition hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submittingImage ? "Identifying..." : "Identify Mushroom"}
              </button>
              {!imageModelReady && (
                <span className="text-sm text-stone-500">
                  Train the image model before photo predictions are available.
                </span>
              )}
            </div>
          </form>

          <div className="rounded-[2rem] border border-stone-200 bg-[#fff7ef] p-6 shadow-sm">
            <p className="text-sm uppercase tracking-[0.24em] text-stone-500">
              Safety Result
            </p>
            {imagePrediction ? (
              <div className="mt-5">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <h3 className={`font-patua text-3xl capitalize ${
                      riskLabel === "edible"
                        ? "text-green-700"
                        : riskIsUncertain
                          ? "text-amber-700"
                          : "text-red-700"
                    }`}>
                      {riskLabel === "edible"
                        ? "Likely edible"
                        : riskIsUncertain
                          ? "Uncertain - avoid eating"
                          : "Avoid eating"}
                    </h3>
                    <p className="mt-2 text-sm text-stone-600">
                      Safety confidence:{" "}
                      {Math.round((imagePrediction.risk_confidence ?? 0) * 100)}%
                    </p>
                    {imagePrediction.risk_subtype && (
                      <p className="mt-1 text-sm capitalize text-stone-600">
                        Matched risk subtype: {imagePrediction.risk_subtype.replaceAll("_", " ")}
                      </p>
                    )}
                    {topSpeciesCandidate && (
                      <div className="mt-4 rounded-2xl border border-stone-200 bg-white px-4 py-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                          #1 Candidate
                        </p>
                        <p className="mt-1 font-patua text-2xl capitalize text-stone-900">
                          {topSpeciesCandidate.species}
                        </p>
                        <p className="mt-1 text-sm text-stone-600">
                          {Math.round(topSpeciesCandidate.confidence * 100)}% match ·{" "}
                          {topSpeciesCandidate.edibility.replaceAll("_", " ")}
                        </p>
                      </div>
                    )}
                  </div>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-semibold capitalize ${
                      riskLabel === "edible"
                        ? "bg-green-100 text-green-700"
                        : riskIsUncertain
                          ? "bg-amber-100 text-amber-700"
                          : "bg-red-100 text-red-700"
                    }`}
                  >
                    {riskLabel.replaceAll("_", " ")}
                  </span>
                </div>

                <p className="mt-5 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  Avoid eating unless confirmed by a qualified mushroom expert.
                </p>

                <div className="mt-6 space-y-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                    Possible Matches
                  </p>
                  {speciesSuggestions.map((prediction) => (
                    <div
                      key={prediction.species_key}
                      className="rounded-2xl bg-white px-4 py-3 text-sm"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <span className="font-semibold capitalize text-stone-800">
                          {prediction.species}
                        </span>
                        <span className="text-stone-500">
                          {Math.round(prediction.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <p className="mt-5 rounded-2xl border border-stone-200 bg-white px-4 py-3 text-sm text-stone-600">
                  {imagePrediction.warning}
                </p>
              </div>
            ) : (
              <div className="mt-5 rounded-3xl border border-dashed border-stone-300 bg-white/70 px-5 py-16 text-center text-sm text-stone-500">
                Uploaded image predictions will appear here after the image model is trained.
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="bg-stone-950 py-16 text-stone-100">
        <div className="max-w-5xl mx-auto px-6 grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="rounded-[2rem] bg-white/8 p-8 backdrop-blur">
            <p className="mb-3 text-sm uppercase tracking-[0.24em] text-red-200">
              Live Classifier
            </p>
            <h2 className="font-patua text-3xl leading-tight">
              Test the backend prediction output with real mined mushroom features.
            </h2>
            <p className="mt-4 text-sm leading-6 text-stone-300">
              Choose a few feature values from the dataset and the frontend will call both trained models so you can compare edible versus poisonous predictions.
            </p>

            <form onSubmit={handleClassify} className="mt-8 grid gap-4 md:grid-cols-2">
              {FEATURE_PICK_ORDER.map((featureName) => (
                <FeatureSelect
                  key={featureName}
                  featureName={featureName}
                  options={features[featureName] || []}
                  value={selectedFeatures[featureName] || ""}
                  onChange={handleFeatureChange}
                />
              ))}

              <div className="md:col-span-2 flex flex-wrap items-center gap-4 pt-2">
                <button
                  type="submit"
                  disabled={loadingHome || submitting}
                  className="rounded-full bg-red-500 px-7 py-3 text-sm font-semibold text-white transition hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {submitting ? "Classifying..." : "Run Classification"}
                </button>
                <span className="text-sm text-stone-300">
                  Uses Random Forest and Decision Tree outputs from the Flask API.
                </span>
              </div>
            </form>
          </div>

          <div className="rounded-[2rem] bg-[#fff7ef] p-8 text-stone-900 shadow-xl">
            <p className="mb-3 text-sm uppercase tracking-[0.24em] text-stone-500">
              Prediction Output
            </p>
            {classification ? (
              <div className="space-y-6">
                <div>
                  <h3 className="font-patua text-2xl">
                    Models agree: {classification.models_agree ? "Yes" : "No"}
                  </h3>
                  <p className="mt-2 text-sm text-stone-600">
                    Random Forest predicted{" "}
                    <span className="font-semibold">{classification.random_forest.classification}</span>
                    {" "}and Decision Tree predicted{" "}
                    <span className="font-semibold">{classification.decision_tree.classification}</span>.
                  </p>
                </div>

                <div className="grid gap-4">
                  <div className="rounded-3xl bg-white p-5">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Random Forest</p>
                    <p className="mt-2 font-patua text-2xl capitalize">
                      {classification.random_forest.classification}
                    </p>
                    <p className="mt-2 text-sm text-stone-600">
                      Poisonous confidence: {Math.round(classification.random_forest.confidence.poisonous * 100)}%
                    </p>
                  </div>

                  <div className="rounded-3xl bg-white p-5">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Decision Tree</p>
                    <p className="mt-2 font-patua text-2xl capitalize">
                      {classification.decision_tree.classification}
                    </p>
                    <p className="mt-2 text-sm text-stone-600">
                      Poisonous confidence: {Math.round(classification.decision_tree.confidence.poisonous * 100)}%
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-3xl border border-dashed border-stone-300 bg-white/70 px-5 py-10 text-center text-sm text-stone-500">
                Submit feature values to see the backend prediction output here.
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="bg-gray-50 py-16">
        <div className="max-w-5xl mx-auto px-6 grid md:grid-cols-3 gap-8 text-center">

          <div className="p-6 rounded-2xl bg-white shadow-sm hover:shadow-md transition">
            <h3 className="font-patua text-lg font-semibold mb-2">
              Data Mining Algorithms
            </h3>
            <p className="text-gray-600 text-sm">
              Our application uses Random Forests and Decision Trees to accurately classify mushrooms based on their physical features.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-white shadow-sm hover:shadow-md transition">
            <h3 className="font-patua text-lg font-semibold mb-2">
              Real-World Impact
            </h3>
            <p className="text-gray-600 text-sm">
              Correctly identifying poisonous mushrooms is critical. Our system helps prevent dangerous mistakes and ensures safety.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-white shadow-sm hover:shadow-md transition">
            <h3 className="font-patua text-lg font-semibold mb-2">
              Easy to Use
            </h3>
            <p className="text-gray-600 text-sm">
              Simply upload an image or input mushroom features, and get an instant classification using our backend API.
            </p>
          </div>

        </div>
      </section>

      <section className="max-w-5xl mx-auto px-6 py-16">
        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded-[2rem] border border-stone-200 bg-white p-8 shadow-sm">
            <p className="text-sm uppercase tracking-[0.24em] text-stone-500">
              Feature Signals
            </p>
            <h2 className="mt-3 font-patua text-3xl text-stone-900">
              Top attributes learned from the mushroom dataset.
            </h2>
            <div className="mt-6 space-y-4 text-sm text-stone-600">
              <p>
                Random Forest top feature:{" "}
                <span className="font-semibold text-stone-900">
                  {topRandomForestFeature?.feature || "Loading..."}
                </span>
              </p>
              <p>
                Decision Tree top feature:{" "}
                <span className="font-semibold text-stone-900">
                  {topDecisionTreeFeature?.feature || "Loading..."}
                </span>
              </p>
              <p>
                Total features mined by the backend:{" "}
                <span className="font-semibold text-stone-900">
                  {loadingHome ? "..." : stats?.total_features ?? 0}
                </span>
              </p>
            </div>
          </div>

          <div className="rounded-[2rem] border border-stone-200 bg-[#f8f3ea] p-8 shadow-sm">
            <p className="text-sm uppercase tracking-[0.24em] text-stone-500">
              Full Stack Flow
            </p>
            <h2 className="mt-3 font-patua text-3xl text-stone-900">
              Backend mining, frontend presentation.
            </h2>
            <p className="mt-4 text-sm leading-6 text-stone-700">
              The Flask backend reads the local mushroom dataset, trains the models, exposes edible and poisonous entries through the API, and the React frontend renders those results on the dedicated pages.
            </p>
          </div>
        </div>
      </section>

      <section className="py-20 text-center">
        <h2 className="text-3xl font-bold font-patua mb-4">
          Data Mining for Safer Decisions
        </h2>

        <p className="text-gray-600 mb-8">
          Mushroom Classifier guides users in identifying which mushrooms are safe to eat and which are toxic.
        </p>

        <Link
          to="/edible"
          className="px-8 py-3 bg-red-500 text-white rounded-full font-medium shadow-md hover:bg-red-600 transition"
        >
          Get Started
        </Link>
      </section>
    </div>
  );
}

function App() {
  return (
    <>
      <Navbar />

      <ScrollToTop /> 

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/edible" element={<Edible />} />
        <Route path="/poisonous" element={<Poisonous />} />
      </Routes>

      <Footer />
    </>
  );
}

export default App;
