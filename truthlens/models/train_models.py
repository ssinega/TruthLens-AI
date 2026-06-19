"""
TruthLens — Model Training Script
====================================
Trains all required ML models and saves them to models/trained/.

Algorithms trained:
    Supervised:
        1. Logistic Regression (TF-IDF)
        2. Random Forest Classifier
        3. Passive Aggressive Classifier
        4. LinearSVC
        5. Soft Voting Ensemble (1-4)

    Unsupervised:
        6. K-Means Clustering (topic cluster map)
        7. Isolation Forest (anomaly detection)

Usage:
    cd truthlens/
    python models/train_models.py
    python models/train_models.py --dataset isot
    python models/train_models.py --dataset liar
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models" / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset registry (mirrors data/download_data.py DATASETS dict) ────────────
DATASET_REGISTRY = {
    "welfake": {
        "csv":      ROOT / "data" / "welfake_dataset.csv",
        "display":  "WELFake",
    },
    "isot": {
        "csv":      ROOT / "data" / "isot_dataset.csv",
        "display":  "ISOT Fake/Real News",
    },
    "liar": {
        "csv":      ROOT / "data" / "liar_dataset.csv",
        "display":  "LIAR Benchmark",
    },
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("truthlens.train")

# ── Add project root to sys.path ───────────────────────────────────────────────
sys.path.insert(0, str(ROOT))

from utils.feature_extractor import FeatureExtractor
from utils.text_preprocessor import TextPreprocessor
from data.download_data import load_dataset_csv, generate_synthetic_dataset, save_dataset


# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_TFIDF_FEATURES = 10_000
CLUSTER_N = 4           # K-Means clusters for topic map
CLUSTER_SAMPLE = 500    # Articles to use for the cluster map


def load_or_generate_data(dataset_name: str = "welfake") -> pd.DataFrame:
    """
    Load the specified dataset CSV or generate synthetic data as fallback.

    Parameters
    ----------
    dataset_name : str
        One of 'welfake', 'isot', 'liar'.

    Returns
    -------
    pd.DataFrame with columns: text, title, label
    """
    info = DATASET_REGISTRY.get(dataset_name, DATASET_REGISTRY["welfake"])
    data_path = info["csv"]

    if data_path.exists():
        logger.info(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path).dropna(subset=["text"])
    else:
        logger.warning(
            f"Dataset CSV not found at {data_path}. "
            f"Run: python data/download_data.py --dataset {dataset_name}"
        )
        logger.warning("Falling back to synthetic data (2 000 samples)...")
        df = generate_synthetic_dataset(n_samples=2000)
        save_dataset(df, data_path)

    # Binary labels for models that need 2-class output
    # Map: 0=Fake, 1=Suspicious, 2=Real → binary: 0=Fake, 1=Real
    df["label_binary"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

    logger.info(f"Dataset loaded: {len(df):,} records.")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text using TextPreprocessor.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with 'cleaned_text' column added
    """
    logger.info("Preprocessing text...")
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    df["cleaned_text"] = df["text"].apply(preprocessor.clean)
    df = df[df["cleaned_text"].str.len() > 5].reset_index(drop=True)
    logger.info(f"After preprocessing: {len(df)} valid records.")
    return df


def train_supervised_models(
    X_train, X_test, y_train, y_test
) -> dict:
    """
    Train all supervised classifiers and return fitted models + metrics.

    Parameters
    ----------
    X_train, X_test : scipy sparse matrices (TF-IDF features)
    y_train, y_test : array-like of int labels

    Returns
    -------
    dict mapping model_name → {'model': fitted_estimator, 'accuracy': float, 'f1': float}
    """
    results = {}

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    logger.info("[1/5] Training Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(
        C=1.0, solver="liblinear", max_iter=1000,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    results["logistic_regression"] = {
        "model": lr,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds, average="weighted"), 4),
        "fit_time_s": round(time.time() - t0, 2),
    }
    logger.info(f"  Accuracy: {results['logistic_regression']['accuracy']:.4f} | "
                f"F1: {results['logistic_regression']['f1']:.4f} | "
                f"Time: {results['logistic_regression']['fit_time_s']}s")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    logger.info("[2/5] Training Random Forest...")
    t0 = time.time()
    try:
        # RF needs dense matrix for some configs; use sparse-compatible params
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=20, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        results["random_forest"] = {
            "model": rf,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds, average="weighted"), 4),
            "fit_time_s": round(time.time() - t0, 2),
        }
        logger.info(f"  Accuracy: {results['random_forest']['accuracy']:.4f} | "
                    f"F1: {results['random_forest']['f1']:.4f}")
    except Exception as e:
        logger.error(f"Random Forest failed: {e}")
        results["random_forest"] = {"model": None, "accuracy": 0.0, "f1": 0.0, "fit_time_s": 0.0}

    # ── 3. Passive Aggressive ─────────────────────────────────────────────────
    logger.info("[3/5] Training Passive Aggressive Classifier...")
    t0 = time.time()
    pac = PassiveAggressiveClassifier(
        C=1.0, max_iter=1000, tol=1e-3,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    pac.fit(X_train, y_train)
    preds = pac.predict(X_test)
    results["passive_aggressive"] = {
        "model": pac,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds, average="weighted"), 4),
        "fit_time_s": round(time.time() - t0, 2),
    }
    logger.info(f"  Accuracy: {results['passive_aggressive']['accuracy']:.4f} | "
                f"F1: {results['passive_aggressive']['f1']:.4f}")

    # ── 4. LinearSVC ──────────────────────────────────────────────────────────
    logger.info("[4/5] Training LinearSVC...")
    t0 = time.time()
    svc = LinearSVC(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )
    svc.fit(X_train, y_train)
    preds = svc.predict(X_test)
    results["linear_svc"] = {
        "model": svc,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds, average="weighted"), 4),
        "fit_time_s": round(time.time() - t0, 2),
    }
    logger.info(f"  Accuracy: {results['linear_svc']['accuracy']:.4f} | "
                f"F1: {results['linear_svc']['f1']:.4f}")

    # ── 5. Soft Voting Ensemble ────────────────────────────────────────────────
    logger.info("[5/5] Training Soft Voting Ensemble...")
    t0 = time.time()
    try:
        # VotingClassifier needs predict_proba; LinearSVC doesn't support it natively
        # Use LR + PAC (with calibration via CalibratedClassifierCV) + RF
        from sklearn.calibration import CalibratedClassifierCV

        # Calibrate PAC for probability outputs
        pac_cal = CalibratedClassifierCV(
            PassiveAggressiveClassifier(C=1.0, max_iter=1000, class_weight="balanced",
                                        random_state=RANDOM_STATE),
            cv=3
        )
        svc_cal = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )

        ensemble = VotingClassifier(
            estimators=[
                ("lr",  LogisticRegression(C=1.0, solver="liblinear", max_iter=1000,
                                           class_weight="balanced", random_state=RANDOM_STATE)),
                ("rf",  RandomForestClassifier(n_estimators=50, max_depth=15,
                                               class_weight="balanced", n_jobs=-1,
                                               random_state=RANDOM_STATE)),
                ("pac", pac_cal),
                ("svc", svc_cal),
            ],
            voting="soft",
            weights=[0.35, 0.25, 0.20, 0.20],
            n_jobs=-1
        )
        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_test)
        results["voting_ensemble"] = {
            "model": ensemble,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds, average="weighted"), 4),
            "fit_time_s": round(time.time() - t0, 2),
        }
        logger.info(f"  Accuracy: {results['voting_ensemble']['accuracy']:.4f} | "
                    f"F1: {results['voting_ensemble']['f1']:.4f}")
    except Exception as e:
        logger.error(f"Voting Ensemble failed: {e}")
        results["voting_ensemble"] = {"model": None, "accuracy": 0.0, "f1": 0.0, "fit_time_s": 0.0}

    return results


def train_unsupervised_models(X_tfidf, df: pd.DataFrame) -> dict:
    """
    Train K-Means clustering and Isolation Forest for unsupervised analysis.

    Parameters
    ----------
    X_tfidf : sparse matrix (TF-IDF features for a sample of articles)
    df : pd.DataFrame (sampled subset with 'label' column)

    Returns
    -------
    dict with fitted models and PCA-reduced cluster coordinates
    """
    from sklearn.cluster import KMeans
    results = {}

    # Sample for cluster map efficiency
    sample_size = min(CLUSTER_SAMPLE, X_tfidf.shape[0])
    sample_idx = np.random.choice(X_tfidf.shape[0], sample_size, replace=False)

    if hasattr(X_tfidf, "toarray"):
        X_sample = X_tfidf[sample_idx].toarray()
    else:
        X_sample = X_tfidf[sample_idx]

    y_sample = np.array(df["label_binary"].tolist())[sample_idx]

    # ── K-Means ───────────────────────────────────────────────────────────────
    logger.info(f"[Unsupervised] Training K-Means (k={CLUSTER_N})...")
    kmeans = KMeans(n_clusters=CLUSTER_N, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_sample)

    # PCA 2D for visualization
    logger.info("[Unsupervised] Running PCA for cluster visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sample)

    cluster_data = {
        "x": X_pca[:, 0].tolist(),
        "y": X_pca[:, 1].tolist(),
        "cluster": cluster_labels.tolist(),
        "true_label": y_sample.tolist(),
        "text_preview": df["text"].iloc[sample_idx].str[:60].tolist(),
    }

    results["kmeans"] = {
        "model": kmeans,
        "pca": pca,
        "cluster_data": cluster_data,
    }
    logger.info(f"  K-Means: {CLUSTER_N} clusters found.")

    # ── Isolation Forest ──────────────────────────────────────────────────────
    logger.info("[Unsupervised] Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.1,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    iso_forest.fit(X_sample)
    results["isolation_forest"] = {"model": iso_forest}
    logger.info("  Isolation Forest trained.")

    return results


def save_models(supervised: dict, unsupervised: dict, vectorizer) -> dict:
    """
    Persist all trained models to disk using joblib.

    Parameters
    ----------
    supervised : dict of model results from train_supervised_models
    unsupervised : dict of model results from train_unsupervised_models
    vectorizer : fitted TfidfVectorizer

    Returns
    -------
    dict mapping model name → saved file path
    """
    saved_paths = {}

    # Save TF-IDF vectorizer
    vect_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vect_path)
    saved_paths["tfidf_vectorizer"] = str(vect_path)
    logger.info(f"Saved TF-IDF vectorizer → {vect_path}")

    # Save supervised models
    for name, result in supervised.items():
        if result.get("model") is not None:
            path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(result["model"], path)
            saved_paths[name] = str(path)
            logger.info(f"Saved {name} → {path}")

    # Save K-Means and PCA
    if "kmeans" in unsupervised:
        km_path = MODELS_DIR / "kmeans.pkl"
        pca_path = MODELS_DIR / "pca.pkl"
        cluster_data_path = MODELS_DIR / "cluster_data.json"

        joblib.dump(unsupervised["kmeans"]["model"], km_path)
        joblib.dump(unsupervised["kmeans"]["pca"], pca_path)
        with open(cluster_data_path, "w") as f:
            json.dump(unsupervised["kmeans"]["cluster_data"], f)

        saved_paths["kmeans"] = str(km_path)
        saved_paths["pca"] = str(pca_path)
        saved_paths["cluster_data"] = str(cluster_data_path)
        logger.info(f"Saved K-Means → {km_path}")

    # Save Isolation Forest
    if "isolation_forest" in unsupervised:
        iso_path = MODELS_DIR / "isolation_forest.pkl"
        joblib.dump(unsupervised["isolation_forest"]["model"], iso_path)
        saved_paths["isolation_forest"] = str(iso_path)
        logger.info(f"Saved Isolation Forest → {iso_path}")

    return saved_paths


def save_metadata(
    supervised: dict,
    saved_paths: dict,
    dataset_name: str = "welfake",
    dataset_size: int = 0,
) -> None:
    """Save training metadata JSON for the app to display."""
    info = DATASET_REGISTRY.get(dataset_name, DATASET_REGISTRY["welfake"])
    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": info["display"],
        "dataset_key": dataset_name,
        "dataset_size": dataset_size,
        "models": {
            name: {
                "accuracy": res.get("accuracy", 0.0),
                "f1": res.get("f1", 0.0),
                "fit_time_s": res.get("fit_time_s", 0.0),
                "saved_at": saved_paths.get(name, ""),
            }
            for name, res in supervised.items()
        },
        "tfidf_features": MAX_TFIDF_FEATURES,
        "cluster_n": CLUSTER_N,
    }
    meta_path = MODELS_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved → {meta_path}")


def run_training_pipeline(dataset_name: str = "welfake") -> None:
    """
    Execute the full TruthLens model training pipeline.

    Parameters
    ----------
    dataset_name : str
        Dataset to train on: 'welfake' | 'isot' | 'liar'.

    Steps
    -----
    1. Load / generate dataset
    2. Preprocess text
    3. TF-IDF feature extraction
    4. Train supervised models (LR, RF, PAC, SVC, Ensemble)
    5. Train unsupervised models (K-Means, Isolation Forest)
    6. Save all models and metadata to disk
    """
    logger.info("=" * 60)
    logger.info("  TruthLens AI — Model Training Pipeline")
    logger.info(f"  Dataset: {DATASET_REGISTRY.get(dataset_name, {}).get('display', dataset_name)}")
    logger.info("  IBM Edunet AI Internship 2025")
    logger.info("=" * 60)
    pipeline_start = time.time()

    # ── Step 1: Load Data ──────────────────────────────────────────────────────
    df = load_or_generate_data(dataset_name)
    total_samples = len(df)

    # ── Step 2: Preprocess ─────────────────────────────────────────────────────
    df = preprocess_data(df)

    # ── Step 3: TF-IDF Feature Extraction ─────────────────────────────────────
    logger.info("Fitting TF-IDF Vectorizer...")
    fe = FeatureExtractor(max_features=MAX_TFIDF_FEATURES)
    texts = df["cleaned_text"].tolist()
    labels = df["label_binary"].tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    X_train = fe.fit_transform_tfidf(X_train_raw)
    X_test = fe.transform_tfidf(X_test_raw)
    logger.info(f"TF-IDF: {X_train.shape[1]} features, {X_train.shape[0]} training samples.")

    # ── Step 4: Train Supervised Models ────────────────────────────────────────
    supervised = train_supervised_models(X_train, X_test, y_train, y_test)

    # ── Step 5: Train Unsupervised Models ──────────────────────────────────────
    X_all = fe.transform_tfidf(texts)
    df_sample = df.iloc[:len(texts)].reset_index(drop=True)
    unsupervised = train_unsupervised_models(X_all, df_sample)

    # ── Step 6: Save Everything ────────────────────────────────────────────────
    logger.info("Saving all models to disk...")
    saved_paths = save_models(supervised, unsupervised, fe._tfidf)
    save_metadata(supervised, saved_paths, dataset_name=dataset_name, dataset_size=total_samples)

    # ── Summary Leaderboard ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  MODEL PERFORMANCE LEADERBOARD")
    logger.info("=" * 60)
    leaderboard = sorted(
        [(name, res) for name, res in supervised.items() if res.get("model")],
        key=lambda x: x[1]["f1"],
        reverse=True
    )
    for rank, (name, res) in enumerate(leaderboard, 1):
        logger.info(f"  #{rank} {name:<25} Acc: {res['accuracy']:.4f}  F1: {res['f1']:.4f}")

    total_time = round(time.time() - pipeline_start, 1)
    logger.info("=" * 60)
    logger.info(f"  Training complete in {total_time}s. Models saved to: {MODELS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens Model Trainer")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="welfake",
        help="Dataset to train on (default: welfake)",
    )
    args = parser.parse_args()
    run_training_pipeline(dataset_name=args.dataset)
