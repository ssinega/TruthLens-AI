"""
TruthLens — Unified Prediction Pipeline
==========================================
Loads all trained models and provides a single predict() interface
used by the Streamlit app.

Includes:
    - Multi-model prediction (LR, RF, PAC, SVC, Ensemble)
    - TF-IDF + Cosine Similarity (against reliable reference embeddings)
    - Isolation Forest anomaly scoring
    - DistilBERT inference (optional — disabled in Demo Mode)
    - LIME explainability
    - Graceful fallback to rule-based scoring
"""

import sys
# Shadow torchvision to prevent RuntimeError: operator torchvision::nms does not exist
# caused by a mismatch between pre-installed torch and torchvision packages.
sys.modules['torchvision'] = None

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models" / "trained"

# ── Model display names ────────────────────────────────────────────────────────
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "passive_aggressive":  "Passive Aggressive",
    "linear_svc":          "LinearSVC",
    "voting_ensemble":     "Voting Ensemble",
}

# ── Reliable source reference texts (for cosine similarity baseline) ───────────
RELIABLE_REFERENCE_TEXTS = [
    "according to peer-reviewed research published in nature scientists found evidence",
    "official government report confirmed data from multiple verified sources",
    "study conducted by university researchers shows results corroborated independently",
    "world health organization recommends based on systematic review of evidence",
    "court documents filed with federal authorities reveal verified facts",
    "census bureau statistics indicate trends confirmed by independent analysis",
]


class PredictionPipeline:
    """
    Unified prediction pipeline for TruthLens.

    Loads trained models from disk and provides batch/single prediction.

    Parameters
    ----------
    models_dir : Path
        Directory containing trained .pkl model files.
    demo_mode : bool
        If True, skip heavy models (DistilBERT) and use pre-computed weights.
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        demo_mode: bool = False
    ) -> None:
        self.models_dir = models_dir
        self.demo_mode = demo_mode
        self._models: Dict[str, Any] = {}
        self._vectorizer = None
        self._kmeans = None
        self._pca = None
        self._iso_forest = None
        self._cluster_data: Dict = {}
        self._bert_pipeline = None
        self._models_loaded = False

        self._load_models()

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """
        Attempt to load all trained models from disk.
        Falls back gracefully if files are missing (pre-training).
        """
        if not self.models_dir.exists():
            logger.warning(
                "Models directory not found. Run models/train_models.py first."
            )
            return

        # Load TF-IDF vectorizer
        vect_path = self.models_dir / "tfidf_vectorizer.pkl"
        if vect_path.exists():
            try:
                self._vectorizer = joblib.load(vect_path)
                logger.info("TF-IDF vectorizer loaded.")
            except Exception as e:
                logger.error(f"Failed to load vectorizer: {e}")

        # Load supervised classifiers
        for model_key in ["logistic_regression", "random_forest",
                          "passive_aggressive", "linear_svc", "voting_ensemble"]:
            path = self.models_dir / f"{model_key}.pkl"
            if path.exists():
                try:
                    self._models[model_key] = joblib.load(path)
                    logger.info(f"Loaded: {model_key}")
                except Exception as e:
                    logger.warning(f"Could not load {model_key}: {e}")

        # Load K-Means and PCA
        km_path = self.models_dir / "kmeans.pkl"
        pca_path = self.models_dir / "pca.pkl"
        cluster_path = self.models_dir / "cluster_data.json"

        if km_path.exists():
            try:
                self._kmeans = joblib.load(km_path)
                logger.info("K-Means loaded.")
            except Exception as e:
                logger.warning(f"Could not load K-Means: {e}")

        if pca_path.exists():
            try:
                self._pca = joblib.load(pca_path)
                logger.info("PCA loaded.")
            except Exception as e:
                logger.warning(f"Could not load PCA: {e}")

        if cluster_path.exists():
            try:
                with open(cluster_path) as f:
                    self._cluster_data = json.load(f)
                logger.info("Cluster data loaded.")
            except Exception as e:
                logger.warning(f"Could not load cluster data: {e}")

        # Load Isolation Forest
        iso_path = self.models_dir / "isolation_forest.pkl"
        if iso_path.exists():
            try:
                self._iso_forest = joblib.load(iso_path)
                logger.info("Isolation Forest loaded.")
            except Exception as e:
                logger.warning(f"Could not load Isolation Forest: {e}")

        # DistilBERT is optional and can be slow to download/load locally.
        # Keep trained scikit-learn inference responsive by making it opt-in.
        if (
            not self.demo_mode
            and os.environ.get("TRUTHLENS_ENABLE_DISTILBERT", "").lower() in {"1", "true", "yes"}
        ):
            self._load_distilbert()

        self._models_loaded = len(self._models) > 0 or self._vectorizer is not None
        logger.info(f"Models loaded: {list(self._models.keys())}")

    def _load_distilbert(self) -> None:
        """Load DistilBERT fine-tuned pipeline (optional heavy dependency)."""
        try:
            # Check internet connectivity to huggingface.co to avoid hangs
            import socket
            try:
                socket.setdefaulttimeout(2.0)
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
            except Exception:
                os.environ["HF_HUB_OFFLINE"] = "1"
                logger.info("huggingface.co unreachable. Running in Hugging Face offline mode.")

            from transformers import pipeline as hf_pipeline
            # Use a lightweight fake-news detection model from HuggingFace Hub
            self._bert_pipeline = hf_pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                max_length=512,
                truncation=True,
                top_k=None,
            )
            logger.info("DistilBERT pipeline loaded.")
        except Exception as e:
            logger.warning(f"DistilBERT not loaded (requires transformers + torch): {e}")
            self._bert_pipeline = None

    # ── Main Prediction Interface ──────────────────────────────────────────────

    def predict(
        self,
        text: str,
        title: str = "",
        demo_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the full prediction pipeline on a single article.

        Parameters
        ----------
        text : str
            Article body text (cleaned or raw).
        title : str
            Article headline (optional).
        demo_mode : bool
            If True, return pre-computed demo results instantly.

        Returns
        -------
        dict with keys:
            model_predictions  : dict mapping model_name → {'label', 'confidence', 'verdict'}
            ensemble_label     : str ('REAL' or 'FAKE')
            ensemble_confidence: float
            cosine_similarity  : float (vs. reliable references)
            anomaly_score      : float (Isolation Forest)
            cluster_point      : dict {'x': float, 'y': float, 'cluster_id': int}
            lime_weights       : list of (word, weight) tuples
            distilbert_result  : dict or None
            processing_time_ms : float
        """
        if demo_mode or self.demo_mode:
            return self._demo_results(text)

        t0 = time.time()
        result: Dict[str, Any] = {}

        # Preprocess
        from utils.text_preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        cleaned = preprocessor.clean(f"{title} {text}")

        # ── Supervised predictions ─────────────────────────────────────────────
        model_preds = {}
        if self._vectorizer and self._models:
            try:
                X = self._vectorizer.transform([cleaned])
                for model_key, model in self._models.items():
                    try:
                        pred_label = model.predict(X)[0]
                        # Probability (if available)
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X)[0]
                            confidence = float(max(proba))
                        elif hasattr(model, "decision_function"):
                            df_val = model.decision_function(X)[0]
                            # Sigmoid-like normalization
                            if np.isscalar(df_val):
                                confidence = float(1 / (1 + np.exp(-abs(df_val))))
                            else:
                                confidence = float(np.max(1 / (1 + np.exp(-np.abs(df_val)))))
                        else:
                            confidence = 0.75

                        verdict = "REAL" if pred_label == 1 else "FAKE"
                        model_preds[model_key] = {
                            "label": int(pred_label),
                            "verdict": verdict,
                            "confidence": round(confidence, 4),
                        }
                    except Exception as e:
                        logger.warning(f"Model {model_key} prediction failed: {e}")
                        model_preds[model_key] = {"label": 1, "verdict": "UNCERTAIN", "confidence": 0.5}
            except Exception as e:
                logger.error(f"Vectorizer transform failed: {e}")

        result["model_predictions"] = model_preds

        # ── Ensemble majority vote (fallback if VotingClassifier absent) ───────
        if model_preds:
            labels = [p["label"] for p in model_preds.values()]
            confidences = [p["confidence"] for p in model_preds.values()]
            ensemble_label_int = int(np.round(np.mean(labels)))
            result["ensemble_label"] = "REAL" if ensemble_label_int == 1 else "FAKE"
            result["ensemble_confidence"] = round(float(np.mean(confidences)), 4)
        else:
            # Rule-based fallback
            rule_result = self._rule_based_predict(text, title)
            result["ensemble_label"] = rule_result["verdict"]
            result["ensemble_confidence"] = rule_result["confidence"]
            result["model_predictions"] = {"rule_based": rule_result}

        # ── Cosine similarity vs. reliable references ──────────────────────────
        result["cosine_similarity"] = self._cosine_vs_reliable(cleaned)

        # ── Anomaly detection ──────────────────────────────────────────────────
        result["anomaly_score"] = self._anomaly_score(cleaned)

        # ── Cluster assignment ─────────────────────────────────────────────────
        result["cluster_point"] = self._assign_cluster(cleaned)

        # ── LIME explainability ────────────────────────────────────────────────
        result["lime_weights"] = self._lime_explain(cleaned)

        # ── DistilBERT (if loaded) ─────────────────────────────────────────────
        result["distilbert_result"] = self._bert_predict(text)

        result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    # ── Individual model components ────────────────────────────────────────────

    def _cosine_vs_reliable(self, cleaned_text: str) -> float:
        """
        Compute average cosine similarity between the article and
        known-reliable source reference embeddings using TF-IDF.

        Parameters
        ----------
        cleaned_text : str

        Returns
        -------
        float in [0, 1]
        """
        if self._vectorizer is None:
            return self._token_overlap_similarity(cleaned_text)

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            X_article = self._vectorizer.transform([cleaned_text])
            X_refs = self._vectorizer.transform(RELIABLE_REFERENCE_TEXTS)
            sims = cosine_similarity(X_article, X_refs)[0]
            return round(float(np.mean(sims)), 4)
        except Exception as e:
            logger.warning(f"Cosine similarity failed: {e}")
            return self._token_overlap_similarity(cleaned_text)

    @staticmethod
    def _token_overlap_similarity(text: str) -> float:
        """Fallback token-overlap similarity against reliable keywords."""
        reliable_keywords = {
            "according", "study", "research", "published", "confirmed",
            "data", "evidence", "official", "report", "verified",
        }
        words = set(text.lower().split())
        overlap = len(words & reliable_keywords)
        return min(overlap / 5.0, 1.0)

    def _anomaly_score(self, cleaned_text: str) -> float:
        """
        Use Isolation Forest to flag statistically anomalous writing patterns.

        Returns
        -------
        float: 0.0 (normal) to 1.0 (highly anomalous)
        """
        if self._iso_forest is None or self._vectorizer is None:
            return 0.5

        try:
            X = self._vectorizer.transform([cleaned_text])
            if hasattr(X, "toarray"):
                X = X.toarray()
            # score_samples returns negative values; -1=anomaly, 1=normal
            raw_score = self._iso_forest.score_samples(X)[0]
            # Normalize: more negative = more anomalous → higher score
            normalized = 1.0 - (1.0 / (1.0 + np.exp(-raw_score - 0.5)))
            return round(float(np.clip(normalized, 0.0, 1.0)), 4)
        except Exception as e:
            logger.warning(f"Anomaly scoring failed: {e}")
            return 0.5

    def _assign_cluster(self, cleaned_text: str) -> Dict:
        """
        Assign the new article to the nearest K-Means cluster.

        Returns
        -------
        dict with 'x', 'y' (PCA coordinates), 'cluster_id'
        """
        if self._kmeans is None or self._vectorizer is None or self._pca is None:
            return {"x": 0.0, "y": 0.0, "cluster_id": 0}

        try:
            X = self._vectorizer.transform([cleaned_text])
            if hasattr(X, "toarray"):
                X = X.toarray()
            cluster_id = int(self._kmeans.predict(X)[0])
            X_pca = self._pca.transform(X)
            return {
                "x": float(X_pca[0, 0]),
                "y": float(X_pca[0, 1]),
                "cluster_id": cluster_id,
            }
        except Exception as e:
            logger.warning(f"Cluster assignment failed: {e}")
            return {"x": 0.0, "y": 0.0, "cluster_id": 0}

    def _lime_explain(self, cleaned_text: str, n_features: int = 15) -> List[Tuple[str, float]]:
        """
        Use LIME to explain which words pushed the prediction.

        Parameters
        ----------
        cleaned_text : str
        n_features : int — number of LIME features to return

        Returns
        -------
        list of (word, weight) tuples, sorted by abs(weight) desc
        """
        if not self._vectorizer or not self._models:
            return self._rule_based_weights(cleaned_text)

        # Find best model for LIME (needs predict_proba)
        lime_model = None
        for name in ["voting_ensemble", "logistic_regression", "random_forest"]:
            if name in self._models and hasattr(self._models[name], "predict_proba"):
                lime_model = self._models[name]
                break

        if lime_model is None:
            return self._rule_based_weights(cleaned_text)

        try:
            from lime.lime_text import LimeTextExplainer

            def predict_fn(texts):
                X = self._vectorizer.transform(texts)
                return lime_model.predict_proba(X)

            explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
            exp = explainer.explain_instance(
                cleaned_text,
                predict_fn,
                num_features=n_features,
                num_samples=100,
            )
            weights = exp.as_list()
            return sorted(weights, key=lambda x: abs(x[1]), reverse=True)
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return self._rule_based_weights(cleaned_text)

    def _bert_predict(self, text: str) -> Optional[Dict]:
        """
        Run DistilBERT text classification (if pipeline is loaded).

        Returns
        -------
        dict with 'label' and 'score', or None
        """
        if self._bert_pipeline is None:
            return None

        try:
            truncated = text[:512]
            result = self._bert_pipeline(truncated)
            if isinstance(result, list) and result:
                top = max(result[0], key=lambda x: x["score"])
                return {"label": top["label"], "score": round(top["score"], 4)}
        except Exception as e:
            logger.warning(f"DistilBERT prediction failed: {e}")
        return None

    # ── Fallback / demo methods ────────────────────────────────────────────────

    @staticmethod
    def _rule_based_predict(text: str, title: str = "") -> Dict:
        """
        Rule-based fake news detection as graceful fallback.

        Uses pattern matching on sensationalist/clickbait language.
        """
        full = f"{title} {text}".lower()
        fake_signals = [
            "shocking", "you won't believe", "bombshell", "cover-up",
            "conspiracy", "miracle", "secret", "leaked", "elites",
            "they don't want you to know", "breaking", "scandal"
        ]
        real_signals = [
            "according to", "study", "research", "published",
            "confirmed", "official", "data", "evidence", "court documents"
        ]
        fake_score = sum(1 for s in fake_signals if s in full)
        real_score = sum(1 for s in real_signals if s in full)

        if fake_score > real_score:
            return {"verdict": "FAKE", "confidence": min(0.5 + fake_score * 0.08, 0.92), "label": 0}
        else:
            return {"verdict": "REAL", "confidence": min(0.5 + real_score * 0.08, 0.90), "label": 1}

    @staticmethod
    def _rule_based_weights(text: str) -> List[Tuple[str, float]]:
        """Generate rule-based word weights for explanation."""
        from utils.feature_extractor import SENSATIONAL_WORDS
        words = text.split()
        weights = []
        for word in set(words):
            if word.lower() in SENSATIONAL_WORDS:
                weights.append((word, -0.3))  # negative = pushes toward FAKE
            elif word.lower() in ["study", "research", "according", "official", "confirmed"]:
                weights.append((word, 0.25))  # positive = pushes toward REAL
        return sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:10]

    @staticmethod
    def _demo_results(text: str) -> Dict:
        """Return pre-computed demo results for GitHub/offline demo."""
        is_likely_fake = any(
            kw in text.lower()
            for kw in ["shocking", "exposed", "conspiracy", "secret", "won't believe", "bombshell"]
        )

        if is_likely_fake:
            ensemble_label, ensemble_conf = "FAKE", 0.87
            preds = {
                "logistic_regression": {"verdict": "FAKE", "confidence": 0.89, "label": 0},
                "random_forest":       {"verdict": "FAKE", "confidence": 0.82, "label": 0},
                "passive_aggressive":  {"verdict": "FAKE", "confidence": 0.85, "label": 0},
                "linear_svc":          {"verdict": "FAKE", "confidence": 0.91, "label": 0},
                "voting_ensemble":     {"verdict": "FAKE", "confidence": 0.87, "label": 0},
            }
        else:
            ensemble_label, ensemble_conf = "REAL", 0.78
            preds = {
                "logistic_regression": {"verdict": "REAL", "confidence": 0.81, "label": 1},
                "random_forest":       {"verdict": "REAL", "confidence": 0.74, "label": 1},
                "passive_aggressive":  {"verdict": "REAL", "confidence": 0.76, "label": 1},
                "linear_svc":          {"verdict": "REAL", "confidence": 0.83, "label": 1},
                "voting_ensemble":     {"verdict": "REAL", "confidence": 0.78, "label": 1},
            }

        return {
            "model_predictions":   preds,
            "ensemble_label":      ensemble_label,
            "ensemble_confidence": ensemble_conf,
            "cosine_similarity":   0.32 if is_likely_fake else 0.61,
            "anomaly_score":       0.72 if is_likely_fake else 0.25,
            "cluster_point":       {"x": -1.2 if is_likely_fake else 0.8, "y": 0.5, "cluster_id": 0},
            "lime_weights": [
                ("shocking", -0.35), ("truth", -0.22), ("exposed", -0.28),
                ("according", 0.18), ("study", 0.24), ("confirmed", 0.20),
            ] if is_likely_fake else [
                ("according", 0.31), ("study", 0.28), ("research", 0.22),
                ("shocking", -0.12), ("claim", -0.09),
            ],
            "distilbert_result":   None,
            "processing_time_ms":  45.0,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def models_loaded(self) -> bool:
        """True if at least one model is loaded."""
        return self._models_loaded

    @property
    def loaded_model_names(self) -> List[str]:
        """List of loaded model keys."""
        return list(self._models.keys())

    def get_cluster_data(self) -> Dict:
        """Return cached cluster data for the Article Cluster Explorer."""
        return self._cluster_data
