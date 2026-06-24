"""
TruthLens — Feature Extractor
================================
Extracts TF-IDF, VADER sentiment, and linguistic features from text.
"""

import re
import logging
import numpy as np
from typing import Dict, Any

import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Sensationalist / clickbait word lists ─────────────────────────────────────
SENSATIONAL_WORDS = [
    "shocking", "unbelievable", "bombshell", "breaking", "urgent",
    "exclusive", "exposed", "scandal", "outrage", "conspiracy",
    "secret", "leaked", "revealed", "banned", "warning", "alert",
    "miracle", "cure", "hoax", "cover-up", "truth", "crisis"
]

CLICKBAIT_PATTERNS = [
    r"you won\'?t believe",
    r"this is (why|how)",
    r"what they (don\'?t|never) want you to know",
    r"\d+ (things|ways|reasons|tricks|hacks)",
    r"(doctors?|scientists?|experts?) hate",
    r"the truth about",
    r"find out why",
]

HEDGE_WORDS = [
    "allegedly", "reportedly", "claims", "sources say",
    "according to", "may", "might", "possibly", "could"
]

SOURCE_INDICATORS = [
    r"\baccording to\b",
    r"\breported by\b",
    r"\bsources (say|claim|state)\b",
    r"\bstudy (shows|finds|suggests)\b",
    r"\bdata (shows|indicates|reveals)\b",
    r"\bresearchers (found|say|claim)\b",
    r"https?://",
    r"\bwww\.\b",
]


class FeatureExtractor:
    """
    Unified feature extraction for TruthLens models.

    Provides TF-IDF vectorization and hand-crafted linguistic features.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size for TF-IDF.
    vectorizer_path : str or Path, optional
        Path to save/load the fitted TF-IDF vectorizer.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        vectorizer_path: str | Path | None = None
    ) -> None:
        self.max_features = max_features
        self.vectorizer_path = Path(vectorizer_path) if vectorizer_path else None
        self._tfidf: TfidfVectorizer | None = None

        # VADER sentiment analyzer
        self._vader = SentimentIntensityAnalyzer() if _VADER_AVAILABLE else None

    # ── TF-IDF ────────────────────────────────────────────────────────────────

    def fit_transform_tfidf(self, texts: list) -> Any:
        """
        Fit TF-IDF vectorizer on training texts and transform them.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        scipy sparse matrix
        """
        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )
        X = self._tfidf.fit_transform(texts)
        logger.info("TF-IDF fitted: %d features.", X.shape[1])
        return X

    def transform_tfidf(self, texts: list) -> Any:
        """
        Transform texts using the already-fitted TF-IDF vectorizer.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        scipy sparse matrix
        """
        if self._tfidf is None:
            raise RuntimeError("TF-IDF vectorizer is not fitted. Call fit_transform_tfidf first.")
        return self._tfidf.transform(texts)

    def save_tfidf(self, path: str | Path | None = None) -> Path:
        """Persist the TF-IDF vectorizer to disk with joblib."""
        out = Path(path) if path else self.vectorizer_path
        if out is None:
            raise ValueError("No path specified for saving TF-IDF vectorizer.")
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._tfidf, out)
        logger.info("TF-IDF vectorizer saved → %s", out)
        return out

    def load_tfidf(self, path: str | Path | None = None) -> None:
        """Load a previously saved TF-IDF vectorizer."""
        load_path = Path(path) if path else self.vectorizer_path
        if load_path is None or not load_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at: {load_path}")
        self._tfidf = joblib.load(load_path)
        logger.info("TF-IDF vectorizer loaded ← %s", load_path)

    # ── Linguistic features ───────────────────────────────────────────────────

    def extract_linguistic_features(self, text: str, title: str = "") -> Dict[str, float]:
        """
        Extract hand-crafted linguistic and stylistic features.

        Parameters
        ----------
        text : str
            Article body text.
        title : str
            Article headline (optional).

        Returns
        -------
        dict mapping feature name → float value
        """
        full_text = f"{title} {text}".strip()
        words = full_text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", full_text) if s.strip()]
        n_words = len(words) or 1
        n_sentences = len(sentences) or 1

        features: Dict[str, float] = {}

        # ── Basic counts ──────────────────────────────────────────────────────
        features["word_count"] = float(n_words)
        features["sentence_count"] = float(n_sentences)
        features["avg_sentence_len"] = n_words / n_sentences
        features["avg_word_len"] = (
            sum(len(w) for w in words) / n_words if words else 0.0
        )

        # ── Sensationalism score ───────────────────────────────────────────────
        lower_text = full_text.lower()
        sens_hits = sum(1 for w in SENSATIONAL_WORDS if w in lower_text)
        features["sensationalism_score"] = min(sens_hits / 5.0, 1.0)

        # ── Clickbait probability ──────────────────────────────────────────────
        clickbait_hits = sum(
            1 for p in CLICKBAIT_PATTERNS if re.search(p, lower_text, re.IGNORECASE)
        )
        features["clickbait_probability"] = min(clickbait_hits / 3.0, 1.0)

        # ── Exclamation and caps ratio ─────────────────────────────────────────
        features["exclamation_ratio"] = full_text.count("!") / n_sentences
        cap_words = sum(1 for w in words if len(w) > 2 and w.isupper())
        features["caps_ratio"] = cap_words / n_words

        # ── Hedge word density ─────────────────────────────────────────────────
        hedge_hits = sum(1 for h in HEDGE_WORDS if h in lower_text)
        features["hedge_density"] = hedge_hits / n_words

        # ── Source citation index ─────────────────────────────────────────────
        source_hits = sum(
            1 for p in SOURCE_INDICATORS if re.search(p, full_text, re.IGNORECASE)
        )
        features["source_citation_index"] = min(source_hits / 4.0, 1.0)

        # ── Passive voice ratio ────────────────────────────────────────────────
        passive = re.findall(
            r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", full_text, re.IGNORECASE
        )
        features["passive_voice_ratio"] = len(passive) / n_sentences

        # ── VADER sentiment ────────────────────────────────────────────────────
        if self._vader:
            scores = self._vader.polarity_scores(full_text)
            features["vader_pos"] = scores["pos"]
            features["vader_neg"] = scores["neg"]
            features["vader_neu"] = scores["neu"]
            features["vader_compound"] = scores["compound"]
            # Neutrality proxy: closer to 0 compound = more neutral
            features["sentiment_balance"] = 1.0 - abs(scores["compound"])
        else:
            features.update({
                "vader_pos": 0.0, "vader_neg": 0.0,
                "vader_neu": 1.0, "vader_compound": 0.0,
                "sentiment_balance": 1.0
            })

        # ── Headline–body cosine similarity ───────────────────────────────────
        features["headline_body_similarity"] = self._cosine_overlap(title, text)

        return features

    @staticmethod
    def _cosine_overlap(text_a: str, text_b: str) -> float:
        """Simple token overlap cosine similarity (no sklearn required)."""
        if not text_a or not text_b:
            return 0.0
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        return len(intersection) / (len(set_a) ** 0.5 * len(set_b) ** 0.5 + 1e-9)

    def get_radar_scores(self, text: str, title: str = "") -> Dict[str, float]:
        """
        Return the 6-axis radar chart scores (0-100 scale each).

        Axes: Sensationalism, Emotional Bias, Clickbait Language,
              Source Density, Claim Frequency, Complexity

        Parameters
        ----------
        text : str
        title : str

        Returns
        -------
        dict
        """
        feats = self.extract_linguistic_features(text, title)

        # Claim frequency: count assertive verbs
        claims_pattern = r"\b(claim|assert|allege|insist|argue|maintain|state)\b"
        claims = len(re.findall(claims_pattern, text, re.IGNORECASE))
        claim_freq = min(claims / 3.0, 1.0)

        # Complexity: avg word length normalised
        complexity = min(feats.get("avg_word_len", 4.0) / 8.0, 1.0)

        return {
            "Sensationalism":    round(feats["sensationalism_score"] * 100, 1),
            "Emotional Bias":    round((1 - feats["sentiment_balance"]) * 100, 1),
            "Clickbait Language": round(feats["clickbait_probability"] * 100, 1),
            "Source Density":    round(feats["source_citation_index"] * 100, 1),
            "Claim Frequency":   round(claim_freq * 100, 1),
            "Complexity":        round(complexity * 100, 1),
        }
