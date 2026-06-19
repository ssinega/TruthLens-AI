"""
TruthLens — Credibility Scorer
================================
Computes the TruthScore™ (0-100) from multiple credibility signals.
"""

import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Credibility signal weights (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "linguistic_credibility":  0.20,
    "sentiment_balance":       0.15,
    "source_citation_index":   0.25,
    "headline_body_consistency": 0.15,
    "clickbait_probability":   0.25,  # inverse: low clickbait = high cred
}


class CredibilityScorer:
    """
    Computes a composite TruthScore™ from multiple NLP credibility signals.

    The score is on a 0–100 scale:
        0–30   → Likely Fake / Very Low Credibility
        31–55  → Suspicious / Uncertain
        56–100 → Likely Real / High Credibility

    All input features are expected as floats in [0.0, 1.0].
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_truth_score(
        self,
        linguistic_credibility: float,
        sentiment_balance: float,
        source_citation_index: float,
        headline_body_consistency: float,
        clickbait_probability: float,
        model_confidence: float = 0.5,
        model_label: str = "UNCERTAIN",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute TruthScore™ and individual credibility signal scores.

        Parameters
        ----------
        linguistic_credibility : float
            Proxy for writing formality (0=informal, 1=formal).
        sentiment_balance : float
            Neutrality score from VADER (0=extreme, 1=neutral).
        source_citation_index : float
            Proportion of credible source references found (0–1).
        headline_body_consistency : float
            Cosine similarity between headline and article body (0–1).
        clickbait_probability : float
            Probability of clickbait patterns present (0=none, 1=high).
        model_confidence : float
            ML classifier confidence in the REAL class (0–1).
        model_label : str
            Predicted label: 'REAL', 'FAKE', or 'UNCERTAIN'.

        Returns
        -------
        (truth_score, breakdown_dict)
            truth_score : float in [0, 100]
            breakdown_dict : per-signal scores scaled to [0, 100]
        """
        # Invert clickbait (lower clickbait = higher credibility)
        clickbait_credibility = 1.0 - clickbait_probability

        # Weighted sum of credibility signals
        weighted_sum = (
            linguistic_credibility  * SIGNAL_WEIGHTS["linguistic_credibility"] +
            sentiment_balance       * SIGNAL_WEIGHTS["sentiment_balance"] +
            source_citation_index   * SIGNAL_WEIGHTS["source_citation_index"] +
            headline_body_consistency * SIGNAL_WEIGHTS["headline_body_consistency"] +
            clickbait_credibility   * SIGNAL_WEIGHTS["clickbait_probability"]
        )

        # Model confidence adjustment
        if model_label.upper() == "REAL":
            model_boost = model_confidence * 0.30
        elif model_label.upper() == "FAKE":
            model_boost = (1.0 - model_confidence) * 0.30
        else:
            model_boost = 0.15  # neutral

        # Final score = 70% signal-based + 30% model-based
        raw_score = weighted_sum * 0.70 + model_boost
        truth_score = round(max(0.0, min(raw_score * 100, 100.0)), 1)

        breakdown = {
            "Linguistic Credibility":    round(linguistic_credibility * 100, 1),
            "Sentiment Balance":         round(sentiment_balance * 100, 1),
            "Source Citation Index":     round(source_citation_index * 100, 1),
            "Headline–Body Consistency": round(headline_body_consistency * 100, 1),
            "Clickbait Probability":     round(clickbait_probability * 100, 1),  # raw, not inverted
        }

        return truth_score, breakdown

    def get_verdict(self, truth_score: float) -> Tuple[str, str]:
        """
        Return a textual verdict and color code based on TruthScore.

        Parameters
        ----------
        truth_score : float (0–100)

        Returns
        -------
        (verdict_text, hex_color)
        """
        if truth_score <= 30:
            return "🔴 LIKELY FAKE", "#FF4444"
        elif truth_score <= 55:
            return "🟡 SUSPICIOUS", "#FFB800"
        else:
            return "🟢 LIKELY REAL", "#00C851"

    def compute_linguistic_credibility(
        self,
        passive_voice_ratio: float,
        avg_word_length: float,
        caps_ratio: float,
        exclamation_ratio: float,
    ) -> float:
        """
        Estimate linguistic credibility from writing-style features.

        Parameters
        ----------
        passive_voice_ratio : float  (0–1)
        avg_word_length     : float  (typical range 4–7)
        caps_ratio          : float  (0–1)
        exclamation_ratio   : float  (0–∞, clipped at 1)

        Returns
        -------
        float in [0, 1]
        """
        # Longer avg word length → more formal (credible)
        formality = min(avg_word_length / 7.0, 1.0)

        # Passive voice in moderation is fine; extreme is suspicious
        passive_score = 1.0 - min(passive_voice_ratio * 2.0, 1.0)

        # High caps → sensational
        caps_score = 1.0 - min(caps_ratio * 5.0, 1.0)

        # Exclamation marks → sensational
        excl_score = 1.0 - min(exclamation_ratio, 1.0)

        return round((formality * 0.3 + passive_score * 0.2 + caps_score * 0.25 + excl_score * 0.25), 3)

    def get_redflag_explanation(
        self, breakdown: Dict[str, float], truth_score: float
    ) -> list:
        """
        Generate 3 plain-English red-flag bullet points for students.

        Parameters
        ----------
        breakdown : dict from compute_truth_score
        truth_score : float (0–100)

        Returns
        -------
        list of str
        """
        bullets = []

        if breakdown.get("Clickbait Probability", 0) > 50:
            bullets.append(
                "🚩 **Clickbait Language Detected** — The headline uses sensationalist or "
                "emotionally manipulative phrasing to attract clicks without substance."
            )
        if breakdown.get("Sentiment Balance", 100) < 40:
            bullets.append(
                "🚩 **Extreme Emotional Tone** — The article uses strongly biased language "
                "(very positive or very negative), which is a common tactic in misinformation."
            )
        if breakdown.get("Source Citation Index", 100) < 30:
            bullets.append(
                "🚩 **Lacks Source Citations** — Credible journalism references verifiable sources. "
                "This article provides few or no external references."
            )
        if breakdown.get("Headline–Body Consistency", 100) < 40:
            bullets.append(
                "🚩 **Headline Doesn't Match Content** — The headline makes claims that the article "
                "body does not support, a classic misleading-headline tactic."
            )
        if breakdown.get("Linguistic Credibility", 100) < 40:
            bullets.append(
                "🚩 **Informal / Sensational Writing Style** — The writing uses excessive capitals, "
                "exclamation marks, or overly informal language uncommon in credible reporting."
            )
        if truth_score < 30:
            bullets.append(
                "🚩 **ML Models Flag as FAKE** — Multiple AI classifiers independently identified "
                "patterns consistent with known misinformation."
            )

        return bullets[:3] if bullets else [
            "ℹ️ No major red flags detected. Always verify with multiple trusted sources.",
            "ℹ️ Cross-check claims against fact-checking sites like Snopes or FactCheck.org.",
            "ℹ️ Consider the publication date, author credentials, and domain reputation."
        ]
