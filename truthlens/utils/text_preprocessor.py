"""
TruthLens — Text Preprocessor
================================
Cleans and normalizes raw article text for ML feature extraction.
"""

import re
import string
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)

# Download required NLTK data silently
for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


class TextPreprocessor:
    """
    Handles all text cleaning operations before ML inference.

    Parameters
    ----------
    remove_stopwords : bool
        Whether to strip stopwords from the text.
    lemmatize : bool
        Whether to apply lemmatization.
    """

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True) -> None:
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self._stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if lemmatize else None

    # ── Public API ─────────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Apply full cleaning pipeline to a raw text string.

        Steps:
        1. Lowercase
        2. Remove URLs, emails, mentions, hashtags
        3. Remove punctuation and digits
        4. Tokenize
        5. Remove stopwords (optional)
        6. Lemmatize (optional)
        7. Rejoin

        Parameters
        ----------
        text : str
            Raw article or headline text.

        Returns
        -------
        str
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = self._remove_urls(text)
        text = self._remove_special_patterns(text)
        text = self._remove_punctuation(text)
        tokens = self._tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]
        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 1]
        return " ".join(tokens)

    def get_sentences(self, text: str) -> list:
        """
        Split text into a list of sentences.

        Parameters
        ----------
        text : str
            Raw text to split.

        Returns
        -------
        list of str
        """
        try:
            return sent_tokenize(text)
        except Exception:
            return text.split(".")

    def count_words(self, text: str) -> int:
        """Return word count for raw text."""
        return len(text.split())

    def get_avg_word_length(self, text: str) -> float:
        """Compute average word length as a proxy for text complexity."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def passive_voice_ratio(self, text: str) -> float:
        """
        Estimate passive voice ratio using simple regex heuristics.

        Returns
        -------
        float
            Ratio of passive constructions to total sentences (0.0 to 1.0).
        """
        sentences = self.get_sentences(text)
        if not sentences:
            return 0.0
        passive_pattern = re.compile(
            r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE
        )
        passive_count = sum(1 for s in sentences if passive_pattern.search(s))
        return passive_count / len(sentences)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Strip URLs, email addresses from text."""
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        return text

    @staticmethod
    def _remove_special_patterns(text: str) -> str:
        """Remove mentions, hashtags, HTML entities, extra whitespace."""
        text = re.sub(r"@\w+|#\w+", " ", text)  # mentions / hashtags
        text = re.sub(r"&[a-z]+;", " ", text)     # HTML entities
        text = re.sub(r"\s+", " ", text)           # collapse whitespace
        return text.strip()

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Replace punctuation with spaces."""
        table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        return text.translate(table)

    @staticmethod
    def _tokenize(text: str) -> list:
        """Tokenize using NLTK word_tokenize with fallback."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()
