# utils/insights.py - English keyword extraction and visualization
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import spacy
import logging
from typing import List, Set, Optional
import warnings

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TOP_N = 10
DEFAULT_NGRAM_RANGE = (1, 3)
MIN_PHRASE_LENGTH = 2
DEFAULT_MAX_FEATURES = 1000

# Global spaCy model (lazy loading)
_nlp: Optional[spacy.Language] = None

# Expand the default stopwords to include domain-generic terms
CUSTOM_STOPWORDS = list(
    ENGLISH_STOP_WORDS.union(
        {
            "business",
            "use",
            "using",
            "user",
            "real",
            "area",
            "data",
            "information",
            "system",
            "service",
            "application",
            "software",
            "technology",
            "process",
            "method",
            "approach",
            "solution",
            "implementation",
            "development",
        }
    )
)


def _initialize_spacy() -> None:
    """Initialize spaCy model for NLP processing."""
    global _nlp

    if _nlp is None:
        try:
            logger.info("Loading spaCy English model (en_core_web_sm)")
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError as e:
            logger.error(f"Failed to load spaCy model: {e}")
            logger.error(
                "Please install the English model: python -m spacy download en_core_web_sm"
            )
            raise Exception(f"spaCy model not found: {e}")
        except Exception as e:
            logger.error(f"Error initializing spaCy: {e}")
            raise Exception(f"spaCy initialization failed: {e}")


def _reset_spacy_model() -> None:
    """Reset spaCy model (for testing purposes)."""
    global _nlp
    _nlp = None


def _extract_noun_chunks(text: str) -> Set[str]:
    """Extract candidate noun chunks from text."""
    try:
        _initialize_spacy()
        doc = _nlp(text)
        candidates = set()

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            # Filter out stopwords and short phrases
            if (
                phrase not in CUSTOM_STOPWORDS
                and len(phrase) > MIN_PHRASE_LENGTH
                and phrase.replace(" ", "").isalpha()
            ):  # Only alphabetic phrases
                candidates.add(phrase)

        return candidates
    except Exception as e:
        logger.error(f"Error extracting noun chunks: {e}")
        return set()


def _validate_input(text: str, top_n: int) -> None:
    """Validate input parameters."""
    if not text or not text.strip():
        raise ValueError("Input text is empty or contains only whitespace")

    if len(text.strip()) < 50:
        raise ValueError(
            "Input text is too short for meaningful keyword extraction (minimum 50 characters)"
        )

    if top_n < 1 or top_n > 100:
        raise ValueError("top_n must be between 1 and 100")


def extract_keywords_phrases(text: str, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """
    Extract keywords and phrases using TF-IDF with noun chunk filtering.

    Args:
        text: Input text for keyword extraction
        top_n: Number of top keywords/phrases to return

    Returns:
        List of keywords and phrases sorted by importance

    Raises:
        ValueError: For invalid input parameters
        Exception: For processing errors
    """
    _validate_input(text, top_n)

    try:
        # Step 1: Extract candidate phrases (noun chunks)
        candidates = _extract_noun_chunks(text)

        if not candidates:
            logger.warning(
                "No noun chunks found, falling back to basic keyword extraction"
            )
            return extract_keywords(text, top_n)

        # Step 2: Run TF-IDF on the candidate phrases only
        vectorizer = TfidfVectorizer(
            stop_words=CUSTOM_STOPWORDS,
            ngram_range=DEFAULT_NGRAM_RANGE,
            max_features=DEFAULT_MAX_FEATURES,
        )

        tfidf = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]

        # Step 3: Keep only scores where the term appears in candidates
        filtered_scores = []
        for i, term in enumerate(feature_names):
            if term in candidates and scores[i] > 0:
                filtered_scores.append((term, scores[i]))

        # Step 4: Sort by score
        sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)

        result = [term for term, score in sorted_scores[:top_n]]

        if not result:
            logger.warning("No phrases found, falling back to basic keyword extraction")
            return extract_keywords(text, top_n)

        return result

    except Exception as e:
        logger.error(f"Error in extract_keywords_phrases: {e}")
        raise Exception(f"Phrase extraction failed: {e}")


def extract_keywords(text: str, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """
    Basic keyword extraction using TF-IDF.

    Args:
        text: Input text for keyword extraction
        top_n: Number of top keywords to return

    Returns:
        List of keywords sorted by importance

    Raises:
        ValueError: For invalid input parameters
        Exception: For processing errors
    """
    _validate_input(text, top_n)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=DEFAULT_MAX_FEATURES,
            ngram_range=(1, 2),  # Single words and bigrams
        )

        tfidf = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]

        # Create score pairs and sort
        score_pairs = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        sorted_scores = sorted(score_pairs, key=lambda x: x[1], reverse=True)

        return [word for word, score in sorted_scores[:top_n]]

    except Exception as e:
        logger.error(f"Error in extract_keywords: {e}")
        raise Exception(f"Keyword extraction failed: {e}")


def plot_keywords(keywords: List[str]) -> plt.Figure:
    """
    Create keyword importance visualization.

    Args:
        keywords: List of keywords to visualize

    Returns:
        Matplotlib figure with horizontal bar chart

    Raises:
        ValueError: For empty keywords list
        Exception: For plotting errors
    """
    if not keywords:
        raise ValueError("Keywords list is empty")

    try:
        # Suppress matplotlib warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fig, ax = plt.subplots(figsize=(10, max(6, len(keywords) * 0.4)))

            # Create horizontal bar chart
            y_pos = range(len(keywords))
            ax.barh(y_pos, range(len(keywords), 0, -1), alpha=0.7)

            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(keywords)
            ax.set_xlabel("Importance (ranked)")
            ax.set_ylabel("Keywords")
            ax.set_title("Keyword Importance Ranking")

            # Invert y-axis to show highest importance at top
            ax.invert_yaxis()

            # Improve layout
            plt.tight_layout()

            return fig

    except Exception as e:
        logger.error(f"Error creating keyword plot: {e}")
        raise Exception(f"Plotting failed: {e}")
