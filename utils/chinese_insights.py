import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import re
import os
import logging
from typing import List, Optional
import warnings

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TOP_N = 10
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MAX_FEATURES = 1000
MIN_TOKEN_LENGTH = 1

# Font configuration
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
FONT_PATH = "/tmp/NotoSansCJKsc-Regular.otf"
_zh_font: Optional[fm.FontProperties] = None


def _initialize_chinese_font() -> None:
    """Initialize Chinese font for matplotlib."""
    global _zh_font

    if _zh_font is None:
        try:
            # Download font if not already cached
            if not os.path.exists(FONT_PATH):
                logger.info("Downloading Chinese font for visualization")
                response = requests.get(FONT_URL, timeout=10)
                response.raise_for_status()
                with open(FONT_PATH, "wb") as f:
                    f.write(response.content)
                logger.info("Chinese font downloaded successfully")

            _zh_font = fm.FontProperties(fname=FONT_PATH)
            logger.info("Chinese font initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Chinese font: {e}")
            logger.warning("Chinese characters may not display correctly in plots")
            # Use default font as fallback
            _zh_font = fm.FontProperties()


def _reset_chinese_font() -> None:
    """Reset Chinese font (for testing purposes)."""
    global _zh_font
    _zh_font = None


# Chinese stopwords list (expandable)
CHINESE_STOPWORDS = set(
    [
        "的",
        "了",
        "和",
        "是",
        "我",
        "也",
        "在",
        "有",
        "就",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "中",
        "大",
        "用",
        "对",
        "地",
        "与",
        "之",
        "及",
        "或",
        "而",
        "被",
        "从",
        "正在",
        "但",
        "等",
        "很",
        "到",
        "说",
        "要",
        "会",
        "可",
        "你",
        "自己",
        "我们",
        "没有",
        "他们",
        "它",
        "其",
        "这",
        "那",
        "这",
        "那",
        "个",
        "些",
        "种",
        "样",
        "时",
        "年",
        "月",
        "日",
        "天",
        "年",
        "月",
        "日",
        "天",
        "年",
        "月",
        "日",
        "天",
    ]
)

# Domain-specific blocklist
CUSTOM_BLOCKLIST = set(["公司", "业务", "使用", "系统", "服务", "应用", "软件"])


def _jieba_tokenizer(text: str) -> List[str]:
    """Tokenize Chinese text and filter out stopwords."""
    tokens = []
    try:
        for token in jieba.cut(text):
            token = token.strip()
            if not token:
                continue

            # Skip stopwords and blocklist
            if token in CHINESE_STOPWORDS or token in CUSTOM_BLOCKLIST:
                continue

            # Skip single characters (except numbers/letters)
            if len(token) == 1 and not re.match(r"[A-Za-z0-9]", token):
                continue

            # Skip very short tokens
            if len(token) < MIN_TOKEN_LENGTH:
                continue

            tokens.append(token)

        return tokens
    except Exception as e:
        logger.error(f"Error in jieba tokenization: {e}")
        return []


def _validate_input(text: str, top_n: int) -> None:
    """Validate input parameters for Chinese keyword extraction."""
    if not text or not text.strip():
        raise ValueError("Input text is empty or contains only whitespace")

    if len(text.strip()) < 50:
        raise ValueError(
            "Input text is too short for meaningful keyword extraction (minimum 50 characters)"
        )

    if top_n < 1 or top_n > 100:
        raise ValueError("top_n must be between 1 and 100")


def extract_chinese_keywords(text: str, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """
    Extract Chinese keywords using TF-IDF with jieba segmentation.

    Args:
        text: Chinese text for keyword extraction
        top_n: Number of top keywords to return

    Returns:
        List of Chinese keywords sorted by importance

    Raises:
        ValueError: For invalid input parameters
        Exception: For processing errors
    """
    _validate_input(text, top_n)

    try:
        # TF-IDF with uni- and bi-grams for richer phrases
        vectorizer = TfidfVectorizer(
            tokenizer=_jieba_tokenizer,
            ngram_range=DEFAULT_NGRAM_RANGE,  # single words + 2-word phrases
            max_features=DEFAULT_MAX_FEATURES,
            max_df=1.0,
            min_df=1,
        )

        tfidf = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]

        # Create score pairs and sort
        score_pairs = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        sorted_scores = sorted(score_pairs, key=lambda x: x[1], reverse=True)

        return [word for word, score in sorted_scores[:top_n]]

    except Exception as e:
        logger.error(f"Error in extract_chinese_keywords: {e}")
        raise Exception(f"Chinese keyword extraction failed: {e}")


def plot_chinese_keywords(keywords: List[str]) -> plt.Figure:
    """
    Create Chinese keyword importance visualization.

    Args:
        keywords: List of Chinese keywords to visualize

    Returns:
        Matplotlib figure with horizontal bar chart

    Raises:
        ValueError: For empty keywords list
        Exception: For plotting errors
    """
    if not keywords:
        raise ValueError("Keywords list is empty")

    try:
        # Initialize Chinese font (with fallback)
        try:
            _initialize_chinese_font()
        except Exception as e:
            logger.warning(f"Font initialization failed, using default font: {e}")

        # Suppress matplotlib warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fig, ax = plt.subplots(figsize=(10, max(6, len(keywords) * 0.4)))

            # Create horizontal bar chart
            y_pos = range(len(keywords))
            ax.barh(y_pos, range(len(keywords), 0, -1), alpha=0.7)

            # Customize the plot with Chinese labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(keywords, fontproperties=_zh_font)
            ax.set_xlabel("Importance (ranked)")
            ax.set_ylabel("关键词", fontproperties=_zh_font)
            ax.set_title("关键词权重图", fontproperties=_zh_font)

            # Invert y-axis to show highest importance at top
            ax.invert_yaxis()

            # Improve layout
            plt.tight_layout()

            return fig

    except Exception as e:
        logger.error(f"Error creating Chinese keyword plot: {e}")
        raise Exception(f"Chinese plotting failed: {e}")
