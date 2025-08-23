# utils/insights.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import spacy

# Pick a system font that supports Chinese
# On macOS, PingFang SC is a good default
zh_font = fm.FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc")

# Load small English model for speed; swap for 'en_core_web_lg' for better accuracy
nlp = spacy.load("en_core_web_sm")

# Expand the default stopwords to include domain-generic terms
CUSTOM_STOPWORDS = list(
    ENGLISH_STOP_WORDS.union(
        {"business", "use", "using", "user", "real", "area", "data", "information"}
    )
)

def extract_keywords_phrases(text, top_n=10):
    # Step 1: Extract candidate phrases (noun chunks)
    doc = nlp(text)
    candidates = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if phrase not in CUSTOM_STOPWORDS and len(phrase) > 2:
            candidates.add(phrase)

    # Step 2: Run TF-IDF on the candidate phrases only
    vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOPWORDS, ngram_range=(1, 3))
    tfidf = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])

    # Step 3: Keep only scores where the term appears in candidates
    filtered_scores = [(term, score) for term, score in scores if term in candidates]

    # Step 4: Sort by score
    sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)

    return [term for term, score in sorted_scores[:top_n]]


def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:top_n]]


def plot_keywords(keywords):
    # Simple bar chart
    fig, ax = plt.subplots()
    ax.barh(keywords, range(len(keywords), 0, -1))
    ax.set_xlabel("Importance (ranked)")
    ax.set_ylabel("Keywords")
    return fig

def plot_chinese_keywords(keywords):
    fig, ax = plt.subplots()
    ax.barh(keywords, range(len(keywords), 0, -1))
    ax.set_xlabel("Importance (ranked)")
    ax.set_ylabel("关键词", fontproperties=zh_font)
    ax.set_title("关键词权重图", fontproperties=zh_font)

    # Apply font to tick labels
    ax.set_yticklabels(keywords, fontproperties=zh_font)
    return fig