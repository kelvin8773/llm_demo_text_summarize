# utils/insights.py
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:top_n]]


def cluster_keywords(keywords, n_clusters=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(keywords)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    clusters = {}
    for keyword, label in zip(keywords, km.labels_):
        clusters.setdefault(label, []).append(keyword)
    return clusters


def plot_keywords(keywords):
    # Simple bar chart
    fig, ax = plt.subplots()
    ax.barh(keywords, range(len(keywords), 0, -1))
    ax.set_xlabel("Importance (ranked)")
    ax.set_ylabel("Keywords")
    return fig
