# utils/insights.py
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# import numpy as np


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
        # Force label to be a built-in Python int
        clusters.setdefault(int(label), []).append(keyword)
    clusters = {int(k): v for k, v in clusters.items()}
    return clusters


def plot_cuslters(clusters):
    # Simple bar chart
    fig, ax = plt.subplots()
    cluster_ids = list(clusters.keys())
    counts = [len(v) for v in clusters.values()]
    ax.bar(cluster_ids, counts)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Keywords")
    ax.set_title("Keyword Clusters")
    return fig


def plot_keywords(keywords):
    # Simple bar chart
    fig, ax = plt.subplots()
    ax.barh(keywords, range(len(keywords), 0, -1))
    ax.set_xlabel("Importance (ranked)")
    ax.set_ylabel("Keywords")
    return fig


# # Flatten to DataFrame
#     df = pd.DataFrame(
#         [(cluster_id, kw) for cluster_id, kws in clusters.items() for kw in kws],
#         columns=["Cluster", "Keyword"]
#     )

#     # Group by cluster and join keywords for labels
#     cluster_keywords = df.groupby("Cluster")["Keyword"].apply(lambda kws: ", ".join(kws)).reset_index()
#     cluster_counts = df.groupby("Cluster").size().reset_index(name="Count")
#     cluster_counts["Keywords"] = cluster_keywords["Keyword"]

#     # Bar chart: count of keywords per cluster, show keywords as text
#     fig = px.bar(
#         cluster_counts,
#         x="Cluster",
#         y="Count",
#         title="Keyword Count per Cluster",
#         text="Keywords"  # Show keywords as bar labels
#     )
#     fig.update_traces(textposition='outside', cliponaxis=False)
#     st.plotly_chart(fig)
