import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import re
import os

# Download font if not already cached
font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
font_path = "/tmp/NotoSansCJKsc-Regular.otf"

if not os.path.exists(font_path):
    r = requests.get(font_url)
    with open(font_path, "wb") as f:
        f.write(r.content)

zh_font = fm.FontProperties(fname=font_path)

# Optional: load a standard Chinese stopword list (you can expand it)
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
    ]
)

# Add your own domain-specific "banned" words here
CUSTOM_BLOCKLIST = set(["公司", "数据", "业务", "使用"])


def jieba_tokenizer(text):
    """Tokenize and filter out stopwords / single chars."""
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok:
            continue
        # Skip stopwords and blocklist
        if tok in CHINESE_STOPWORDS or tok in CUSTOM_BLOCKLIST:
            continue
        # Skip single characters (except numbers/letters)
        if len(tok) == 1 and not re.match(r"[A-Za-z0-9]", tok):
            continue
        tokens.append(tok)
    return tokens


def extract_chinese_keywords(text, top_n=10):
    # TF-IDF with uni- and bi-grams for richer phrases
    vectorizer = TfidfVectorizer(
        tokenizer=jieba_tokenizer,
        ngram_range=(1, 2),  # single words + 2-word phrases
        max_df=1.0, 
        min_df=1,
    )
    tfidf = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:top_n]]


def plot_chinese_keywords(keywords):
    fig, ax = plt.subplots()
    ax.barh(keywords, range(len(keywords), 0, -1))
    ax.set_xlabel("Importance (ranked)")
    ax.set_ylabel("关键词", fontproperties=zh_font)
    ax.set_title("关键词权重图", fontproperties=zh_font)

    # Apply font to tick labels
    ax.set_yticklabels(keywords, fontproperties=zh_font)
    return fig
