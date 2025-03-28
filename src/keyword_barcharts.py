# src/keyword_barcharts.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import Counter
from wordcloud import STOPWORDS


def load_texts_by_label(text_dir="output/texts", label_file="output/labels.csv", label_column="risk_level"):
    labels_df = pd.read_csv(label_file)
    label_texts = {}
    for label in labels_df[label_column].unique():
        label_texts[label] = []

    for _, row in labels_df.iterrows():
        file_path = os.path.join(text_dir, row["filename"])
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                label_texts[row[label_column]].append(f.read())

    return label_texts


def plot_top_keywords_by_label(label_texts, top_n=10):
    default_stopwords = set(word.lower() for word in STOPWORDS)
    custom_stopwords = {
        "et", "al", "data", "using", "study", "based", "used", "covid", "results", "conclusion",
        "in", "of", "and", "to", "the", "is", "for", "with", "on", "by", "from", "at", "as", "that"
    }
    stopwords = default_stopwords.union(custom_stopwords)

    for label, texts in label_texts.items():
        if texts:
            full_text = " ".join(texts).lower()
            full_text = full_text.translate(str.maketrans('', '', string.punctuation))
            words = full_text.split()
            filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

            word_counts = Counter(filtered_words).most_common(top_n)
            words, counts = zip(*word_counts)

            plt.figure(figsize=(10, 5))
            plt.bar(words, counts, color="skyblue")
            plt.title(f"Top {top_n} Keywords: {label}")
            plt.xticks(rotation=45)
            plt.xlabel("Keyword")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()