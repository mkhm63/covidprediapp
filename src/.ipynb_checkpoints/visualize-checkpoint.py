# src/visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string


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


def plot_wordclouds_by_label(label_texts):
    from collections import Counter

    default_stopwords = set(word.lower() for word in STOPWORDS)
    custom_stopwords = {"et", "al", "data", "using", "study", "based", "used", "covid", "results", "conclusion",
                        "in", "of", "and", "to", "the", "is", "for", "with", "on", "by", "from", "at", "as", "that"}
    stopwords = default_stopwords.union(custom_stopwords)

    for label, texts in label_texts.items():
        if texts:
            # Tokenize and clean
            full_text = " ".join(texts).lower()
            full_text = full_text.translate(str.maketrans('', '', string.punctuation))
            words = full_text.split()
            filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
            cleaned_text = " ".join(filtered_words)

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                stopwords=None,  # already removed
                collocations=False
            ).generate(cleaned_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud: {label}")
            plt.tight_layout()
            plt.show()