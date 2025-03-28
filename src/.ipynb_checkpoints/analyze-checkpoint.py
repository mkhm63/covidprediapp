# src/analyze.py
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def load_texts(text_dir="output/texts"):
    texts = []
    filenames = []
    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
                filenames.append(filename)
    return filenames, texts

def topic_modeling(texts, n_topics=3, n_words=10):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    topics = []
    for idx, topic in enumerate(lda.components_):
        terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_words-1:-1]]
        topics.append((f"Topic {idx+1}", terms))
    return topics

def print_topics(topics):
    print("\n[ Topics Discovered ]")
    for name, terms in topics:
        print(f"{name}: {', '.join(terms)}")