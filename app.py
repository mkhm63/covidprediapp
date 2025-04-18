# app.py (flattened)
import os
import streamlit as st
import matplotlib.pyplot as plt
import spacy
import pycountry

from extract_text import extract_pdfs_to_txt
from classify import update_labels_csv, load_texts_with_labels, classify_texts
from visualize import load_texts_by_label, plot_wordclouds_by_label
from keyword_barcharts import plot_top_keywords_by_label

st.set_page_config(page_title="COVID Research Classifier", layout="wide")
st.title("COVID Scientific Article Classifier")

st.sidebar.header("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])

pdf_dir = "data/pdfs"
os.makedirs(pdf_dir, exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(pdf_dir, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s)")

if st.sidebar.button("Extract Text from PDFs"):
    extract_pdfs_to_txt()
    st.success("Text extracted from all PDFs.")

# Monkey patch classify.update_labels_csv to use spacy and pycountry
import classify

def improved_update_labels_csv(text_dir="output/texts", label_file="output/labels.csv"):
    import pandas as pd
    from spacy import load
    from collections import Counter

    nlp = load("en_core_web_sm")

    country_list = {country.name.lower() for country in pycountry.countries}
    known_regions = {"europe", "asia", "africa", "south america", "north america", "middle east", "australia"}

    all_files = sorted(f for f in os.listdir(text_dir) if f.endswith(".txt"))
    data = []
    for fname in all_files:
        path = os.path.join(text_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # Detect risk_level
        if any(term in text for term in ["mortality", "death", "icu"]):
            risk = "high"
        elif any(term in text for term in ["hospital", "infection", "spread"]):
            risk = "medium"
        else:
            risk = "low"

        # Detect study_type
        if any(term in text for term in ["model", "forecast", "simulation"]):
            study = "modeling"
        elif any(term in text for term in ["x-ray", "scan", "detection", "diagnostic"]):
            study = "diagnostic"
        elif "vaccine" in text:
            study = "vaccine"
        else:
            study = "other"

        # Detect region
        doc = nlp(text)
        found = [ent.text.lower() for ent in doc.ents if ent.label_ == "GPE"]
        locations = Counter(found)
        region = "global"
        for loc, _ in locations.most_common():
            if loc in country_list:
                region = loc
                break
            elif loc in known_regions:
                region = loc
                break

        data.append({
            "filename": fname,
            "risk_level": risk,
            "study_type": study,
            "region": region
        })

    df = pd.DataFrame(data)
    df.to_csv(label_file, index=False)
    print(f"✅ labels.csv auto-filled with detected labels for {len(all_files)} files.")

classify.update_labels_csv = improved_update_labels_csv

if st.sidebar.button("Update Labels Automatically"):
    update_labels_csv()
    st.success("Labels.csv updated with detected risk_level, study_type, and region.")

st.sidebar.markdown("---")

label_option = st.sidebar.selectbox("Choose Label Type for Analysis", ["risk_level", "study_type", "region"])

# Initialize session state
if 'classifier_ready' not in st.session_state:
    st.session_state.classifier_ready = False
if 'label_texts' not in st.session_state:
    st.session_state.label_texts = {}

if st.sidebar.button("🤖 Train Classifier & Visualize"):
    with st.spinner("Training model and generating visualizations..."):
        try:
            texts, y = load_texts_with_labels(label_column=label_option)
            st.write(f"Loaded {len(texts)} labeled documents for '{label_option}'")
            if not texts:
                st.warning("No texts found. Please upload and extract PDFs first.")
            clf, vectorizer = classify_texts(texts, y)

            st.session_state.label_texts = load_texts_by_label(label_column=label_option)
            st.session_state.classifier_ready = True
            st.success("Model trained. Now select a label to visualize.")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")

# Show selectbox and visualizations after classifier is ready
if st.session_state.classifier_ready and st.session_state.label_texts:
    detected_labels = list(st.session_state.label_texts.keys())
    selected_label_value = st.selectbox(f"Select a {label_option} to visualize:", detected_labels)
    filtered_texts = {selected_label_value: st.session_state.label_texts[selected_label_value]}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top Keywords for {selected_label_value}")
        plt.figure(figsize=(12, 6))
        plot_top_keywords_by_label(filtered_texts)
        st.pyplot(plt.gcf())

    with col2:
        st.subheader(f"Word Cloud for {selected_label_value}")
        plt.figure(figsize=(12, 6))
        plot_wordclouds_by_label(filtered_texts)
        st.pyplot(plt.gcf())
