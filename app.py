# app.py (flattened)
import os
import streamlit as st
import matplotlib.pyplot as plt

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
