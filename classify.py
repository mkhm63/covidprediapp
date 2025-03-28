# src/classify.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def update_labels_csv(text_dir="output/texts", label_file="output/labels.csv"):
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
        if "china" in text:
            region = "china"
        elif "india" in text:
            region = "india"
        elif any(term in text for term in ["usa", "america", "united states"]):
            region = "usa"
        else:
            region = "global"

        data.append({
            "filename": fname,
            "risk_level": risk,
            "study_type": study,
            "region": region
        })

    df = pd.DataFrame(data)
    df.to_csv(label_file, index=False)
    print(f"✅ labels.csv auto-filled with detected labels for {len(all_files)} files.")


def load_texts_with_labels(text_dir="output/texts", label_file="output/labels.csv", label_column="risk_level"):
    labels_df = pd.read_csv(label_file)
    texts = []
    y = []
    for _, row in labels_df.iterrows():
        file_path = os.path.join(text_dir, row['filename'])
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                y.append(row[label_column])
    return texts, y


def classify_texts(texts, y):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # stratify removed for small datasets
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\n[ Classification Report ]\n")
    print(report)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return clf, vectorizer


def predict_unlabeled(clf, vectorizer, input_dir="output/unlabeled"):
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(input_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                X = vectorizer.transform([text])
                pred = clf.predict(X)[0]
                results.append((fname, pred))
    print("\n[ Predictions on Unlabeled Files ]\n")
    for fname, label in results:
        print(f"{fname}: {label}")
    return results