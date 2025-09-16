import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk

# --- Load and Clean Data -----------------------------------------------------
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Keep relevant columns
    core_cols = ["ProductId", "UserId", "ProfileName", "Score", "Time", "Summary", "Text",
                 "HelpfulnessNumerator", "HelpfulnessDenominator"]
    df = df[[c for c in core_cols if c in df.columns]]

    # Rename Helpfulness columns if needed
    help_cols = [c for c in df.columns if "Helpfulness" in c]
    if len(help_cols) == 2 and "HelpfulnessNumerator" not in df.columns:
        df = df.rename(columns={help_cols[0]: "HelpfulnessNumerator",
                                help_cols[1]: "HelpfulnessDenominator"})

    # Drop missing and duplicate reviews
    df = df.dropna(subset=["Text"])
    df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"], keep="first")

    # Combine Summary and Text
    df["review"] = df["Summary"].fillna("") + ". " + df["Text"].fillna("")
    df = df[df["review"].astype(str).str.strip().ne("")].copy()

    return df

# --- Labeling ---------------------------------------------------------------
def score_to_label(score):
    return "pos" if score >= 4 else "neg" if score <= 2 else "neu"

def apply_labels(df):
    df["label"] = df["Score"].astype(int).map(score_to_label)
    return df

# --- VADER Baseline ---------------------------------------------------------
def apply_vader(df):
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    def vader_label(text):
        compound = sia.polarity_scores(text)["compound"]
        return "pos" if compound > 0.05 else "neg" if compound < -0.05 else "neu"

    df["vader"] = df["review"].map(vader_label)
    print("\n=== VADER Baseline ===")
    print(classification_report(df["label"], df["vader"]))
    print(confusion_matrix(df["label"], df["vader"]))
    return df

# --- TF-IDF + Logistic Regression -------------------------------------------
def train_model(df):
    X = df["review"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)

    print("\n=== TF-IDF + Logistic Regression ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=["neg", "neu", "pos"]))

    return clf, tfidf, X, y, X_test, y_test, y_pred

# --- Feature Inspection -----------------------------------------------------
def show_top_features(clf, tfidf, top_n=15):
    for label in clf.classes_:
        idx = list(clf.classes_).index(label)
        coefs = clf.coef_[idx]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        features = tfidf.get_feature_names_out()[top_idx]
        weights = coefs[top_idx]
        print(f"\nTop features for '{label}':")
        for feat, weight in zip(features, weights):
            print(f"  {feat:30s} {weight: .3f}")

# --- Enrich Data with Predictions -------------------------------------------
def enrich_predictions(df, clf, tfidf):
    X_all = tfidf.transform(df["review"].astype(str))
    df["clf_pred"] = clf.predict(X_all)
    proba = clf.predict_proba(X_all)
    for i, label in enumerate(clf.classes_):
        df[f"prob_{label}"] = proba[:, i]
    return df

# --- Visualizations ---------------------------------------------------------
def generate_visuals(df, y_test, y_pred, file_path):
    plot_dir = os.path.join(os.path.dirname(file_path), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Class distribution
    fig1 = plt.figure()
    true_counts = df["label"].value_counts().reindex(["neg", "neu", "pos"]).fillna(0)
    pred_counts = df["clf_pred"].value_counts().reindex(["neg", "neu", "pos"]).fillna(0)
    x = np.arange(len(true_counts.index))
    width = 0.35
    plt.bar(x - width/2, true_counts.values, width, label="True")
    plt.bar(x + width/2, pred_counts.values, width, label="Predicted")
    plt.xticks(x, ["neg", "neu", "pos"])
    plt.title("Sentiment distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "class_distribution.png"), dpi=150)
    plt.show()

    # 2. Confusion matrix
    fig2 = plt.figure()
    cm = confusion_matrix(y_test, y_pred, labels=["neg", "neu", "pos"])
    im = plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(x, ["neg", "neu", "pos"])
    plt.yticks(x, ["neg", "neu", "pos"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"), dpi=150)
    plt.show()

    # 3. Monthly sentiment trend
    if np.issubdtype(df["Time"].dtype, np.number):
        df["dt"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    else:
        df["dt"] = pd.to_datetime(df["Time"], errors="coerce")

    monthly = (
        df.dropna(subset=["dt"])
          .set_index("dt")
          .assign(is_pos=(df["clf_pred"] == "pos").astype(int))
          .resample("M")
          .agg(reviews=("clf_pred", "size"), pos=("is_pos", "mean"))
          .query("reviews >= 50")
    )

    fig3 = plt.figure()
    plt.plot(monthly.index, monthly["pos"])
    plt.title("Monthly Positive Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "monthly_positive_rate.png"), dpi=150)
    plt.show()

# --- Main Execution ---------------------------------------------------------
def main():
    file_path = r"my_work/Reviews.csv"
    df = load_and_clean_data(file_path)
    df = apply_labels(df)
    df = apply_vader(df)
    clf, tfidf, X, y, X_test, y_test, y_pred = train_model(df)
    show_top_features(clf, tfidf)
    df = enrich_predictions(df, clf, tfidf)
    generate_visuals(df, y_test, y_pred, file_path)

    # Save enriched data
    out_path = os.path.join(os.path.dirname(file_path), "reviews_with_sentiment.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved enriched dataset to: {out_path}")

if __name__ == "__main__":
    main()