import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv("data/tiktok_scroll.csv")

# Combine text fields into one input
df["text"] = (
    df["caption_text"].fillna("") + " " +
    df["onscreen_text"].fillna("") + " " +
    df["spoken_summary"].fillna("") + " " +
    df["claim"].fillna("")
)

X = df["text"]
y = df["label"]

# Train / test split (small, but fine for now)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Build a simple text classification pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Train
pipeline.fit(X_train, y_train)



y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))

print("\nDetailed predictions:")
for text, true_label, pred_label in zip(X_test, y_test, y_pred):
    print("-" * 60)
    print("TRUE:", true_label, "| PRED:", pred_label)
    print("TEXT SNIPPET:", text[:180].replace("\n", " "), "...")

# Save confusion matrix plot
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=200)
print("\nSaved confusion matrix to results/confusion_matrix.png")

