# detection/ml_model/train_model.py
"""
Train multiple classifiers for Fake News Detection using Kaggle True/False CSVs.
- Models: LogisticRegression, SVC, RandomForest, GradientBoosting, XGBoost
- Uses Pipeline with 'tfidf' and 'model' steps. Hyperparams referenced as model__*
- Performs GridSearchCV for each model, compares performance, visualizes results,
  and saves the best model (pipeline) and the TF-IDF vectorizer.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DATA_DIR = os.path.dirname(__file__)               # detection/ml_model
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
OUT_DIR = DATA_DIR                                 # save models & plots here
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3                                       # GridSearch CV folds (adjust for speed)
N_JOBS = 1                                         # run on single core to avoid memory error

# -----------------------------------------------------------------------------
# 1) Load + prepare dataset
# -----------------------------------------------------------------------------
def load_and_prepare(true_path, fake_path):
    # Read CSV files (Kaggle format: title, text, subject, date)
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df['label'] = 0   # REAL
    fake_df['label'] = 1   # FAKE

    df = pd.concat([true_df[['title', 'text', 'label']], fake_df[['title', 'text', 'label']]], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

    # Combine title + text (if title exists)
    df['text_full'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Basic cleaning function - kept minimal and consistent with inference pipeline
    def _clean(text):
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df['cleaned'] = df['text_full'].apply(_clean)
    df = df[df['cleaned'].str.len() > 20]  # drop extremely short items (optional)
    return df[['cleaned', 'label']]

df = load_and_prepare(TRUE_CSV, FAKE_CSV)
print(f"Loaded dataset: {df.shape[0]} rows (labels distribution):")
print(df['label'].value_counts())

# -----------------------------------------------------------------------------
# 2) Train/test split
# -----------------------------------------------------------------------------
X = df['cleaned'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------------------------------------------------------
# 3) Models dictionary (simple naming) with hyperparameter grids using model__*
# -----------------------------------------------------------------------------
models = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "params": {
            "model__C": [0.1, 1, 5]
        }
    },
    # "SVC": {
    #     "estimator": SVC(probability=True, random_state=RANDOM_STATE),
    #     "params": {
    #         "model__C": [0.1, 1, 5],
    #         "model__kernel": ["linear", "rbf"]
    #     }
    #  },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [100],
            "model__max_depth": [10]
        }
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__learning_rate": [0.1],
            "model__n_estimators": [100],
            "model__max_depth": [3]
        }
    },
    # "XGBoost": {
    #     "estimator": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
    #     "params": {
    #         "model__n_estimators": [100, 200],
    #         "model__max_depth": [3, 5],
    #         "model__learning_rate": [0.01, 0.1]
    #     }
    # }
}

# -----------------------------------------------------------------------------
# 4) Train each model with GridSearchCV, evaluate and store results
# -----------------------------------------------------------------------------
results = []
best_overall = {"name": None, "model": None, "acc": 0.0}

for name, cfg in models.items():
    print(f"\n=== Training: {name} ===")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=0.85, ngram_range=(1,2), max_features=20000)),
        ("model", cfg["estimator"])
    ])

    param_grid = cfg["params"]
    grid = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=N_JOBS, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{name}] Best CV params: {grid.best_params_}")
    print(f"[{name}] Test accuracy: {acc:.4f}")
    print(f"[{name}] Classification report:\n{classification_report(y_test, y_pred, digits=4)}")

    # confidence / probability (where available)
    if hasattr(best, "predict_proba"):
        y_proba = best.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        # Use decision function where possible for a proxy to ROC AUC
        try:
            scores = best.decision_function(X_test)
            auc = roc_auc_score(y_test, scores)
            y_proba = None
        except:
            auc = None
            y_proba = None

    # Confusion matrix plot save
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(OUT_DIR, f"{name}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # ROC curve plot (if probabilities available)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} - ROC Curve")
        plt.legend(loc="lower right")
        roc_path = os.path.join(OUT_DIR, f"{name}_roc.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        print(f"Saved ROC curve to: {roc_path}")

    results.append({
        "name": name,
        "accuracy": acc,
        "auc": auc,
        "best_params": grid.best_params_
    })

    # track best
    if acc > best_overall["acc"]:
        best_overall["name"] = name
        best_overall["model"] = best
        best_overall["acc"] = acc

# -----------------------------------------------------------------------------
# 5) Results summary and visualization (accuracy bar chart)
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
print("\n=== Summary ===")
print(results_df)

# save summary CSV
summary_csv = os.path.join(OUT_DIR, "model_comparison_summary.csv")
results_df.to_csv(summary_csv, index=False)
print(f"Saved model comparison summary to: {summary_csv}")

# Plot accuracies
plt.figure(figsize=(8,4))
sns.barplot(x="name", y="accuracy", data=results_df, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xlabel("")
plt.xticks(rotation=30)
acc_plot = os.path.join(OUT_DIR, "model_accuracy_comparison.png")
plt.tight_layout()
plt.savefig(acc_plot, bbox_inches="tight")
plt.close()
print(f"Saved accuracy comparison plot to: {acc_plot}")

# -----------------------------------------------------------------------------
# 6) Save best model
# -----------------------------------------------------------------------------
best_model = best_overall["model"]
best_name = best_overall["name"]
if best_model is None:
    raise RuntimeError("Training failed to select a best model.")

pipeline_path = os.path.join(OUT_DIR, "fake_news_pipeline.pkl")

# Save ONLY the full pipeline (tfidf + classifier)
joblib.dump(best_model, pipeline_path)

print(f"Saved best model pipeline ({best_name}) to: {pipeline_path}")
print("\nTraining complete.")
