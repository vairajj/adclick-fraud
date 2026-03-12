"""
Ad Click Fraud Detection - Full Implementation
Based on: Alzahrani, Aljabri & Mohammad (IEEE Access, 2025)

Dataset: advertising.csv (Kaggle - gabrielsantello)
Target:  'Clicked on Ad' (0 = did not click, 1 = clicked)

Pipeline:
  1. Data Loading & Preprocessing
  2. Feature Engineering
  3. Feature Selection (RFE)
  4. ML Models: LR, DT, RF, KNN, ANN, GB, NB, SVM
  5. Evaluation (10-fold CV): Accuracy, Precision, Recall, F1
  6. Plots & Results
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath="advertising.csv"):
    df = pd.read_csv(filepath)
    print(f"[STEP 1] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"         Target distribution:\n{df['Clicked on Ad'].value_counts().to_string()}")
    return df

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    print("\n[STEP 2] Preprocessing...")
    df = df.drop(columns=["Ad Topic Line", "City", "Country"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["hour"]      = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"]     = df["Timestamp"].dt.month
    df = df.drop(columns=["Timestamp"])
    df = df.fillna(df.median(numeric_only=True))
    print(f"         Shape after preprocessing: {df.shape}")
    return df

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    print("\n[STEP 3] Feature engineering...")
    df["time_per_usage"]   = df["Daily Time Spent on Site"] / (df["Daily Internet Usage"] + 1e-6)
    df["age_bin"]          = pd.cut(df["Age"], bins=[0,25,35,45,60,100], labels=[0,1,2,3,4]).astype(int)
    df["income_per_usage"] = df["Area Income"] / (df["Daily Internet Usage"] + 1e-6)
    df["is_peak_hour"]     = df["hour"].between(9, 18).astype(int)
    df["is_weekend"]       = (df["dayofweek"] >= 5).astype(int)
    print(f"         Shape after feature engineering: {df.shape}")
    return df

# ─────────────────────────────────────────────
# 4. FEATURE SELECTION (RFE)
# ─────────────────────────────────────────────

def select_features(X, y, n_features=10):
    print("\n[STEP 4] Feature selection (RFE)...")
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]), step=1)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    print(f"         Selected {len(selected)} features: {selected}")
    return selected

# ─────────────────────────────────────────────
# 5. ML MODELS + 10-FOLD CV
# ─────────────────────────────────────────────

def get_models():
    return {
        "LR":  LogisticRegression(max_iter=1000, random_state=42),
        "DT":  DecisionTreeClassifier(max_depth=15, random_state=42),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
        "GB":  GradientBoostingClassifier(n_estimators=100, random_state=42),
        "NB":  GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }


def evaluate_models(X, y, n_splits=10):
    print(f"\n[STEP 5] Training & evaluating models ({n_splits}-fold CV)...")
    models  = get_models()
    cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    results = []

    for name, model in models.items():
        print(f"  [{name}] Training...", end=" ", flush=True)
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {
            "Model":     name,
            "Accuracy":  scores["test_accuracy"].mean(),
            "Precision": scores["test_precision_weighted"].mean(),
            "Recall":    scores["test_recall_weighted"].mean(),
            "F1-Score":  scores["test_f1_weighted"].mean(),
        }
        results.append(row)
        print(f"Acc={row['Accuracy']*100:.2f}%  "
              f"P={row['Precision']*100:.2f}%  "
              f"R={row['Recall']*100:.2f}%  "
              f"F1={row['F1-Score']*100:.2f}%")

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────

def plot_results(results_df):
    print("\n[STEP 6] Generating plots...")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = results_df[metric] * 100
        bars = ax.bar(results_df["Model"], vals, color="steelblue",
                      edgecolor="white", linewidth=0.5)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_ylim(50, 105)
        ax.set_xticklabels(results_df["Model"], rotation=30, ha="right", fontsize=10)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle("Ad Click Detection — Model Comparison (10-fold CV)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("         Saved: model_comparison.png")
    plt.show()


def plot_confusion_matrix(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Clicked", "Clicked"],
                yticklabels=["Not Clicked", "Clicked"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("         Saved: confusion_matrix.png")
    plt.show()


def plot_feature_importance(X, y, feature_names):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importances — Random Forest", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    print("         Saved: feature_importance.png")
    plt.show()

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AD CLICK DETECTION — PIPELINE")
    print("  Dataset: advertising.csv")
    print("=" * 60)

    df       = load_data("advertising.csv")
    df       = preprocess(df)
    df       = engineer_features(df)

    target   = "Clicked on Ad"
    X_df     = df.drop(columns=[target])
    y        = df[target]

    selected  = select_features(X_df, y, n_features=10)
    X_sel     = X_df[selected]

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_sel)

    results   = evaluate_models(X_scaled, y)

    print("\n" + "=" * 60)
    display = results.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display[col] = (display[col] * 100).round(2).astype(str) + "%"
    print(display.to_string(index=False))
    print("=" * 60)

    results.to_csv("results_summary.csv", index=False)
    print("\n[INFO] Results saved to results_summary.csv")

    plot_results(results)
    plot_confusion_matrix(X_scaled, y.values)
    plot_feature_importance(X_sel, y, selected)

    print("\n[DONE] All steps completed.")


if __name__ == "__main__":
    main()