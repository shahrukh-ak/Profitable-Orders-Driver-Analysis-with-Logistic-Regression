"""
Profitable Orders Driver Analysis with Logistic Regression
===========================================================
Uses statsmodels Logit to model the probability that a sales order is
profitable. Covers outlier removal, dummy variable creation, coefficient
interpretation, and standard classification metrics.

Dataset: profitable_orders.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


RANDOM_STATE = 1502


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the profitable orders dataset."""
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(df.head())
    return df


# ── Target Variable ───────────────────────────────────────────────────────────

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create profit_binary: 1 if profit > 0, else 0."""
    df["profit_binary"] = np.where(df["profit"] > 0, 1, 0)
    print(f"\nClass balance: {df['profit_binary'].value_counts(normalize=True).round(3).to_dict()}")
    return df


# ── EDA and Outlier Removal ───────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, cols: list):
    """Histograms for selected columns."""
    df[cols].hist(figsize=(10, 4), bins=30)
    plt.suptitle("Feature Distributions (before outlier removal)")
    plt.tight_layout()
    plt.savefig("distributions_before.png", dpi=150)
    plt.show()
    print("Saved: distributions_before.png")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme values in items_per_order and average_item_value."""
    df = df[df["items_per_order"] < 10]
    df = df[df["average_item_value"] < 250]
    print(f"Shape after outlier removal: {df.shape}")
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple:
    """
    One-hot encode categoricals, isolate X and y, add constant,
    convert to float, and produce an 80/20 train/test split.
    """
    df = pd.get_dummies(df, drop_first=True)

    y = df["profit_binary"]
    X = df.drop(columns=["order_number", "profit", "profit_binary"], errors="ignore")

    X = sm.add_constant(X).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


# ── Model ─────────────────────────────────────────────────────────────────────

def fit_logit(X_train, y_train):
    """Fit a statsmodels Logit model and print the summary."""
    model = sm.Logit(y_train, X_train).fit(disp=False)
    print(model.summary())
    return model


# ── Coefficient Interpretation ────────────────────────────────────────────────

def interpret_coefficient(coefficient: float, name: str = ""):
    """
    Print the percentage change in probability of a profitable order
    associated with a one-unit increase in the given coefficient.
    """
    prob_change = round((np.exp(coefficient) - 1) * 100, 2)
    direction = "increases" if prob_change > 0 else "decreases"
    print(f"{name}: likelihood {direction} by {abs(prob_change):.2f}%")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold: float = 0.5):
    """
    Generate predictions, compute accuracy, F1, specificity,
    sensitivity, and plot the confusion matrix.
    """
    probs = model.predict(X_test)
    y_pred = np.where(probs > threshold, 1, 0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1          = 2 * tp / (2 * tp + fp + fn)

    print(f"\nAccuracy    : {accuracy:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Profitable", "Profitable"])
    disp.plot(colorbar=False)
    plt.title("Confusion Matrix – Profitable Orders")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "profitable_orders.csv"

    df = load_data(DATA_PATH)
    df = create_binary_target(df)

    plot_distributions(df, ["discount_rate", "average_item_value", "items_per_order"])
    df = remove_outliers(df)

    X_train, X_test, y_train, y_test = preprocess(df)

    model = fit_logit(X_train, y_train)

    print("\nCoefficient Interpretations:")
    interpret_coefficient(0.23, name="new_customer")
    interpret_coefficient(0.02, name="average_item_value (per unit)")

    evaluate(model, X_test, y_test)
