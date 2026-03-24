# =========================
# IMPORTS
# =========================
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# =========================
# LOAD DATA
# =========================
cols = [f"feature_{i}" for i in range(57)] + ["label"]
df = pd.read_csv("spambase.data", header=None, names=cols)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision: {precision_score(y_test, preds):.4f}")
    print(f"  Recall:    {recall_score(y_test, preds):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Legit","Spam"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Spam"],
                yticklabels=["Legit","Spam"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"confusion_{name.replace(' ','_')}.png")
    plt.show()


# =========================
# PROGRESSIVE DATASET TESTING
# =========================
dataset_sizes = [0.1, 0.3, 0.6, 1.0]

for size in dataset_sizes:
    print(f"\n\n🚀 Training with {int(size*100)}% of dataset")

    subset = df.sample(frac=size, random_state=42)

    X = subset.drop("label", axis=1)
    y = subset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # MODEL 1: Multinomial NB
    # =========================
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)  # NOTE: no scaling for NB

    # =========================
    # MODEL 2: SVM (TUNED)
    # =========================
    svm_params = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"]
    }

    svm_grid = GridSearchCV(SVC(probability=True), svm_params,
                            cv=3, scoring="f1", n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    svm_model = svm_grid.best_estimator_

    print("Best SVM Params:", svm_grid.best_params_)

    # =========================
    # MODEL 3: Logistic Regression (TUNED)
    # =========================
    lr_params = {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs"]
    }

    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000),
                           lr_params, cv=3, scoring="f1", n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    lr_model = lr_grid.best_estimator_

    print("Best LR Params:", lr_grid.best_params_)

    # =========================
    # EVALUATION
    # =========================
    evaluate_model("Naive Bayes", nb_model, X_test, y_test)
    evaluate_model("SVM", svm_model, X_test_scaled, y_test)
    evaluate_model("Logistic Regression", lr_model, X_test_scaled, y_test)

    # =========================
    # SAVE BEST MODEL (only for full dataset)
    # =========================
    if size == 1.0:
        joblib.dump(lr_model, "best_spam_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        print("✅ Model saved!")


# =========================
# SAMPLE PREDICTION FUNCTION
# =========================
def predict_sample(sample, model, scaler):
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)
    return "Spam" if prediction[0] == 1 else "Legit"


# Example usage:
# print(predict_sample(X_test.iloc[0], lr_model, scaler))