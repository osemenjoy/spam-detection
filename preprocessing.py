from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)

# The dataset has no header, so we name the columns manually
# 57 features + 1 label
cols = [f"feature_{i}" for i in range(57)] + ["label"]

df = pd.read_csv("spambase.data", header=None, names=cols)

# Separate features and label
X = df.drop("label", axis=1)
y = df["label"]

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features — important for SVM and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")


# Model 1: Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Model 2: Support Vector Machine
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Model 3: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

print("All models trained ✅")


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

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Spam"],
                yticklabels=["Legit","Spam"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"confusion_{name.replace(' ','_')}.png")
    plt.show()

evaluate_model("Naive Bayes",        nb_model,  X_test_scaled, y_test)
evaluate_model("SVM",                svm_model, X_test_scaled, y_test)
evaluate_model("Logistic Regression",lr_model,  X_test_scaled, y_test)