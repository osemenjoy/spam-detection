import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The dataset has no header, so we name the columns manually
# 57 features + 1 label
cols = [f"feature_{i}" for i in range(57)] + ["label"]

df = pd.read_csv("spambase.data", header=None, names=cols)

# Basic exploration
print(df.shape)          # (4601, 58)
print(df["label"].value_counts())  # 0 = legit, 1 = spam
print(df.describe())
print(df.isnull().sum())  # Check for missing values

# Visualise class balance
df["label"].value_counts().plot(kind="bar", color=["steelblue","tomato"])
plt.title("Spam vs Legitimate Emails")
plt.xticks([0,1], ["Legit", "Spam"], rotation=0)
plt.savefig("class_balance.png")
plt.show()