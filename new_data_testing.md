def predict_email(features_array, model, scaler):
    """
    features_array: a list of 57 numerical values (word frequencies)
    """
    sample = np.array(features_array).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    result = model.predict(sample_scaled)
    return "🚨 SPAM" if result[0] == 1 else "✅ Legitimate"

# Example: use the first row of X_test as a demo
sample = X_test.iloc[0].values
print(predict_email(sample, svm_model, scaler))


# progressive testing on new data
for size in [0.1, 0.25, 0.5, 0.75, 1.0]:
    X_sub = X_train_scaled[:int(len(X_train_scaled) * size)]
    y_sub = y_train.iloc[:int(len(y_train) * size)]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_sub, y_sub)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Training size {int(size*100)}% → Accuracy: {acc:.4f}")