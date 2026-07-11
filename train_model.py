"""
Epic 5: Train/Test Split
Epic 6: Model Fitting
Epic 7: Save the Trained Model
--------------------------------------------------------------------
Trains a RandomForestClassifier to predict HDI Category from the
4 development indicators, evaluates it, and saves the trained
model + scaler + label encoder to disk using pickle so the Flask
app can load them later.
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_preprocess

# ---------------------------------------------------------
# Epic 5: Load data & split into train/test sets
# ---------------------------------------------------------
X, y, scaler, label_encoder, feature_cols = load_and_preprocess(
    csv_path="data/hdi_dataset.csv"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ---------------------------------------------------------
# Epic 6: Fit the model
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---- Confusion Matrix plot (saved as image, useful for report) ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("HDI Category - Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
print("Confusion matrix saved to model/confusion_matrix.png")

# ---- Feature importance plot ----
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=feature_cols)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("model/feature_importance.png")
print("Feature importance chart saved to model/feature_importance.png")

# ---------------------------------------------------------
# Epic 7: Save model, scaler, and label encoder
# ---------------------------------------------------------
with open("model/hdi_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel, scaler, and label encoder saved in the 'model/' folder.")
