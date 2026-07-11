"""
Epic 8: Building the Flask Web Application
--------------------------------------------------------------------
Loads the trained model (from model/hdi_model.pkl) along with the
scaler and label encoder, and serves a simple web form where a user
can enter the 4 development indicators to get a predicted HDI
Category (Very High / High / Medium / Low).
"""

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---- Load trained artifacts ----
with open("model/hdi_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

FEATURE_ORDER = [
    "Life_Expectancy",
    "Mean_Years_Schooling",
    "Expected_Years_Schooling",
    "GNI_per_Capita",
]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---- Read form inputs ----
        life_expectancy = float(request.form["life_expectancy"])
        mean_schooling = float(request.form["mean_schooling"])
        expected_schooling = float(request.form["expected_schooling"])
        gni_per_capita = float(request.form["gni_per_capita"])

        input_values = np.array([[life_expectancy, mean_schooling,
                                   expected_schooling, gni_per_capita]])

        # ---- Scale using the SAME scaler used during training ----
        input_scaled = scaler.transform(input_values)

        # ---- Predict ----
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        # ---- Probability / confidence ----
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        return render_template(
            "result.html",
            prediction=prediction_label,
            confidence=confidence,
            life_expectancy=life_expectancy,
            mean_schooling=mean_schooling,
            expected_schooling=expected_schooling,
            gni_per_capita=gni_per_capita,
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
