"""
Epic 3: Dataset Generation for HDI (Human Development Index) Project
----------------------------------------------------------------------
This script generates a realistic country-level dataset with the four
core HDI indicators:
    1. Life Expectancy at Birth (years)
    2. Mean Years of Schooling
    3. Expected Years of Schooling
    4. GNI per Capita (PPP $)

Based on these indicators, an HDI score (0-1) is computed using the
official UNDP-style methodology (geometric mean of normalized
dimension indices), and countries are classified into 4 categories:
    - Very High  (HDI >= 0.800)
    - High       (0.700 <= HDI < 0.800)
    - Medium     (0.550 <= HDI < 0.700)
    - Low        (HDI < 0.550)

Run this file to create data/hdi_dataset.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 1200

# ---------------------------------------------------------
# Step 1: Simulate raw indicator values across a realistic range
# ---------------------------------------------------------
life_expectancy = np.random.normal(loc=68, scale=10, size=N_SAMPLES).clip(40, 85)
mean_schooling = np.random.normal(loc=8, scale=3, size=N_SAMPLES).clip(0, 15)
expected_schooling = mean_schooling + np.random.normal(loc=4, scale=2, size=N_SAMPLES)
expected_schooling = expected_schooling.clip(0, 20)
gni_per_capita = np.random.lognormal(mean=9.2, sigma=1.0, size=N_SAMPLES).clip(500, 90000)

# ---------------------------------------------------------
# Step 2: Normalize each dimension (UNDP min-max method)
# ---------------------------------------------------------
def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

life_index = normalize(life_expectancy, 20, 85)
edu_index = (normalize(mean_schooling, 0, 15) + normalize(expected_schooling, 0, 18)) / 2
income_index = normalize(np.log(gni_per_capita), np.log(100), np.log(75000))

life_index = life_index.clip(0, 1)
edu_index = edu_index.clip(0, 1)
income_index = income_index.clip(0, 1)

# ---------------------------------------------------------
# Step 3: HDI = geometric mean of the three dimension indices
# ---------------------------------------------------------
hdi_score = (life_index * edu_index * income_index) ** (1 / 3)

# ---------------------------------------------------------
# Step 4: Classify into HDI categories
# ---------------------------------------------------------
def classify_hdi(score):
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"

hdi_category = [classify_hdi(s) for s in hdi_score]

# ---------------------------------------------------------
# Step 5: Assemble final dataframe
# ---------------------------------------------------------
df = pd.DataFrame({
    "Life_Expectancy": np.round(life_expectancy, 2),
    "Mean_Years_Schooling": np.round(mean_schooling, 2),
    "Expected_Years_Schooling": np.round(expected_schooling, 2),
    "GNI_per_Capita": np.round(gni_per_capita, 2),
    "HDI_Score": np.round(hdi_score, 4),
    "HDI_Category": hdi_category
})

df.to_csv("/home/claude/hdi_project/data/hdi_dataset.csv", index=False)

print("Dataset generated successfully!")
print(f"Shape: {df.shape}")
print("\nClass distribution:")
print(df["HDI_Category"].value_counts())
print("\nSample rows:")
print(df.head())
