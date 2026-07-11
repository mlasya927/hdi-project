# A Comprehensive Measure of Well-Being (HDI Predictor)

A machine learning project that predicts a country's **Human Development
Index (HDI) category** — Very High, High, Medium, or Low — from four core
indicators: Life Expectancy, Mean Years of Schooling, Expected Years of
Schooling, and GNI per Capita. Includes a Flask web app for interactive
predictions.

## Project Structure
```
hdi_project/
├── data/
│   ├── generate_dataset.py      # Epic 3: creates the dataset
│   └── hdi_dataset.csv          # generated dataset (1200 rows)
├── model/
│   ├── hdi_model.pkl            # Epic 7: trained RandomForest model
│   ├── scaler.pkl               # feature scaler
│   ├── label_encoder.pkl        # target label encoder
│   ├── confusion_matrix.png     # evaluation chart
│   └── feature_importance.png   # evaluation chart
├── templates/
│   ├── index.html               # input form page
│   └── result.html              # prediction result page
├── static/
│   └── style.css
├── preprocessing.py              # Epic 2 & 4: libraries, cleaning, encoding
├── train_model.py                # Epic 5 & 6 & 7: split, train, save
├── app.py                        # Epic 8: Flask web application
├── requirements.txt
└── README.md
```

## Epics Covered
| Epic | Description | File |
|---|---|---|
| 1 | Environment Setup & Package Installation | `requirements.txt` |
| 2 | Importing Required Libraries | `preprocessing.py` |
| 3 | Dataset Download and Understanding | `data/generate_dataset.py` |
| 4 | Data Preprocessing and Label Encoding | `preprocessing.py` |
| 5 | Train/Test Split | `train_model.py` |
| 6 | Fitting the Model | `train_model.py` |
| 7 | Saving the Model | `train_model.py` (outputs `model/*.pkl`) |
| 8 | Building the Flask Web Application | `app.py`, `templates/`, `static/` |

## How to Run

### 1. Setup environment
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_dataset.py
```

### 3. Train the model
```bash
python train_model.py
```
This prints accuracy/classification report and saves the trained model,
scaler, and label encoder into `model/`.

### 4. Run the Flask app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser, enter the four indicator
values, and click **Predict HDI Category**.

## Model
- **Algorithm:** Random Forest Classifier (`n_estimators=200, max_depth=8`)
- **Accuracy:** ~87% on held-out test data
- **Target classes:** Very High, High, Medium, Low

## Example Scenarios (from the project brief)
| Scenario | Life Exp. | Mean School | Expected School | GNI/Capita | Predicted |
|---|---|---|---|---|---|
| Very High Development | 82 | 13 | 17 | $55,000 | Very High |
| Emerging Economy | 68 | 8 | 12 | $8,000 | Medium |
| Needs Intervention | 52 | 3 | 6 | $1,200 | Low |

## Notes
- The dataset is synthetically generated using UNDP-style HDI methodology
  (geometric mean of normalized Life, Education, and Income indices) since
  it needed to be self-contained and reproducible. You can swap in a real
  UNDP HDI dataset by replacing `data/hdi_dataset.csv` — as long as it has
  the same column names, no other code changes are needed.
- To deploy, update the GitHub repo link and record a short demo video of
  the Flask app, then submit both via the **Overview** tab in SkillWallet
  (Add GitHub Link / Add Demo Link) as per the submission guide.
