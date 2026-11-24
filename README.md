# Customer Purchase Prediction

An end-to-end data science and machine learning project that predicts whether a customer is likely to purchase based on customer and website-engagement attributes.

## Project objective
Businesses can use customer behavior data to understand purchase patterns and prioritize marketing efforts. This project compares Logistic Regression and Random Forest classifiers and evaluates them on a held-out test set.

## Dataset
`data/customer_data.csv` contains 1,200 **synthetic** customer records. It is generated for educational demonstration and does not contain real customer information.

| Column | Description |
|---|---|
| `age` | Customer age in years |
| `annual_income` | Annual income in currency units (synthetic) |
| `website_time_minutes` | Time spent on the website |
| `previous_purchases` | Number of previous purchases |
| `website_visits` | Number of website visits |
| `discount_received` | 1 if a discount was received, otherwise 0 |
| `purchased` | Target: 1 for purchase, 0 for no purchase |

## Tools and libraries
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib

## Project workflow
1. Load and inspect the dataset.
2. Separate features and target.
3. Split data into training and test sets using stratification.
4. Build preprocessing and classification pipelines.
5. Train Logistic Regression and Random Forest.
6. Compare accuracy and ROC-AUC.
7. Save evaluation metrics and visualizations.
8. Save the selected model and run an example prediction.

## Run locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/customer-purchase-prediction.git
cd customer-purchase-prediction
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train and evaluate
```bash
python train.py
```
This creates `outputs/metrics.json`, three chart images, and `models/best_model.joblib`.

### 5. Run an example prediction
```bash
python predict.py
```
Run `train.py` before `predict.py`, because training creates the saved model.

## Evaluation
The script reports:
- Accuracy
- ROC-AUC
- Precision, recall and F1-score
- Confusion matrix

The actual scores are generated when you run the training script and depend on the supplied synthetic dataset. Do not claim a score without running the code.

## Repository structure
```text
customer-purchase-prediction/
├── data/
│   └── customer_data.csv
├── outputs/                 # generated after training
├── models/                  # generated after training
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations and responsible use
- The dataset is synthetic; results are not evidence of real customer behavior.
- Synthetic patterns may be simpler than real purchasing decisions.
- A real deployment would require representative, consented data, privacy safeguards, bias checks, and ongoing monitoring.
- Model predictions are estimates, not guarantees.

## Future improvements
- Test on a real, appropriately licensed customer dataset.
- Add cross-validation and hyperparameter tuning.
- Compare additional models and calibrate probabilities.
- Build a Streamlit dashboard for interactive predictions.
