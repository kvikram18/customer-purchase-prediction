# Project Report: Customer Purchase Prediction

## 1. Abstract
This project demonstrates an end-to-end machine learning workflow for predicting customer purchase behavior. A synthetic dataset containing demographic and website-engagement features is used to train and compare Logistic Regression and Random Forest classifiers. Performance is assessed using accuracy, ROC-AUC, precision, recall, F1-score and a confusion matrix. The selected model is saved for reuse in an example prediction script.

## 2. Problem Statement
Online businesses want to understand which visitors may make a purchase. Manually reviewing every customer is inefficient. A classification model can learn patterns from historical examples and estimate the likelihood of purchase. This project demonstrates the workflow using generated educational data.

## 3. Objectives
- Explore customer-related numerical data.
- Prepare features for machine learning.
- Train two classification algorithms.
- Compare their performance on unseen test data.
- Visualize results and demonstrate a prediction.

## 4. Dataset Description
The CSV contains 1,200 synthetic rows and seven columns: age, annual income, website time, previous purchases, website visits, discount received and the purchase target. The target is binary: 1 means purchase and 0 means no purchase.

## 5. Methodology
1. Read the CSV using Pandas.
2. Separate predictors from the target.
3. Split the dataset into 80% training and 20% testing data with stratification.
4. Impute missing numerical values using the median and scale numerical features.
5. Train Logistic Regression and Random Forest pipelines.
6. Evaluate predictions with accuracy, ROC-AUC and a classification report.
7. Select the model with the higher test ROC-AUC, save it, and generate plots.

## 6. Algorithms
**Logistic Regression:** A baseline classification algorithm that estimates class probabilities using a logistic function.

**Random Forest:** An ensemble of decision trees whose predictions are combined. It can model non-linear relationships and feature interactions.

## 7. Evaluation
Run `python train.py` to generate the measured results in `outputs/metrics.json`. The script also saves a confusion matrix, purchase distribution chart and feature-importance/coefficient chart. Report the actual values from that file; do not insert assumed results.

## 8. Conclusion
The project provides a reproducible demonstration of data preparation, model training, comparison and evaluation for a binary classification problem. The best model is saved and can be used by `predict.py`. Since the data is synthetic, this work is educational and should not be used to make decisions about real customers.

## 9. Future Scope
The project can be extended with real consented data, cross-validation, hyperparameter optimization, probability calibration, fairness testing and a Streamlit user interface.

## 10. References
- Pandas documentation: https://pandas.pydata.org/docs/
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Matplotlib documentation: https://matplotlib.org/stable/
- Seaborn documentation: https://seaborn.pydata.org/
