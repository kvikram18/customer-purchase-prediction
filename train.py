"""Train and evaluate customer purchase prediction models."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_data.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    target = "purchased"
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.columns.tolist()
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        )
    }

    results = {}
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, score)),
            "classification_report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            )
        }
        fitted[name] = pipe

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    best_score = best_model.predict_proba(X_test)[:, 1]
    joblib.dump(best_model, MODELS / "best_model.joblib")

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "results": results}, f, indent=2)

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt="d",
                cmap="Blues", xticklabels=["No purchase", "Purchase"],
                yticklabels=["No purchase", "Purchase"])
    plt.title(f"Confusion Matrix – {best_name.replace('_', ' ').title()}")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close()

    # Target distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=target)
    plt.xticks([0, 1], ["No purchase", "Purchase"])
    plt.title("Customer Purchase Distribution")
    plt.tight_layout()
    plt.savefig(OUT / "purchase_distribution.png", dpi=160)
    plt.close()

    # Feature importance for random forest if it wins; otherwise coefficient magnitude
    model_step = best_model.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importance = model_step.feature_importances_
    else:
        importance = abs(model_step.coef_[0])
    imp = pd.Series(importance, index=numeric_features).sort_values()
    plt.figure(figsize=(7, 4))
    imp.plot(kind="barh")
    plt.title(f"Feature Importance – {best_name.replace('_', ' ').title()}")
    plt.xlabel("Importance (coefficient magnitude or tree importance)")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=160)
    plt.close()

    print(f"Selected model: {best_name}")
    for name, item in results.items():
        print(f"{name}: accuracy={item['accuracy']:.3f}, ROC-AUC={item['roc_auc']:.3f}")
    print(f"Outputs saved in: {OUT}")

if __name__ == "__main__":
    main()
