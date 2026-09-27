"""The 'naive' baseline: train a classifier to predict company survival
using only the observable subset of companies -- exactly the survivorship-
bias mechanism this project is about. Uses only features knowable at the
company's FIRST filing (no lookahead)."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_panel import load_all_quarters
from survivorship import build_survival_labels, SURVIVAL_YEARS


def build_first_filing_features(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per company: features from its FIRST filing only."""
    panel = panel.sort_values("FILING_DATE")
    first = panel.groupby("CIK").first().reset_index()

    first["TOTALOFFERINGAMOUNT"] = pd.to_numeric(
        first["TOTALOFFERINGAMOUNT"], errors="coerce"  # "Indefinite" etc -> NaN
    )
    first["log_offering_amount"] = np.log1p(first["TOTALOFFERINGAMOUNT"])

    return first[["CIK", "STATEORCOUNTRY", "ENTITYTYPE", "INDUSTRYGROUPTYPE",
                   "log_offering_amount"]]


def run_naive_model():
    panel = load_all_quarters()
    panel = panel[panel["is_operating_company"]]

    as_of = panel["FILING_DATE"].max()
    labels = build_survival_labels(panel, as_of)
    labels = labels[labels["observable"]]  # <-- the naive/biased step: silently
                                            #     drop everything not yet observable

    features = build_first_filing_features(panel)

    data = labels.merge(features, on="CIK", how="inner")
    data["survived"] = data["survived"].astype(int)

    X = data[["STATEORCOUNTRY", "ENTITYTYPE", "INDUSTRYGROUPTYPE", "log_offering_amount"]]
    y = data["survived"]

    categorical = ["STATEORCOUNTRY", "ENTITYTYPE", "INDUSTRYGROUPTYPE"]
    numeric = ["log_offering_amount"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])
    model = Pipeline([("preprocess", preprocess), ("clf", LogisticRegression(max_iter=1000))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"Survival horizon: {SURVIVAL_YEARS} years")
    print(f"Companies used (observable, operating): {len(data)}")
    print(f"Base rate (fraction survived): {y.mean():.3f}")
    print(f"Naive model accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"Naive model AUC: {roc_auc_score(y_test, probs):.3f}")
    majority_baseline_acc = max(y.mean(), 1 - y.mean())
    print(f"Majority-class baseline accuracy (always guess 'not survived'): {majority_baseline_acc:.3f}")
    print(f"--> model's accuracy edge over the dumb baseline: {accuracy_score(y_test, preds) - majority_baseline_acc:+.3f}")
    return model, data


if __name__ == "__main__":
    run_naive_model()