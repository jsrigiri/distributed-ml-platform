import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from core.config import settings
from models.evaluate import evaluate_classifier


def train_models(df: pd.DataFrame, feature_columns):
    X = df[feature_columns].copy()

    y_clf = df["target_clf"]
    y_reg = df["target_reg"]

    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y_clf
    )

    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_reg,
        test_size=settings.test_size,
        random_state=settings.random_state
    )

    results = {}

    # -------------------
    # Logistic Regression
    # -------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_clf_train)

    clf_pred = lr.predict(X_test)
    results["logistic_regression"] = {
        "model": lr,
        "metrics": evaluate_classifier(y_clf_test, clf_pred)
    }

    # -------------------
    # XGBoost Classifier
    # -------------------
    xgb_clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=settings.random_state,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_clf_train)

    clf_pred = xgb_clf.predict(X_test)
    results["xgboost_classifier"] = {
        "model": xgb_clf,
        "metrics": evaluate_classifier(y_clf_test, clf_pred)
    }

    # -------------------
    # LightGBM Classifier
    # -------------------
    lgbm_clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=settings.random_state
    )
    lgbm_clf.fit(X_train, y_clf_train)

    clf_pred = lgbm_clf.predict(X_test)
    results["lightgbm_classifier"] = {
        "model": lgbm_clf,
        "metrics": evaluate_classifier(y_clf_test, clf_pred)
    }

    # -------------------
    # XGBoost Regressor
    # -------------------
    xgb_reg = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=settings.random_state
    )
    xgb_reg.fit(X_train, y_reg_train)

    results["xgboost_regressor"] = {
        "model": xgb_reg,
        "metrics": {
            "rmse": float(((xgb_reg.predict(X_test) - y_reg_test) ** 2).mean() ** 0.5)
        }
    }

    # -------------------
    # LightGBM Regressor
    # -------------------
    lgbm_reg = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=settings.random_state
    )
    lgbm_reg.fit(X_train, y_reg_train)

    results["lightgbm_regressor"] = {
        "model": lgbm_reg,
        "metrics": {
            "rmse": float(((lgbm_reg.predict(X_test) - y_reg_test) ** 2).mean() ** 0.5)
        }
    }

    return results