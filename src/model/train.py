import logging
import pickle
from pathlib import Path
import sys
sys.path.append('.')

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

from src.data.data_loader import load_data, validate_data
from src.data.preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def train(data_path: str = "src/data/creditcard.csv"):
    df = load_data(data_path)
    df = validate_data(df)
    X_train, X_test, y_train, y_test = preprocess(df)

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 42,
            "scale_pos_weight": len(y_train[y_train==0]) / len(y_train[y_train==1])
        }

        # Log params
        mlflow.log_params(params)

        # Train
        logger.info("Training XGBoost model...")
        model = XGBClassifier(**params, eval_metric='logloss')
        model.fit(X_train, y_train)
        logger.info("Training complete ✅")

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log metrics
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        # Log model
        mlflow.xgboost.log_model(model, "fraud_model")

        logger.info(f"Precision : {precision:.4f}")
        logger.info(f"Recall    : {recall:.4f}")
        logger.info(f"F1 Score  : {f1:.4f}")
        logger.info(f"ROC-AUC   : {roc_auc:.4f}")

        # Save locally too
        Path("models").mkdir(exist_ok=True)
        with open("models/fraud_model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved ✅")
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    return model

if __name__ == "__main__":
    train()