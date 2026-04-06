import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.features.preprocessing import preprocess


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Credit Risk - Logistic Regression")


def train():
    X_train, X_test, y_train, y_test = preprocess()

    params = {
        "max_iter": 1000,
    }

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("model", "Logistic Regression")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        os.makedirs("artifacts", exist_ok=True)

        cm_path = "artifacts/logistic_confusion_matrix.png"
        roc_path = "artifacts/logistic_roc_curve.png"

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        plt.title("Logistic Regression - Confusion Matrix")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        plt.title("Logistic Regression - ROC Curve")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)


if __name__ == "__main__":
    train()