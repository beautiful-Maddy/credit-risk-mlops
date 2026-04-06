import os
import joblib

from sklearn.linear_model import LogisticRegression
from src.features.preprocessing import preprocess


def save_model():
    X_train, X_test, y_train, y_test = preprocess()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.pkl")

    print("Model saved in models/logistic_model.pkl")


if __name__ == "__main__":
    save_model()