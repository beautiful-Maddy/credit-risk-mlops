import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.features.preprocessing import preprocess


def train_logistic():

    X_train, X_test, y_train, y_test = preprocess()

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Classification Report")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("ROC AUC Score")
    print(roc_auc_score(y_test, y_pred))


if __name__ == "__main__":
    train_logistic()