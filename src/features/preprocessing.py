import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = None


def preprocess():
    global FEATURE_COLUMNS

    df = pd.read_csv("data/loan_data.csv")

    df = pd.get_dummies(df, columns=["purpose"], drop_first=True)

    X = df.drop("not.fully.paid", axis=1)
    y = df["not.fully.paid"]

    FEATURE_COLUMNS = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def get_feature_columns():
    df = pd.read_csv("data/loan_data.csv")
    df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
    X = df.drop("not.fully.paid", axis=1)
    return X.columns.tolist()


def fit_scaler_and_columns():
    df = pd.read_csv("data/loan_data.csv")
    df = pd.get_dummies(df, columns=["purpose"], drop_first=True)

    X = df.drop("not.fully.paid", axis=1)

    scaler = StandardScaler()
    scaler.fit(X)

    feature_columns = X.columns.tolist()
    return scaler, feature_columns