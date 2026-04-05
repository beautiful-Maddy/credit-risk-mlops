import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess():

    df = pd.read_csv("data/loan_data.csv")

    # One hot encoding
    df = pd.get_dummies(df, columns=['purpose'], drop_first=True)

    # Features / Target
    X = df.drop('not.fully.paid', axis=1)
    y = df['not.fully.paid']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)