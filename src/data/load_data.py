import pandas as pd

def load_data():
    df = pd.read_csv("data/loan_data.csv")
    return df

if __name__ == "__main__":
    df = load_data()

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    print("\nPreview:")
    print(df.head())