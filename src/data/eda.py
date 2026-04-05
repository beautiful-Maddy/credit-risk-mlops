import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/loan_data.csv")

print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df['not.fully.paid'].value_counts())

sns.countplot(x='not.fully.paid', data=df)
plt.show()