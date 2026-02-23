import pandas as pd

df = pd.read_csv('dataset.csv', sep=",", header=None)
print(df.head())

trasp = df.transpose()
trasp["label"] = 0
trasp.rename(columns={0 : "string"}, inplace=True)
print(trasp.head())

