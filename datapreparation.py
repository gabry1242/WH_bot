import pandas as pd

df = pd.read_csv('dataset.csv', sep=",", header=None)
print(df.head())

trasp = df.transpose()
trasp["label"] = 0
trasp.rename(columns={0 : "string"}, inplace=True)
trasp["label"][123:] = 1
print(trasp.head())

trasp.to_csv('processed_dataset.csv', index=False)

processed_df = pd.read_csv('processed_dataset.csv', sep=",")
print(processed_df)