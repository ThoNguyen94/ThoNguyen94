import pandas as pd
import numpy  as np

# Doc data frame
df = pd.read_csv("data\subset-covid-data.csv")
# print(df.head())

# phan tich logic
df_group = df.groupby(['continent']).agg({'day':lambda x: x.nunique(), "population":"sum"})

df_group = df.groupby('country')["cases"].sum().sort_values(ascending=False).head(5)
print(df_group)
