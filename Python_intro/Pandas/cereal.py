import pandas as pd

filename = 'cereal.xlsx'

cereal = pd.read_excel(filename, header= 1)

print(cereal.head())

print(cereal.tail())

print(cereal.dtypes)
print(cereal.info())

print(cereal.shape)

print(cereal['name'])
print(cereal[['name']])

print(cereal.loc[50:55, ['name']])