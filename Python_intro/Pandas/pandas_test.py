import pandas as pd 

filename = 'bostonHousing1978.xlsx'
filename2 = 'bostonHousing1978.csv'
filename3 = 'mortgages.csv'

df1 = pd.read_excel(filename)
df2 = pd.read_csv(filename2)
df3 = pd.read_csv(filename3)
# print(test1)
# print(test2)

print(df1.head())
print(df1.tail())

print(df1.dtypes)

print(df1.info())

print(df1.shape) #comes in row,column

print(df1[['RM']].head())

print(df1[['RM','target']].head())

print(df1[['RM','LSTAT','PTRATIO','target']].tail())

print(df1['RM'].head()) #single brackets only allows for one column at a time
print(type(df1['RM'].head()))


print(df1['RM'][0:10])

print(df1.loc[:, ['RM']].head())
print(df1.loc[:, 'RM'].head())

print(df3.head())

print(df3['Mortgage Name'])
print(df3['Mortgage Name'].value_counts())

print(df3[df3['Mortgage Name'] == '30 Year'])
mortgage_filer = df3['Mortgage Name'] == '30 Year'
print(df3[mortgage_filer])
df3_30_year = df3[mortgage_filer]
print(df3_30_year)
print(df3_30_year.head())

print(df3.loc[df3['Mortgage Name'] == '30 Year', :])