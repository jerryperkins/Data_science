import pandas as pd 

filename = 'pandas_data/bostonHousing1978.xlsx'
filename2 = 'pandas_data/bostonHousing1978.csv'
filename3 = 'pandas_data/mortgages.csv'


df1 = pd.read_excel(filename)
df2 = pd.read_csv(filename2)
mort = pd.read_csv(filename3)
# print(test1)
# print(test2)

# print(df1.head())
# print(df1.tail())

# print(df1.dtypes)

# print(df1.info())

# print(df1.shape) #comes in row,column

# print(df1[['RM']].head())

# print(df1[['RM','target']].head())

# print(df1[['RM','LSTAT','PTRATIO','target']].tail())

# print(df1['RM'].head()) #single brackets only allows for one column at a time
# print(type(df1['RM'].head()))


# print(df1['RM'][0:10])

# print(df1.loc[:, ['RM']].head())
# print(df1.loc[:, 'RM'].head())

# print(df3.head())

# print(df3['Mortgage Name'])
# print(df3['Mortgage Name'].value_counts())

# print(df3[df3['Mortgage Name'] == '30 Year'])
# mortgage_filer = df3['Mortgage Name'] == '30 Year'
# print(df3[mortgage_filer])
# df3_30_year = df3[mortgage_filer]
# print(df3_30_year)
# print(df3_30_year.head())

# print(df3.loc[df3['Mortgage Name'] == '30 Year', :])

# print(mort.head())

# mort = mort.rename(columns={'Starting Balance': 'starting_balance', 'Interest Paid': 'interest_paid', 'Principal Paid': 'principal_paid'})
# print(mort.head())

# mort.columns = ['month',
#                 'starting_balance',
#                 'repayment',
#                 'interest_paid',
#                 'principal_paid',
#                 'new_balance',
#                 'mortgage_name',
#                 'interest_rate'
#                 ]

# print(mort.head())

# mort = mort.drop(columns=['new_balance'])
# print(mort.head())

# del mort['starting_balance']
# print(mort.head())

linear = pd.read_csv('pandas_data/linear.csv')

# print(linear.head())

# linear.info()

# print(linear['y'].isna().head())

# y_missing = linear['y'].isna()
# print(linear.loc[y_missing, :])
# print(linear.loc[~y_missing, :])

# print(linear['y'].isna().sum())


# print(linear.loc[0:10, :])
# print(linear.loc[0:10, :].dropna(how = 'any')) # what does all instead of any mean?


# print(linear.loc[0:10, 'y'])

# print(linear.loc[0:10, 'y'].fillna(0))

# print(linear.loc[0:10, 'y'].fillna(method="bfill"))

# print(linear.loc[0:10, 'y'].fillna(method="ffill"))

# print(linear.loc[0:10, 'y'].interpolate(method = 'linear')) #what is the interpolate doing??

# print(linear.to_numpy())
# print(linear.values)
# print(linear.to_dict())



# mort.to_csv(path_or_buf='pandas_data/oneMortgage.csv', index = False)
# mort.to_excel(excel_writer='pandas_data/oneMortgage.xlsx', index=False)

# mort_filter = mort['Mortgage Name']=='30 Year'
# interest_filter = mort['Interest Rate']==0.03
# mort = mort.loc[mort_filter & interest_filter, :]
# print(mort)
# print(mort.shape)

# print(mort['Interest Paid'].sum())
# print(mort.sum())

df_cities = pd.DataFrame(data = [['Seattle', 1],
                            ['Kirkland', 2],
                            ['Redmond', 3],
                            ['Seattle', 4],
                            ['Kirkland', 5],
                            ['Redmond', 6]], columns = ['key', 'data'])

print(df_cities)

print(df_cities.groupby(['key']).sum())

print(df_cities.groupby(['key'])[['data']].sum()) # this is the same thing as above ONLY becasue there are only two columns in this example

print(mort.head())

print(mort.groupby(['Mortgage Name', 'Interest Rate'])[['Interest Paid']].sum())