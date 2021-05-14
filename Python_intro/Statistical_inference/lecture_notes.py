import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sales = pd.read_excel('data/Irish Whiskey Sales by Volume.xlsx')

print(sales.head())

print(sales.info())

# print(sales['Cases'].describe().apply(lambda s: s.apply('{0:.2f}'.format)))

print(sales['Country'].value_counts())

print(sales['Category'].value_counts())

print(sales.isnull().sum())

print(sales[sales.isnull().any(axis=1)]['Country'].value_counts())

# sales['Cases'].hist(bins = 20)

# sns.boxplot(x = sales['Cases'])

us_sales = sales.loc[sales['Country'] == 'United States', :]
print(us_sales)

print(us_sales['Quality'].value_counts())

us_sales['quality_col'] = us_sales['Quality'].map({'Super Premium': 'blue', 'Standard': 'red', 'Premium': 'green'})

plt.scatter(us_sales['Year'], us_sales['Cases'], c = us_sales['quality_col'])

# us_sales['quality_col'] = us_sales['Quality'].map({'Super Premium': 'blue', 'Standard': 'orange', 'Premium': 'green'})

# plt.scatter(us_sales['Year'], us_sales['Cases'], c = us_sales['quality_col'])


plt.show()
