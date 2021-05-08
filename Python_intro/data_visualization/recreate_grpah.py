import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mort = pd.read_csv('data/mortgages.csv')

print(mort.head())
mort_30yr = mort['Mortgage Name']=='30 Year'
mort_30yr = mort.loc[mort_30yr]
# print(mort_30yr)
per_3 = mort_30yr['Interest Rate']== 0.03
per_5 = mort_30yr['Interest Rate']== 0.05
payments_3 = mort_30yr.loc[per_3]['Interest Paid'].values
payments_5 = mort_30yr.loc[per_5]['Interest Paid'].values
months = mort_30yr.loc[per_3]['Month'].values


plt.style.use('seaborn')
plt.plot(months, payments_3.cumsum())
plt.plot(months, payments_5.cumsum())
plt.show()
