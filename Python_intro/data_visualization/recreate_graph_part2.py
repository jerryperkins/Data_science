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



plt.rcParams['figure.figsize'] = 7,4 #https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
plt.plot(months, payments_3.cumsum(), label = '30 Year 3% Apr', c = 'b')
plt.plot(months, payments_5.cumsum(), label = '30 Year 5% Apr', c = 'k')
plt.ylim(bottom=0, top=400000)
plt.xlim(left = 0, right =  360)
plt.xlabel('Month')
plt.ylabel('Dollars')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.legend(loc='lower right')
plt.title('Cumlative Interest Paid')
plt.grid()
plt.show()
