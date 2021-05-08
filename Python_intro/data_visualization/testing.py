import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

olympians = pd.read_csv('data/athleteEventsNoPersonal.csv')

x = [10,20,30,40,50]
y = [5,10,15,20,25]

# plt.show()
# plt.plot([10,20,30,40,50], [100,400,900,1600,2500])
# print(plt.style.available)
# plt.style.use('seaborn')
# plt.plot(x,y)
# plt.show()

number_unique_year = olympians.groupby(['Year'])['ID'].nunique()
print(number_unique_year)
# unique_years = list(number_unique_year.index)
# num_olympians = list(number_unique_year.values)

# plt.style.use('seaborn')
# plt.plot(unique_years, num_olympians, marker = '.', markersize = 10, c = 'magenta')
# plt.show()

unique_years = np.array(number_unique_year.index)
num_oympians = np.array(number_unique_year.values)
print(unique_years, num_oympians)

plt.style.use('seaborn')
# plt.plot(unique_years, num_oympians, c = 'k')
# plt.show()

fig, axes = plt.subplots(nrows = 1, ncols=1)
axes.plot(unique_years, num_oympians, c = 'y')
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.show()

