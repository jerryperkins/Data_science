import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

olympians = pd.read_csv('data/athleteEventsNoPersonal.csv')
linear = pd.read_csv('data/linearRegPredicted.csv')

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

# plt.style.use('seaborn')
# plt.plot(unique_years, num_oympians, c = 'k')
# plt.show()

# fig, axes = plt.subplots(nrows = 1, ncols=1)
# axes.plot(unique_years, num_oympians, c = 'y')
# plt.xticks(fontsize = 25)
# plt.yticks(fontsize = 25)
# plt.show()

# plt.plot(unique_years, num_oympians, c = 'k')
# plt.xlim(left=1892, right=2020)
# plt.ylim(bottom=0, top=1800)
# plt.xlabel('Year', fontsize=4)
# plt.ylabel('Number of Olympians', fontsize=4)
# plt.title('Number of Olympians Over the Years')
# plt.grid(c = 'g', alpha = .99, linestyle = 'dotted')


# fig, axes = plt.subplots(nrows = 1, ncols = 1)
# axes.plot(unique_years, num_oympians, c = 'y')
# axes.set_xlim(left=1892, right=2020)
# axes.set_ylim(bottom=0, top=1800)
# axes.set_xlabel('Year', fontsize=16)
# axes.set_ylabel('Number of Olympains', fontsize=16)
# axes.set_title('Number of Olympians Over the Years')
# axes.grid(c= 'r', alpha = .4, linestyle='-')

# plt.show()

print(linear.head())
feature = linear['feature'].values
actual = linear['actual'].values
predicted = linear['predicted'].values

plt.plot(feature, predicted, c = 'r', label = 'Prediction')
plt.scatter(feature, actual, c = 'k', label = 'Actual')
plt.legend(loc='best')

fig, axes = plt.subplots(nrows=1, ncols=1)
axes.plot(feature, predicted, c = 'b', label = "Prediction")
axes.scatter(feature, actual, c = 'k', label="Actual")
axes.legend(loc =(1.02, 0))

plt.show()


