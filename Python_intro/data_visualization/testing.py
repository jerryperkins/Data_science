import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import folium
import plotly.express as px

olympians = pd.read_csv('data/athleteEventsNoPersonal.csv')
linear = pd.read_csv('data/linearRegPredicted.csv')
with_without = pd.read_csv('data/linearWithWithout.csv')

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

# plt.plot(feature, predicted, c = 'r', label = 'Prediction')
# plt.scatter(feature, actual, c = 'k', label = 'Actual')
# plt.legend(loc=(1.02,0))
# plt.tight_layout()
# plt.savefig('MATLABlegendcutoff.png', dpi = 300)

# fig, axes = plt.subplots(nrows=1, ncols=1)
# axes.plot(feature, predicted, c = 'b', label = "Prediction")
# axes.scatter(feature, actual, c = 'k', label="Actual")
# axes.legend(loc =(1.02, 0))
# fig.tight_layout()
# plt.savefig('OOPlegendcutoff.png', dpi = 300)

# plt.show()

print(with_without.head())

intercept_filter = with_without['intercept'] == True
df_intercept = with_without.loc[intercept_filter, :]
df_no_intercept = with_without.loc[~intercept_filter, :]

# plt.figure(figsize=(8,4))
# #subplot 1
# plt.subplot(1,2,1) #rows, columns, which plot
# plt.plot(df_intercept['feature'].values, df_intercept['predicted'].values, c = 'r')
# plt.scatter(df_intercept['feature'].values, df_intercept['actual'].values, c = 'k')
# plt.title('intercept', fontsize = 12)
# #subplot 2 
# plt.subplot(1,2,2)
# plt.plot(df_no_intercept['feature'].values, df_no_intercept['predicted'].values, c = 'b')
# plt.scatter(df_no_intercept['feature'].values, df_no_intercept['actual'].values, c = 'y')
# plt.title('no intercept', fontsize = 12)

# fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(8,4))
# #subplot1
# axes[0].plot(df_intercept['feature'].values, df_intercept['predicted'], c = 'r')
# axes[0].scatter(df_intercept['feature'].values, df_intercept['actual'], c = 'k')
# axes[0].set_title('intercept', fontsize = 12)
# #subplot2
# axes[1].plot(df_no_intercept['feature'].values, df_no_intercept['predicted'], c = 'b')
# axes[1].scatter(df_no_intercept['feature'].values, df_no_intercept['actual'], c = 'y')
# axes[1].set_title('intercept', fontsize = 12)
# plt.show()


print(olympians.head())
year_filter = olympians['Year'] == 2016
top_20_height = olympians.loc[year_filter, :].groupby(['NOC'])['Height'].mean().sort_values(ascending = False).head(20)
print(top_20_height)

# plt.bar(top_20_height.index, top_20_height.values)
# plt.xticks(rotation = 90)


# print(top_20_height.plot.bar())
# plt.ylabel('Average Height (cm)')
# plt.show()

# sns.barplot(x = top_20_height.index, y = top_20_height.values)
# plt.xticks(rotation = 90)
# plt.show()

# map = folium.Map(location = [39.742043, -104.991531])
# map.save('index.html')

df = px.data.gapminder()
fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
                            hover_name="country", log_x=True, size_max=60)
fig.show()