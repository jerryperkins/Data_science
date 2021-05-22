import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Formatting Data for ML

# data = load_iris()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['species'] = data.target
# print(df.head())

# feature_names = ['sepal length (cm)',
#                 'sepal width (cm)',
#                 'petal length (cm)',
#                 'petal width (cm)']
# print(df.loc[:, feature_names])

# X = df.loc[:, feature_names].to_numpy()
# print(X.shape)
# print(X)

# print(df.loc[:, 'species'])
# y = df.loc[:, 'species'].values
# print(y.shape)

# # Linear Regresssion

# bh = pd.read_csv('data/modifiedBostonHousing.csv')
# print(bh.head(), bh.shape)

# bh = bh.loc[:, ['RM', 'LSTAT', 'PTRATIO', 'price']]
# print(bh.head())
# print(bh.shape)
# print(bh.info())
# print(bh.isnull().sum())
# bh = bh.dropna(how = 'any')
# print(bh.info())

# print(bh.corr())
# print(bh.corr().sort_values(by = ['price']))

# sns.heatmap(bh.corr().sort_values(by = ['price']), cmap = sns.diverging_palette(240, 10, n=9)) 
# # plt.show()

# bh.loc[:, :].hist(bins=25, figsize=[16,16], xlabelsize='10', ylabelsize='10', xrot=-15)
# # plt.show()

# price_filer = bh.loc[:, 'price'] < 0
# bh = bh.loc[~price_filer, :]
# bh.loc[:, :].hist(bins=25, figsize=[16,16], xlabelsize='10', ylabelsize='10', xrot=-15)
# # plt.show()

# print(bh.corr())

# # the below figures only work in colab

# # fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10,2), dpi=1000)
# # sns.regplot(x='RM', y='price', data=bh, ci=None, ax = axes[0], scatter_kws={'alpha':0.3})
# # sns.regplot(x='LSTAT', y='price', data=bh, ci=None, ax = axes[1], scatter_kws={'alpha':0.3})
# # sns.regplot(x='PTRATIO', y='price', data=bh, ci=None, ax = axes[2], scatter_kws={'alpha':0.3})
# # plt.tight_layout()
# # plt.show()

# X = bh.loc[:, ['RM', 'LSTAT', 'PTRATIO']].values
# print(X.shape)
# y = bh.loc[:, 'price'].values
# print(y.shape)

# reg = LinearRegression(fit_intercept=True)
# reg.fit(X,y)
# print(reg.predict(X[0].reshape(-1,3)))
# print(reg.predict(X[0:10]))

# score = reg.score(X,y)
# print(score)
# print(reg.coef_)
# print(reg.intercept_)

# m1 = reg.coef_[0]
# m2 = reg.coef_[1]
# m3 = reg.coef_[2]
# b = reg.intercept_
# print("formula: y = {:.2f}*RM + {:.2f}*LSTAT + {:.2f}*'PTRATIO + {:.2f}".format(m1,m2,m3,b) )

# predictions = reg.predict(X)
# print(predictions)

# print(mean_absolute_error(y, predictions))
# print(mean_squared_error(y,predictions))
# print(mean_squared_error(y,predictions, squared=False)) #Here is Root mean squared error

# scaler = StandardScaler()
# scaler.fit(X)
# print(scaler.transform(X))

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
X = df.loc[:, ['sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)']]
print(X.shape)
y = df.loc[:, 'target'].values
print(y.shape)

scaler = StandardScaler()
print(X)
scaler.fit(X)
X = scaler.transform(X)
print(X)

knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X,y)
predictions = knn.predict(X)
print(predictions)

score = knn.score(X, y)
print(score)