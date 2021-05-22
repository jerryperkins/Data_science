import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


bh = pd.read_csv('data/modifiedBostonHousing.csv')
print(bh.head(), bh.shape)

bh = bh.loc[:, ['RM', 'LSTAT', 'PTRATIO', 'price']]
bh = bh.dropna(how = 'any')

price_filer = bh.loc[:, 'price'] < 0
bh = bh.loc[~price_filer, :]

X = bh.loc[:, ['RM', 'LSTAT', 'PTRATIO']].values
print(X.shape)
y = bh.loc[:, 'price'].values
print(y.shape)

reg = LinearRegression(fit_intercept=True)
reg.fit(X,y)
print(reg.predict(X[0].reshape(-1,3)))
print(reg.predict(X[0:10]))


predictions = reg.predict(X)
print(predictions)

print(r2_score(y, predictions))
print(mean_absolute_error(y, predictions))
print(mean_squared_error(y,predictions))
print(mean_squared_error(y,predictions, squared=False)) #Here is Root mean squared error  https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python