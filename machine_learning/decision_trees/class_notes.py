import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# What is the bias/variance trade-off?
# The bias/variance trade-off is the trade-off we make in machine learning between having too high bias and too high variance. You ultimately want to strike a good balance between the two.



# High Bias
# Bias is essentially just how good of a job your model does at fitting to your data. (High bias = bad model).

# High bias = underfit

# High Variance
# Variance is essentially how well your model does at generalizing to new data. High variance means your model is too overfit to the data it was trained on.

# High variance = overfit


mpg = pd.read_csv('data/auto-mpg.csv')
print(mpg.head())

print(mpg['origin'].value_counts())
print(mpg['model year'].nunique())
mpg = pd.get_dummies(mpg, columns = ['origin', 'model year'], drop_first = True)
print(mpg['car name'].nunique())
mpg.drop(columns = 'car name', inplace = True)

X = mpg.drop(columns='mpg')
y = mpg['mpg']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(y_train.mean())

preds = [y_train.mean()] * len(y_test)
print(preds)
print(np.sqrt(mean_squared_error(y_test, preds))) # this is the amount of mpg we will be off by. This serves as a baseline to measure our actualy model against

knn = KNeighborsRegressor(n_neighbors=len(X_train))
knn.fit(X_train, y_train)
train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

print(np.sqrt(mean_squared_error(y_train, train_preds)))
print(np.sqrt(mean_squared_error(y_test, test_preds)))

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)
train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

print('Training RMSE:', np.sqrt(mean_squared_error(y_train, train_preds)))
print('Testing RMSE:', np.sqrt(mean_squared_error(y_test, test_preds)))
