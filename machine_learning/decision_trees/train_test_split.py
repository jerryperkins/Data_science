import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/abalone.csv', header=None)
print(df.head())

df = df.rename(columns={0: 'Sex', 1: 'Length', 2: "Diameter", 3: 'Height', 4: 'Whole weight', 5: 'Shucked weight', 6: 'Viscera weight', 7: 'Shell weight', 8: 'Rings'})
print(df.head())
print(df.info())

def to_age(y):
    return y + 1.5
df['Rings'] = df['Rings'].apply(to_age)

df_dummies = pd.get_dummies(df, columns=['Sex']) #need to get the sex column from words to numbers
print(df_dummies)
df_dummies = df_dummies.drop(columns='Rings')
print(df_dummies)

X = df_dummies
print(X.shape)
y = df['Rings'].values
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# Using linear regression with train_test_split

reg = LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)

predictions = reg.predict(X_train)
print(predictions)
score = reg.score(X_train, y_train)
print(score) #0.5455866100349975

test_predictions = reg.predict(X_test)
print(test_predictions)
score = reg.score(X_test, y_test)
print(score) #0.5070799318499823

# LinearRegression without train_test_split

reg = LinearRegression(fit_intercept=True)

reg.fit(X,y)
predictions = reg.predict(X)
print(predictions)
score = reg.score(X,y)
print(score) #0.5374264529390634

# Using KNeighborsRegressor with train_test_split

scaler = StandardScaler()   
scaler.fit(X_train)
print(X_train)
X_train = scaler.transform(X_train)
print(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_train)
print(predictions)
score = knn.score(X_train, y_train)
print(score) # when k = 5 score =  0.6593613759478307, when k = 1 score = 1

test_predictions = knn.predict(X_test)
print(test_predictions)
score = knn.score(X_test, y_test)
print(score) # when k = 5 score =  0.4725628672618698, when k = 1 score = 0.13038847871797632

# Using KNeighborsRegressor without train_test_split

scaler.fit(X)
X = scaler.transform(X)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X,y)
predictions = knn.predict(X)
print(predictions)
score = knn.score(X,y)
print(score) # when k = 5 score = 0.6593507213110799, when k = 1 score = 1

#Questions to answer

# 1. Which of KNN or linear regression seemed like a better model when you didn't use train test split? 

# Knn seemed to perform a little better but it was dependant on the k value

# 2. Which of KNN or linear regression seemed like a better model when you used train test split? 

#I thought linear regression seemed to be a better model when using to train test split as it had higher scores 0.5070799318499823(linreg) vs 0.4725628672618698(knn) 

# 3.Was there an advantage to linear regression in terms the amount of code you had to write? 

# There was a less code written for linear regression

# 4.Is there any way you could show someone which of the two models was more effective? 

# By showing them the scores

# 5.Is there any way you think you could have improved KNN to be more effective of a model?

# Finding a better balance b/t bias and variance perhaps?