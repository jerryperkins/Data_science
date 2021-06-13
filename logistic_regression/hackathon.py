import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/merc.csv')

print(df.head())
print(df.info())

print(df['model'].value_counts())
print(df['fuelType'].value_counts())

# df.drop(columns=['fuelType'] == 'other')
# print(df.info())

temp = df.loc[:,'fuelType'] == 'Other'
print(temp)
df = df.loc[~temp, :]
print(df.info())

# df.drop(columns = 'price')
print(df.duplicated().any())
df.duplicated().any()
df = df.drop_duplicates()
print(df.info())

y = df['price'].values
X = df.drop(columns = 'price')


OneHotEncoder = pd.get_dummies(X, columns = ['model', 
                                            'transmission', 
                                            'fuelType'],
                                            drop_first = True)
X = OneHotEncoder.loc[:,:].values                                            
print(y.shape)
print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3)

reg = LinearRegression(fit_intercept=True)
reg.fit(X_train,y_train)
score = reg.score(X_train, y_train)
print(score)
score = reg.score(X_test,y_test)
predictions = reg.predict(X_test)
print(score)
print(np.sqrt(mean_squared_error(y_test,predictions)))


rf_reg = RandomForestRegressor(n_estimators=50, bootstrap=True, oob_score=True)
rf_reg.fit(X_train, y_train)
score = rf_reg.score(X_train, y_train)
print("X_train rf_reg", score)
score = rf_reg.score(X_test, y_test)
print("X_test rf_reg", score)
predictions = rf_reg.predict(X_test)
print("rmsqe rf_reg",np.sqrt(mean_squared_error(y_test,predictions)))

