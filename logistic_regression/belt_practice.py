import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

df = pd.read_csv('data/Fish.csv')

print(df.head())

print(df.info())

print(df['Species'].value_counts())




print(df.corr().sort_values(by=['Weight']))

# sns.heatmap(df.corr().sort_values(by=['Weight']),cmap=sns.diverging_palette(240,10,n=9))
# plt.show()



y = df['Weight'].values
print(y.shape)
df.drop(columns='Weight', inplace=True)
print(df.head())
OneHotEncoder = pd.get_dummies(df, columns = ['Species'], drop_first = True)

X = OneHotEncoder.loc[:,:].values
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3)

rf_reg = RandomForestRegressor(n_estimators=50, bootstrap=True, oob_score=True)
rf_reg.fit(X_train, y_train)
score = rf_reg.score(X_train, y_train)
print("X_train rf_reg", score)
score = rf_reg.score(X_test, y_test)
print("X_test rf_reg", score)
predictions = rf_reg.predict(X_test)
print("rmsqe rf_reg",np.sqrt(mean_squared_error(y_test,predictions)))

features = rf_reg.feature_importances_
print(features)
print(df.columns)

features_df = pd.DataFrame({'feature':OneHotEncoder.columns, 'importance': rf_reg.feature_importances_}) #creating data fram to look at fearture importance
features_df.sort_values(by='importance', inplace = True)
print("rf_reg feature importances")
print(features_df)

plt.barh(OneHotEncoder.columns, rf_reg.feature_importances_, height=.5)
plt.show() #visual display of feature importance

