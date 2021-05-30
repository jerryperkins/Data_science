import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Import bagging tree and random forest classifiers
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/kc_house_data.csv')
print(df.head())
print(df.info())
df = df.drop(columns=['id', 'date', 'zipcode', 'lat', 'long']) # don't need id, date would not fit into these algos, zipcode is categorical with 70 unique vals so I did not want to make that many dummies. Lat and long seem out of place
print(df.head())

X = df.loc[:, df.columns != 'price']
print(X.shape)

y = df.loc[:,'price'].values
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=3)



clf = RandomForestRegressor(n_estimators=100, bootstrap=True, oob_score=True)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test[0:10])
print(predictions)
score = clf.score(X_test, y_test)
print(score)

features = clf.feature_importances_
for i in range(0, len(X.columns),1):
    print(features[i], " " ,X.columns[i])

features_df = pd.DataFrame({'feature':X.columns, 'importance': clf.feature_importances_})
features_df.sort_values(by='importance', inplace = True)
print(features_df) # saw this way of showing feature importance in your solution and tried it out here too but my way was seen above. Not quite as clean;)


clf = BaggingRegressor(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test[0:10])
print(predictions)
score = clf.score(X_test, y_test)
print(score)


#1. What are the most important features for your model?  
#Grade and sqft_living were the most important factors by far.

# 2.What other parameters could you have tried tuning? 
# BaggingRegressor:  n_estimators, max_samples, max_features, bootstrap, bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose

# RandomForestRegressor:n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, ccp_alpha, max_samples