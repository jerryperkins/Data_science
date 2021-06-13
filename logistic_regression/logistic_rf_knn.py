import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

col_names = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins','Color intensity', 'Hue', 'OD280/OD315 of diluted wines','Proline']
wine = pd.read_csv('data/wine.csv', header=None, names=col_names)

print(wine.head())
print('Class labels', np.unique(wine['Class label']))
print(wine['Class label'].value_counts(dropna=False))

X = wine.loc[:, wine.columns[(wine.columns != 'Class label')]]
y = wine.loc[:, 'Class label'].values
print(X.shape, y.shape)

# logisticRegregression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', multi_class='ovr')
log_reg.fit(X_train, y_train)
print('training accuracy:', log_reg.score(X_train, y_train)) # l1: 1, l2: 1
print('testing accuracy', log_reg.score(X_test, y_test)) # l1: 1, l2: 0.9814814814814815

print(log_reg.intercept_)
print(log_reg.coef_)

print(log_reg.predict_proba(X_test[0:1])) # l1: [[0.00313761 0.91217649 0.0846859 ]], l2: [[0.00217222 0.92294096 0.07488681]]
print(log_reg.predict(X_test[0:1])) #l1: [2], l2 [2]


# One vs Rest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ovr = OneVsRestClassifier(LinearSVC(random_state=3, multi_class='ovr'))
ovr.fit(X_train, y_train)
print('ovr training accuracy:', ovr.score(X_train, y_train)) # 1.0
print('ovr testing accuracy', ovr.score(X_test, y_test)) #0.9629629629629629

#One vs One

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ovo = OneVsOneClassifier(LinearSVC(random_state=3, multi_class='ovr'))
ovo.fit(X_train, y_train)
print('ovo training accuracy:', ovo.score(X_train, y_train)) # 1.0
print('ovo testing accuracy', ovo.score(X_test, y_test)) #0.9629629629629629


# Random Forest

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3, stratify=y)

rf_class = RandomForestClassifier(n_estimators=50, bootstrap=True, oob_score=True)
rf_class.fit(X_train, y_train)
score = rf_class.score(X_train, y_train)
print("X_train rf_class", score) #1.0
score = rf_class.score(X_test, y_test)
print("X_test rf_class", score) #0.9777777777777777
predictions = rf_class.predict(X_test) 
print("rf_class X_test predictions", predictions)

features = rf_class.feature_importances_

features_df = pd.DataFrame({'feature':X.columns, 'importance': rf_class.feature_importances_})
features_df.sort_values(by='importance', inplace = True)
print("rf_class feature importances")
print(features_df)

# bagged Trees

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3, stratify=y)

bg_class = BaggingClassifier(n_estimators=50,)
bg_class.fit(X_train, y_train)
score = bg_class.score(X_train, y_train)
print("X_train bg_class", score)
predictions = bg_class.predict(X_test)
score = bg_class.score(X_test, y_test)
print("X_test bg_class", score) #0.9555555555555556
print("bg_class X_test predictions", predictions)


# KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn_class = KNeighborsClassifier(n_neighbors=5)
knn_class.fit(X_train, y_train)
score = knn_class.score(X_train, y_train)
print("X_train knn_class", score) #0.9699248120300752
score = knn_class.score(X_test, y_test)
print("X_test knn_class", score) #0.9777777777777777
predictions = knn_class.predict(X_test) 
print("knn_class X_test predictions", predictions)




# Questions:
#  1. Which model performed the best?:
# The best performing model seemed to be logistic regression with a score of .98

# 1. What are the most important features for your models? Is there a model that you liked the best and why?
# The most important features are proline and Color intensity. I like random forest the best because of the built in feature importance
