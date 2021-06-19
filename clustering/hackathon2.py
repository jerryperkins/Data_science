import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('data/nba_rookies.csv')

print(df.head())
print(df.info())
print(df['TARGET_5Yrs'].value_counts())

df = df.replace({'Yes': 1, 'No': 0})
print(df.head())

df = df.drop(columns='Name')
print(df.head())
y = df['TARGET_5Yrs'].values
print(y.shape)
X = df.drop(columns='TARGET_5Yrs')
print(X)
X = X.loc[:,:]
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', multi_class='ovr')
log_reg.fit(X_train, y_train)
print('training accuracy: log_reg', log_reg.score(X_train, y_train))
print('testing accuracy log_reg', log_reg.score(X_test, y_test))

# random Forest

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3, stratify=y)

# # for i in range(3,150, 5):
# rf_class = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True, max_depth=6, min_samples_split=6)
# rf_class.fit(X_train, y_train)
# score = rf_class.score(X_train, y_train)
# print("X_train rf_class", score)
# score = rf_class.score(X_test, y_test)
# print("X_test rf_class",  score) #0.7027027027027027


# rfc = RandomForestClassifier(random_state=42)

# params={'n_estimators': [75,100,125,150],
#             'min_samples_split': [3,4,5,6,8,10],
#             'max_depth': [4,5,6,7,8]}

scores = []
# Use train test split with each value of n_estimators (Warning: Slow!).
estimator_range = []
for estimator in range(10,310,10):
    clf = RandomForestClassifier(n_estimators=estimator,
                                random_state=1,
                                bootstrap=True)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
    print("here is the score", estimator, scores)
    estimator_range.append(estimator)


fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5))
axes.plot(estimator_range, scores)
axes.set_xlabel('n_estimators', fontsize =20)
axes.set_ylabel('Accuracy', fontsize = 20)
axes.grid()
plt.show()


# # cv_rs = GridSearchCV(estimator=rfc,param_distributions=param_grid,cv=3,n_iter = 10,random_state=42)
# # cv_rs.fit(X_train,y_train)
# # print('Random Search : ',cv_rs.best_params_)

# gs = GridSearchCV(RandomForestClassifier(), param_grid = params)
# gs.fit(X_train, y_train)
# print("grid search", gs.best_params_)
# print(gs.score)



# KNeighborsClassifier

# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3, stratify=y)

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# knn_class = KNeighborsClassifier(n_neighbors=5)
# knn_class.fit(X_train, y_train)
# score = knn_class.score(X_train, y_train)
# print("X_train knn_class", score) 
# score = knn_class.score(X_test, y_test)
# print("X_test knn_class", score) 
