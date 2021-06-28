import pandas as pd
import numpy as np
# import RF 
from sklearn.ensemble import RandomForestClassifier
# import gridsearch
from sklearn.model_selection import GridSearchCV
# import data
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Data
iris = load_iris()
# features
X = pd.DataFrame(iris.data, columns = iris.feature_names)
# target
y = iris.target
# View data
print(X.head())


params = {'C': [1, .8, .6, .4, .2],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],}

gs = GridSearchCV(LogisticRegression(), param_grid = params)
print(gs.fit(X, y))
print(gs.best_params_) 
gs.cv_results_ = pd.DataFrame(gs.cv_results_)
print(gs.cv_results_)
print(gs.score(X,y))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, stratify=y, test_size=.3)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(penalty='l1', C=1, solver='liblinear', multi_class='ovr')
log_reg.fit(X_train, y_train)
print("current c value")
print('training accuracy:', log_reg.score(X_train, y_train))
print('testing accuracy', log_reg.score(X_test, y_test))

