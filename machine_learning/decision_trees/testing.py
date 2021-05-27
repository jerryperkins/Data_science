import pandas as pd
import numpy as np
# For scaling data
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
# For model validation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

X = df.loc[:, ['sepal length (cm)', 
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)']]
print(X, X.shape)

y = df.loc[:, 'target'].values
print(y, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

scaler = StandardScaler()
scaler.fit(X_train)
print(X_train)
X_train = scaler.transform(X_train)
print(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X_train, y_train)

predictions = knn.predict(X_train)
print(predictions)
test_predictions = knn.predict(X_test)
print(test_predictions)

score = knn.score(X_test, y_test)
print(score)