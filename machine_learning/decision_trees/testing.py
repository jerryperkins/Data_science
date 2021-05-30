import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# For scaling data
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
# For model validation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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


print(df.head())

X = df.loc[:, ['sepal length (cm)', 
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)']]
print(X.shape)

y = df.loc[:, 'target'].values
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

clf = DecisionTreeClassifier(max_depth=2, random_state=0,)
print(clf)
clf.fit(X_train, y_train)
print(clf.predict(X_test.iloc[0].values.reshape(1,-1))) # one prediction
print(clf.predict(X_test[0:10])) # multiple predictions

score = clf.score(X_test, y_test)
print(score)
print(clf.get_n_leaves())
print(clf.get_depth())

max_depth_range = list(range(1,6))
accuracy = []
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)
print(accuracy)
plt.plot(max_depth_range, accuracy)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show()