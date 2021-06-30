import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


df = pd.read_csv('data/iris.csv', names = ['sepal length','sepal width','petal length','petal width','target'])
print(df.head())

X = df.drop(columns = 'target')
le = LabelEncoder()
y = le.fit_transform(df['target'])
print(y)

scaler = StandardScaler()
# Fit & transform data.
scaled_df = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(scaled_df) 

# print(pcs)

# plt.figure(figsize = (8, 4))
# plt.scatter(pcs[:,0], pcs[:,1], c = y)
# plt.title('Visualization of all of our data using the first two Principal Components')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()

df = pd.read_csv('data/wisconsinBreastCancer.csv', index_col='id')
print(df.head())
print(df.info())

# Drop unnecessary column
df.drop(columns = 'Unnamed: 32', inplace = True)
# Binarize target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# Save X data
X = df.drop(columns = 'diagnosis')
# Encode our target
y = df['diagnosis']

print(X.shape)

scaler = StandardScaler()
# Fit & transform data.
scaled_df = scaler.fit_transform(X)

pca = PCA()
pca.fit(scaled_df)

# plt.plot(range(1, 11), pca.explained_variance_ratio_[:10], marker = '.')
# plt.xticks(ticks = range(1, 11))
# plt.xlabel('Principal Component')
# plt.ylabel('Proportion of Explained Variance')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
scaler = StandardScaler()
# Fit & transform data.
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

pca = PCA(n_components = 3)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)
# fit logistic regression
logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)

print('Training accuracy:', logreg.score(X_train_pca, y_train))
print('Testing accuracy:', logreg.score(X_test_pca, y_test))

#scikti learn pipelines

print(df.head())
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

pipe = make_pipeline(StandardScaler(), PCA(n_components = 3), LogisticRegression())
pipe.fit(X_train, y_train)

print('Training accuracy:', pipe.score(X_train, y_train))
print('Testing accuracy:', pipe.score(X_test, y_test))