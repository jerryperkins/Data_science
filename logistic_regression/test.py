import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/breast_cancer.csv')
print(df.head())

# X = df[['concave points_worst']]
# y = df['diagnosis']

# logreg = LogisticRegression(C=1)
# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=3)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# logreg.fit(X_train, y_train)

# example_df = pd.DataFrame(data={'worst_concave_points': X_test.flatten(), 'diagnosis': y_test})
# example_df['logistic_preds'] = pd.DataFrame(logreg.predict_proba(X_test)).loc[:,1].values
# example_df = example_df.sort_values(['logistic_preds'])

# plt.scatter(example_df['worst_concave_points'], example_df['diagnosis'])
# plt.plot(example_df['worst_concave_points'], example_df['logistic_preds'].values, color='red')
# plt.ylabel('malignant (1) or benign (0)', fontsize = 12)
# plt.xlabel('concave points_worst', fontsize = 12)
# plt.show()


X = df[['concave points_worst']]
y = df['diagnosis']
# Make an instance of the model
logreg = LogisticRegression(C = .001)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
# Scaling logistic regression
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Training the model on the data, storing the information learned from the data
# Model is learning the relationship between X and y 
logreg.fit(X_train,y_train)
example_df = pd.DataFrame(data = {'worst_concave_points': X_test.flatten(),
                     'diagnosis': y_test})
example_df['logistic_preds'] = pd.DataFrame(logreg.predict_proba(X_test)).loc[:, 1].values
example_df = example_df.sort_values(['logistic_preds'])
plt.scatter(example_df['worst_concave_points'], example_df['diagnosis'])
plt.plot(example_df['worst_concave_points'], example_df['logistic_preds'].values, color='red')
plt.ylabel('malignant (1) or benign (0)', fontsize = 12)
plt.xlabel('concave points_worst', fontsize = 12)
plt.show()