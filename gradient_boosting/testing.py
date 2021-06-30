import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


df = pd.read_csv('data/wisconsinBreastCancer (2).csv')
print(df.head())

# Drop unnecessary column
df.drop(columns = 'Unnamed: 32', inplace = True)
# Binarize target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# Save X data
X = df.drop(columns = 'diagnosis')
# Encode our target
y = df['diagnosis']

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

print('Training accuracy:', gbc.score(X_train, y_train))
print('Testing accuracy:', gbc.score(X_test, y_test))

#LightGBM

df = pd.read_csv('data/wisconsinBreastCancer (2).csv')
print(df.head())

# Drop unnecessary column
df.drop(columns = 'Unnamed: 32', inplace = True)
# Binarize target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# Save X data
X = df.drop(columns = 'diagnosis')
# Encode our target
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)

print('Training accuracy:', lgbm.score(X_train, y_train))
print('Testing accuracy:', lgbm.score(X_test, y_test))

#XGBoost

df = pd.read_csv('data/wisconsinBreastCancer (2).csv')
print(df.head())

# Drop unnecessary column
df.drop(columns = 'Unnamed: 32', inplace = True)
# Binarize target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# Save X data
X = df.drop(columns = 'diagnosis')
# Encode our target
y = df['diagnosis']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print('Training accuracy:', xgb.score(X_train, y_train))
print('Testing accuracy:', xgb.score(X_test, y_test))