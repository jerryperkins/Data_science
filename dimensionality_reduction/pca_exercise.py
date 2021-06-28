import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
# view the shape of the dataset
# print(mnist.data.shape)

print(mnist.head())

# X = df.drop(columns = 'target')
# le = LabelEncoder()
# y = le.fit_transform(df['target'])
# print(y)

# scaler = StandardScaler()
# # Fit & transform data.
# scaled_df = scaler.fit_transform(X)

# pca = PCA(n_components=2)
# pcs = pca.fit_transform(scaled_df) 