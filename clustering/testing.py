import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

wine = pd.read_csv('data/wine2.csv')
print(wine.head())
df = wine[['malic_acid', 'flavanoids']]
print(df.head())

scaler = StandardScaler() #no train test split needed for kmeans

scaled_df = scaler.fit_transform(df)

plt.scatter(df['malic_acid'], df['flavanoids'])
plt.xlabel('Malic Acid')
plt.ylabel('Flavanoids')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(scaled_df)

df['clusters'] = kmeans.labels_
print(df['clusters'])

plt.scatter(df['malic_acid'], df['flavanoids'], c = df['clusters'])
plt.xlabel('Malic Acid')
plt.ylabel('Flavanoids')
plt.title('Clusters of Wine Varieties')
plt.show()