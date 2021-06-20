import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

wine = pd.read_csv('data/wine2.csv')
print(wine.head())
df = wine[['malic_acid', 'flavanoids']]
print(df.head())

scaler = StandardScaler() #no train test split needed for kmeans

scaled_df = scaler.fit_transform(df)

# plt.scatter(df['malic_acid'], df['flavanoids'])
# plt.xlabel('Malic Acid')
# plt.ylabel('Flavanoids')
# plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(scaled_df)

df['clusters'] = kmeans.labels_
print(df['clusters'])

# plt.scatter(df['malic_acid'], df['flavanoids'], c = df['clusters'])
# plt.xlabel('Malic Acid')
# plt.ylabel('Flavanoids')
# plt.title('Clusters of Wine Varieties')
# plt.show()

df = wine[['malic_acid', 'flavanoids']]
print(df.head())

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# plt.figure(figsize=(15,5))
# sch.dendrogram(sch.linkage(scaled_df, method = 'ward'))
# plt.xlabel('Data Points')
# plt.show()

# for i in range(1,7,1): # you should use 2 or 3 clusters in this example but I wanted to loop through and see a variety of options
#     hc = AgglomerativeClustering(n_clusters=i)
#     hc.fit(scaled_df)

#     df['clusters'] = hc.labels_
#     plt.scatter(df['malic_acid'], df['flavanoids'], c = df['clusters'])
#     plt.xlabel('Malic Acid')
#     plt.ylabel('Flavanoids')
#     plt.title('Clusters of Wine Varieties')
#     plt.show()



# dbs = DBSCAN(eps=.5, min_samples=5).fit(scaled_df)
# df['clusters'] = dbs.labels_
# plt.scatter(df['malic_acid'], df['flavanoids'], c=df['clusters'])
# plt.xlabel('Malic Acid')
# plt.ylabel('Flavanoids')
# plt.title('Clusters of Wine Varieties')
# plt.show()

silhouette_scores = []
for i in range(2,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_df)
    silhouette_scores.append(silhouette_score(scaled_df, kmeans.labels_))
plt.plot(range(2,11), silhouette_scores, marker='.')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouetter Score')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(scaled_df)
print(silhouette_score(scaled_df, kmeans.labels_))

hc = AgglomerativeClustering(n_clusters = 2)
hc.fit(scaled_df)
print(silhouette_score(scaled_df, hc.labels_))

dbs = DBSCAN(eps = 0.5, min_samples = 5).fit(scaled_df)
print(silhouette_score(scaled_df, dbs.labels_))