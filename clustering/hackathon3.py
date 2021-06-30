from os import scandir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

df=pd.read_csv('data/Melbourne_housing_small.csv')

print(df.head())

print(df['land'].value_counts().sort_index())

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=5)
kmeans.fit(scaled_df)

df['clusters'] = kmeans.labels_

print(df['clusters'].value_counts())

# # this creates a scatter plot for every combination of columns
# for i in range(0,len(df.columns),1):
#     for j in range(0,len(df.columns), 1):
#         if j == i:
#             continue
#         print(df.columns[i])
#         # print(df[df.columns[i]])
#         plt.scatter(df[df.columns[i]], df[df.columns[j]], c=df['clusters'])
#         plt.xlabel(df.columns[i])
#         plt.ylabel(df.columns[j])
#         plt.show()

# silhouette_scores = []
# for i in range(2,11):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(scaled_df)
#     silhouette_scores.append(silhouette_score(scaled_df, kmeans.labels_))
# plt.plot(range(2,11), silhouette_scores, marker='.')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouetter Score')
# plt.show() # shows 2 clusters to have the highest score


# AgglomerativeClustering

# plt.figure(figsize=(15,5))
# sch.dendrogram(sch.linkage(scaled_df, method = 'ward'))
# plt.xlabel('Data Points')
# plt.show() #shows that 3 clusters makes the most sense

# silhouette_scores = []
# for i in range(2,11):
#     hc = AgglomerativeClustering(n_clusters=i)
#     hc.fit(scaled_df)
#     silhouette_scores.append(silhouette_score(scaled_df, hc.labels_))
# plt.plot(range(2,11), silhouette_scores, marker='.')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouetter Score')
# plt.show() # shows that two clusters makes the most sense as that has the highest score but it does not line up with the dendrogram

hc = AgglomerativeClustering(n_clusters = 3)
hc.fit(scaled_df)
print(silhouette_score(scaled_df, hc.labels_)) #0.3357297821671033 slightly higher than kmeans but only by .001
df['clusters'] = hc.labels_

for i in range(0,len(df.columns),1): # shows scatter plots of each column paired with each other column.
    for j in range(1,len(df.columns), 1):
        if j == i:
            continue        
        plt.scatter(df[df.columns[i]], df[df.columns[j]], c=df['clusters'])
        plt.xlabel(df.columns[i])
        plt.ylabel(df.columns[j])
        plt.tight_layout()
        plt.show()
