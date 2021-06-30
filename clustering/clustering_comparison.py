import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

cust = pd.read_csv('data/cust_seg.csv')

print(cust.head())

cust.drop(columns=['Unnamed: 0', 'Customer Id'], inplace=True)
print(cust.head())
print(cust.info())
cust.dropna(inplace=True)
print(cust.info())



scaler = StandardScaler()
scaled_df = scaler.fit_transform(cust)

#Kmeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_df)


cust['clusters'] = kmeans.labels_

# uses seaborn to visualise customer segments
# sns.pairplot(cust, hue='clusters')
# plt.tight_layout()
# plt.show()

# # this creates a scatter plot for every combination of columns
# for i in range(1,len(cust.columns)-1,1):
#     for j in range(1,len(cust.columns)-1, 1):
#         if j == i:
#             continue
#         print(cust.columns[i])
#         # print(cust[cust.columns[i]])
#         plt.scatter(cust[cust.columns[i]], cust[cust.columns[j]], c=cust['clusters'])
#         plt.xlabel(cust.columns[i])
#         plt.ylabel(cust.columns[j])
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

# kmeans = KMeans(n_clusters=2)
# kmeans.fit(scaled_df)
# print(silhouette_score(scaled_df, kmeans.labels_)) #0.33496643365707623

# # Hierachical clustering

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

# hc = AgglomerativeClustering(n_clusters = 2)
# hc.fit(scaled_df)
# print(silhouette_score(scaled_df, hc.labels_)) #0.3357297821671033 slightly higher than kmeans but only by .001


# for i in range(0,len(cust.columns),1): # shows scatter plots of each column paired with each other column.
#     for j in range(1,len(cust.columns), 1):
#         if j == i:
#             continue
#         hc = AgglomerativeClustering(n_clusters=3).fit(scaled_df)
#         cust['clusters'] = hc.labels_
#         plt.scatter(cust[cust.columns[i]], cust[cust.columns[j]], c=cust['clusters'])
#         plt.xlabel(cust.columns[i])
#         plt.ylabel(cust.columns[j])
#         plt.tight_layout()
#         plt.show()


# hc = AgglomerativeClustering(n_clusters=3).fit(scaled_df)
# cust['clusters'] = hc.labels_
# seg1 = cust[cust['clusters'] == 0]
# seg2 = cust[cust['clusters'] == 1]
# seg3 = cust[cust['clusters'] == 2]

# cust_segs = [seg1, seg2, seg3]
# for segment in cust_segs:
#     print(segment.describe())  

# age_mean = [np.mean(segment['Age']) for segment in cust_segs]
# income_mean = [np.mean(segment['Income']) for segment in cust_segs]
# debt_income_mean = [np.mean(segment['DebtIncomeRatio']) for segment in cust_segs]
# default_mean = [np.mean(segment['Defaulted']) for segment in cust_segs]

# stats_df = pd.DataFrame({'Cust Segment': ['seg1', 'seg2', 'seg3'], 'Age Mean': age_mean, 'Income Mean': income_mean, 'debt_income_mean': debt_income_mean, "Defaulted": default_mean})
# print(stats_df.head()) # we can see our three groups of customers and those that default have high income to debt ratios low income and low age. Young people who have low debt to income ratio and moderate income never default and older folks with lots of income default rarely.





# DBSCAN

eps_range = []
silhouette_scores = []
for  i in [float(j) / 10 for j in range(5, 50, 1)]:
    dbs = DBSCAN(eps = i, min_samples = 5).fit(scaled_df)
    silhouette_scores.append(silhouette_score(scaled_df, dbs.labels_))
    eps_range.append(i)
plt.plot(eps_range, silhouette_scores, marker='.')
plt.xlabel('eps')
plt.ylabel('Silhouette Score')
plt.show() # this shows an eps of 4.9 to be the best

# dbs = DBSCAN(eps = 4.9, min_samples = 6).fit(scaled_df)
# print(silhouette_score(scaled_df, dbs.labels_)) #eps = 5 and min samples = 5: 0.6915619668168279, eps = 4.9 and min samples = 4: 0.734494641708757, #epse = 4.9 and min samples = 6: 0.6915619668168279


dbs = DBSCAN(eps=1.5, min_samples=4).fit(scaled_df)
cust['clusters'] = dbs.labels_
print(cust['clusters'].value_counts())

for i in range(0,len(cust.columns),1):
    for j in range(1,len(cust.columns), 1):
        if j == i:
            continue
        # dbs = DBSCAN(eps=4.9, min_samples=4).fit(scaled_df)
        # cust['clusters'] = dbs.labels_
        print(silhouette_score(scaled_df, dbs.labels_))
        plt.scatter(cust[cust.columns[i]], cust[cust.columns[j]], c=cust['clusters'])
        plt.xlabel(cust.columns[i])
        plt.ylabel(cust.columns[j])
        plt.tight_layout()
        plt.show() # when looking at these scatter plots it is clear that despite having the best silhouette score eps of 4.9 and min_samples = 4 this does not provide useful data because all of the pointd except one are a sinlge color.


# For this assingment AgglomerativeClustering was slightly better as it gave nearly the same results as Kmeans but with a higher silhouette score.


