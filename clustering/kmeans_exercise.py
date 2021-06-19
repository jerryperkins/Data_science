import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

cust = pd.read_csv('data/cust_seg.csv')

print(cust.head())

cust.drop(columns=['Unnamed: 0', 'Customer Id'], inplace=True)
print(cust.head())
print(cust.info())
cust.dropna(inplace=True)
print(cust.info())



scaler = StandardScaler()
scaled_df = scaler.fit_transform(cust)

kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_df)


cust['clusters'] = kmeans.labels_

#uses seaborn to visualise customer segments
sns.pairplot(cust, hue='clusters')
plt.tight_layout()
plt.show()

# this creates a scatter plot for every combination of columns
for i in range(1,len(cust.columns)-1,1):
    for j in range(1,len(cust.columns)-1, 1):
        if j == i:
            continue
        print(cust.columns[i])
        # print(cust[cust.columns[i]])
        plt.scatter(cust[cust.columns[i]], cust[cust.columns[j]], c=cust['clusters'])
        plt.xlabel(cust.columns[i])
        plt.ylabel(cust.columns[j])
        plt.show()

# Explore your various customer segments. What are trends in the segments? Create at least two visualizations that explore trends in the groups.

#There is a clear group that defaults. The third cluster does not have a single person that did not default. That group tends to be under 40(not always though), have low income, and a high debt to income ratio