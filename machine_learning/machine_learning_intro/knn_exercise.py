import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# task 1: predict the age of abalone from physical measurements using KNN.
df = pd.read_csv('data/abalone.csv', header=None)
print(df.head())

df = df.rename(columns={0: 'Sex', 1: 'Length', 2: "Diameter", 3: 'Height', 4: 'Whole weight', 5: 'Shucked weight', 6: 'Viscera weight', 7: 'Shell weight', 8: 'Rings'})
print(df.head())
print(df.info())
def to_age(y):
    return y + 1.5
df['Rings'] = df['Rings'].apply(to_age) # we are looking for age and age = ring + 1.5

df_dummies = pd.get_dummies(df, columns=['Sex']) #need to get the sex column from words to numberse
print(df_dummies)
df.loc[:,:].hist(bins=25, figsize=[16,16], xlabelsize='10', ylabelsize='10', xrot=-15) #based off the packet of info that came with this data I was pretty sure that the data was clean but thought it would be good practice to visualize it anyways to confirm. Data looked good from this hist
# plt.show()
y = df_dummies.loc[:, 'Rings'].values
df_dummies =  df_dummies.drop("Rings", axis=1)
print(df_dummies)

X = df_dummies.loc[:,:].values
print(X, X.shape)

print(y, y.shape)

#Could you have used linear regression and KNN regression to solve the regression problem: I am going to go ahead and answer question 1 here. The answer is Yes and I acutally did linear regression first because I did not read the original question carefully and thought linear regression was what was being asked for. You can uncomment the section of code below to see the results.

# reg = LinearRegression(fit_intercept=True)
# reg.fit(X,y)
# score = reg.score(X,y)
# print(score)
# predictions = reg.predict(X)
# print(predictions)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

knn_reg = KNeighborsRegressor(n_neighbors=5) #if 1 is used than prediction is equal to actual. 
knn_reg.fit(X,y)
predictions = knn_reg.predict(X)
print(predictions) # here is prediction using KNN
print(knn_reg.score(X,y)) # When I checked my solutions against yours I saw you had this as well. Was this part of the ask? Is this just a good thing to include?



# task2: predict sex from its features

df = pd.read_csv('data/abalone.csv', header=None)
print(df.head()) # I just went ahead and reloaded the data so I did not have to worry about how I manipulated it earlier in the file

df = df.rename(columns={0: 'Sex', 1: 'Length', 2: "Diameter", 3: 'Height', 4: 'Whole weight', 5: 'Shucked weight', 6: 'Viscera weight', 7: 'Shell weight', 8: 'Rings'})
print(df.head())
print(df.info())

y = df.loc[:, 'Sex']
print(y)
df = df.drop("Sex", axis=1)
X = df.loc[:,:].values
print(X, X.shape)
y =  pd.get_dummies(y, columns=['Sex']) # need to get the sex column from words to numbers
print(y)
# we need to create separate arrays of values for each type of sex so we can predict them using knnclassifier
# Talked to Nichole and it turns out for KNeighborsClassifier the target can be categorical and does not require being changed via get_dummies. I should have just set y = df.loc[:, 'Sex'].values 
F = y.loc[:, 'F'].values
print(F)
M = y.loc[:, 'M'].values
print(M)
I = y.loc[:, 'I'].values
print(I)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

k = 5 # this way I don't have to change the n_neighbors value for all of them.

knn_F = KNeighborsClassifier(n_neighbors=k)
knn_M = KNeighborsClassifier(n_neighbors=k)
knn_I = KNeighborsClassifier(n_neighbors=k)
knn_F.fit(X,F)
knn_M.fit(X,M)
knn_I.fit(X,I)
predictions_F = knn_F.predict(X) 
print(predictions_F) # Female predictions
predictions_M = knn_M.predict(X)
print(predictions_M)# Male predictions
predictions_I = knn_I.predict(X)
print(predictions_I)# Infant predictions

print(knn_F.score(X,F)) # same as noted earlier. I added these when looking at your solutions
print(knn_M.score(X,M))
print(knn_I.score(X,I))


print(predictions_I[0])
count = 0
for i in range(0, len(predictions_F), 1):
    if predictions_F[i] == 0 & predictions_M[i] == 0 & predictions_I[i] == 0:
        count += 1
        # print("We failed")
print(count)

# Question 2: Could you have used linear regression for the classification problem. Linear Regression is not very good at classification so KNN is the better choice here

