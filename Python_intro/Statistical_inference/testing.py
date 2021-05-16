from os import replace
import matplotlib.pyplot as plt
from numpy.random import sample
import seaborn as sns
import pandas as pd
import numpy as np
# First time importing the machine learning library scikit-learn 
# Don't worry if this seems like a lot of code
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

df = pd.read_csv('data/titanic.csv')
print(df.head())


# # Transform Sex column into a non text form.
# # I choose four features, you could have chosen others
# feature_cols = ['Pclass', 'Parch', 'Age', 'Sex']
# # Make Sex column into something you can feed into a model
# # Has 
# df['Sex'] = df.Sex.map({'male': 0, 'female': 1})
# # Remove rows where age is nan from the dataset
# df = df.loc[~df['Age'].isnull(), :]
# # Create features matrix and target vector
# X = df.loc[:, feature_cols]
# y = df['Survived']
# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
# # Standardize data
# scaler = StandardScaler()
# # Fit on training set only.
# scaler.fit(X_train)
# # Apply transform to both the training set and the test set.
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# # Make an instance of a model
# logreg = LogisticRegression()
# # Train the model on the data
# logreg.fit(X_train, y_train);
# # class predictions (not predicted probabilities)
# predictions = logreg.predict(X_test)
# print(predictions)

# # score = logreg.score(X_test, y_test)
# # print(score)
# print(y_test.values)

# cm = metrics.confusion_matrix(y_test.values, predictions)
# print(cm)
# print(cm.ravel())
# print(cm.ravel().shape)
# tn, fp, fn, tp = cm.ravel()
# print(tn, fp, fn, tp)

# sensitivity = tp / (tp + fn)
# specificity  = tn / (tn + fp)
# print('Sensitivity: {:.3f}'.format(sensitivity))
# print('Specificity: {:.3f}'.format(specificity))

# precision = tp / (tp + fp)

# print('Precision: {:.3f}'.format(precision))

# type_one_error = 1 - specificity
# type_two_error = 1 - sensitivity
# print('Type 1 Error Rate: {:.3f}'.format(type_one_error))
# print('Type 2 Error Rate: {:.3f}'.format(type_two_error))

houses = pd.read_csv('data/kingCountyHouseData.csv')

print(houses.head())


# print(houses['price'])
# houses['price'].hist(bins=30)
# plt.xticks(rotation = 45)
# plt.ticklabel_format(useOffset=False, style='plain')
# plt.tight_layout()

# plt.show()

# plt.style.use('seaborn')
# price_filter = houses.loc[:, 'price'] <= 3000000
# houses.loc[price_filter, 'price'].hist(bins=30, edgecolor='black')
# plt.xticks(rotation=45)
# plt.ticklabel_format(useOffset=False ,style='plain')
# plt.tight_layout()

# plt.show()


bc = pd.read_csv('data/1620745362__wisconsinBreastCancer (4).csv')

print(bc.head())

# malignant = bc.loc[bc['diagnosis']=='M', 'area_mean'].values
# print(malignant)
# benign = bc.loc[bc['diagnosis']=='B', 'area_mean'].values
# print(benign)

# plt.boxplot([malignant, benign], labels=['M', 'B'])
# plt.show()

# bc.boxplot(column='area_mean', by='diagnosis')
# plt.title('')
# plt.suptitle('')
# plt.show()

# sns.boxplot(x='diagnosis', y='area_mean', data=bc)
# plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=1)
# axes.boxplot([malignant, benign], notch = True, labels=['M', 'B'])
# plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=1)
# boxplots = axes.boxplot([malignant,benign],
#             notch = True,
#             labels=['M', 'B'],
#             widths = .7,
#             patch_artist=True,
#             medianprops = dict(linestyle='-', linewidth=2, color='Yellow'),
#             boxprops = dict(linestyle='--', linewidth=2, color='Black', facecolor = 'blue', alpha = .4)
#             );
# boxplot1 = boxplots['boxes'][0]
# boxplot1.set_facecolor('red')
# plt.xlabel('diagnosis', fontsize = 20);
# plt.ylabel('area_mean', fontsize = 20);
# plt.xticks(fontsize = 16);
# plt.yticks(fontsize = 16);

# plt.show()

data = load_boston()
boston = pd.DataFrame(data.data, columns=data.feature_names)
boston['target'] = data.target
print(boston.head())
print(boston.info())

#sampling with replacement
# print(np.random.choice(a=15, size=15, replace=True))

for _ in range(1,15):
    samples = np.random.choice(a=15, size=15, replace=True)
    print(samples, ',Num Samples: ' + str(len(samples)) + ',numberunique: ' + str(len(set(samples))))


bootstrappedDF = boston.iloc[samples, :]
print(bootstrappedDF)

#sampling without replacement

print(np.random.choice(a=15, size=15, replace=False))

for _ in range(1,15):
    samples = np.random.choice(a=15, size=15, replace=False)
    print(samples, ',Num Samples: ' + str(len(samples)) + ',numberunique: ' + str(len(set(samples))))

withoutDF = boston.iloc[samples, :]
print(withoutDF)

olympic = pd.read_csv('data/athleteEventsNoPersonal.csv')

print(olympic.head())
print(olympic['Height'].value_counts())

heightCountryYear = pd.DataFrame(olympic.groupby(['NOC', 'Year', 'ID'])['Height'].mean())
print(heightCountryYear)
temp = heightCountryYear.reset_index().groupby(['NOC', 'Year'])['Height'].apply(list)
print(temp)
temp = temp.reset_index()
print(temp)
nanFilter = temp.loc[:, 'Height'].apply(lambda x: np.logical_not(np.isnan(x).any()))
print(nanFilter)
usFilter = temp['NOC']=='USA'
print(usFilter)
print(usFilter & nanFilter)
yearHeight = temp.loc[nanFilter & usFilter, :]
print(yearHeight)
print(yearHeight['Year'].values)
print(yearHeight['Height'].values[0])

plt.style.use('seaborn')
indexList = []
for index, heights in enumerate(yearHeight['Height'].values):
    plt.boxplot(yearHeight['Height'].values[index],
                positions=[index],
                widths=.6,
                medianprops = dict(linestyle = '-', linewidth=2, color='green'),
                showmeans= True,
                meanprops = dict(marker='X', markeredgecolor='black',
                markerfacecolor = 'firebrick'))
    indexList.append(index)
plt.xticks(indexList,yearHeight['Year'].values, fontsize = 16, rotation = 45)
plt.yticks(fontsize = 16)
plt.title('US Olympic Heights Over Time', fontsize = 18)
plt.show()

