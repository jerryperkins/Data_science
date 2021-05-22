import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder




# Categorical Variables
# When doing machine learning, we have to deal with categorical variables to convert them to numbers in some way. In this lesson, we will go over the types of categorical variables and different methods of dealing with them.

# Categorical variables are variables that represent a category. They will show up in most datasets that you get, so it is up to you to determine how to deal with them!

# Types of Categorical Variables
# There are two types of categorical variables:

# Nominal
# Ordinal

# Nominal Variables
# Does not have a distinct ordering.

# Examples:

# male & female
# red, green, & blue

# Ordinal Variables
# Have a clear ordering.

# Examples:

# low, medium, high
# One star, two stars, three stars, four stars, and five stars


# Attribute	Description
# instructor	Course instructor (categorical, 25 categories)
# course	Course (categorical, 26 categories)
# semester	Summer or regular semester (binary)
# class_size	Class size (numerical)
# score	Class rating - low, medium, or high (categorical)



evals = pd.read_csv('data/ta_evals.csv')
evals2 = pd.read_csv('data/ta_evals.csv')

print(evals.head())
print(evals.info())

print(evals['score'].value_counts())

# Ordinal

scores = {'Low': 0, 'Medium': 1, 'High': 2}
print(scores)
# print(evals['score'].map(scores))
evals['score'] = evals['score'].map(scores)
print(evals.head())

# Nominal

#pandas get dummies
evals_dummies = pd.get_dummies(evals, columns= ['instructor', 'course', 'semester'], drop_first=True)
print(evals_dummies.head())

#OHE method
ohe = OneHotEncoder(drop = 'first', sparse=False)
ohe.fit(evals[['instructor', 'course', 'semester']])
evals_ohe = ohe.transform(evals[['instructor', 'course', 'semester']])
print(evals_ohe)
print(pd.DataFrame(evals_ohe).head())