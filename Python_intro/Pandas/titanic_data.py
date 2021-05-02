import pandas as pd

titanic = pd.read_csv('pandas_data/titanic.csv')

#1 Display the head of the dataframe.
print(titanic.head())

# 2 What is the percentage of people who survived?
survival_percantage = (titanic['Survived'].value_counts(normalize=True)[1]) * 100
print(survival_percantage)

#3 How many of each sex survived?
survivors_by_sex = titanic.groupby(['Sex'])[['Survived']].sum()
print(survivors_by_sex)
# male = titanic['Sex']=='male'
# female = titanic['Sex']=='female'
# male_survivors = titanic.loc[survivors & male]['PassengerId'].count()
# female_survivors = titanic.loc[survivors & female]['PassengerId'].count()
# print(male_survivors)
# print(female_survivors)

#4 What is the percentage of people that survived who paid a fare less than 10?

paid_less_than_10 = titanic[titanic['Fare'] < 10]
percentage_of_poor_surviors = paid_less_than_10['Survived'].value_counts(normalize=True)[1] * 100
print(percentage_of_poor_surviors)

#5. What is the average age of those who didn't survive?
perished = titanic[titanic['Survived']==0]
print(perished)
perished_avg_age = perished.groupby(['Survived'])[['Age']].mean()
print(perished_avg_age)

#6 What is the average age of those who did survive?
lived = titanic[titanic['Survived']==1]
print(lived)
lived_avg_age = lived.groupby(['Survived'])[['Age']].mean()
print(lived_avg_age)

#7 What is the average age of those who did and didn't survive grouped by Sex?

answer = titanic.groupby(['Sex', 'Survived'])[['Age']].mean()
print(answer)




