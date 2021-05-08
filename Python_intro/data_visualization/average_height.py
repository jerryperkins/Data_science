import pandas as pd

olympians = pd.read_csv('data/athleteEventsNoPersonal.csv')

#Calculate the average height per individual olympian per year

# print(olympians.head())
# print(olympians.info()) 
# print(olympians['ID'].value_counts())
print(olympians.groupby(['ID', 'Year'])[['Height']].mean())