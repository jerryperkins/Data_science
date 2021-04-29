import pandas as pd

filename = "RailsToTrails_National_Count_Data_week.xlsx"

bike_df = pd.read_excel(filename)

# print(bike_df.head())

print(bike_df.info())

# 1.Select just the data after July 31, 2020.
july_filer = bike_df['Week of'] > '2020-07-31'
print(july_filer)


# 2.Select just data where the 'Change 2019-2020' column is greater than 100% (greater than 1).

greater_than_one = bike_df['Change 2019-2020'] > 1
print(greater_than_one)

# 3.Put these together! Where there any weeks in the second half of the year (after July 31, 2020) where the change was greater than 100%?
combined = bike_df.loc[july_filer & greater_than_one, :]
print(combined)


