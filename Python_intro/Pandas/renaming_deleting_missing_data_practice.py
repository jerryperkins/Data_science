import pandas as pd

bike_df = pd.read_excel('RailsToTrails_National_Count_Data_week.xlsx')

print(bike_df.head())

# 1.Delete the unnecessary columns:
# Unnamed: 7
# Unnamed: 8
# Unnamed: 9


bike_df = bike_df.drop(columns=['Unnamed: 7','Unnamed: 8', 'Unnamed: 9'])

print(bike_df.head())

# 2.Rename the Unnamed: 6 column to percent_change.

bike_df = bike_df.rename(columns={'Unnamed: 6': 'percent_change'})

print(bike_df['percent_change'])

# 3. How many missing values are in the dataset? Which columns are they in?
# There are 780 missing values and they are in columns '2021 Counts' and 'Unamed: 10' through 'Unamed: 23' 
print(bike_df.info())
print(bike_df.isna().sum().sum())

# 4.Fill all missing values in the 2021 Counts column with 0.

bike_df['2021 Counts'] = bike_df.loc[:, '2021 Counts'].fillna(0)
print(bike_df)

#5. Optional/Bonus: There is a challenging but common issue in the name for the 2019 counts column. Can you find it? If so, Rename the 2019 counts (31 counters) column to "counts_2019" (a more Pythonic column name).

bike_df = bike_df.rename(columns={' 2019 counts (31 counters)': 'counts_2019'})
print(bike_df.head())