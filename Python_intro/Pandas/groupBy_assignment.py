import pandas as pd
import numpy as np

df = pd.read_csv('pandas_data/CalIt2.data.txt', header=None)

# print(df.head())
df = df.rename(columns ={0: 'Flow_ID',
                        1: 'Date',
                        2: 'Time',
                        3: 'Count'})

# Selecting Data:

#1 Select all data for the date July 24 2005 having flow id=7.
flow_ID_7 = df['Flow_ID']==7
july_24 = df['Date']=='07/24/05'
july_filter = df.loc[flow_ID_7 & july_24, :]
print(july_filter)

#2 From the data selected in step 1, sort the result on Count in descending order and output the top 10 rows. Assign the output to a variable named top10.
top10 = july_filter.sort_values(by='Count', ascending=False)[0:10]
print(top10)

# Apply Fcuntion
#1 For the 10 rows outputted above, use Pandas Apply function to subtract lowest value of the 10 from all of them and then output the average value of the resulting counts

x = top10['Count'].min()
count_diff = top10['Count'] - [x] 
print(count_diff.mean())
# The below solution is more corect but I got that with nichole's help. The above way was my own creation.
def subtract(y):
    return y - x
count_diff_alt = top10['Count'].apply(subtract)
print(count_diff_alt.mean())

#Grouping
#1 Select data in the month of August 2005 having flow id=7
start_date = '08/01/05'
end_date = '08/31/05'
august = (df['Date'] >= start_date) & (df['Date'] <= end_date)
august_filter = df.loc[august & flow_ID_7]
print(august_filter)

#2 Group the data based on date and get the max count per date
print(august_filter.groupby(['Date']).max())