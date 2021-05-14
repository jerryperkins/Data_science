import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Requirements:
# A non-profit organization focused on promoting and increasing youth literacy rates is looking to hire you to give them data-driven recommendations on where to focus their work and to whom they should be targeting. Your task is to create a detailed presentation to the non-profit executives that describes trends in literacy rate data. They are particularly interested in understanding where they should focus their next project and what strategies to use to have the biggest impact in increasing youth literacy rates in that area.

# You can download the data here. The original dataset is from data.world.

# The first step in this process is to use Python to clean the data to get it ready for analysis. You need to ensure that:

# There are no missing values in this dataset. If there are, deal with these appropriately. Include a brief description of the method you used to deal with missing values along with a justification for that method.
# All columns should match the data types listed in the data dictionary.
# The next step is to complete an analysis based on the data using Python. This should include visualizations and an analysis of trends or any other insights you can gain from the data. You should include at least 3 useful and clear visualizations. Under each visualization, include a brief description of the insights gained from that data visualization.

# The final step is to create a presentation that you could give to the executives. This should be a slide deck (feel free to use any tool you would like - PowerPoint, Google Slides, etc.) with visualizations, insights, and recommendations to the team.

# Some questions you can explore include (but are not limited to):

# What countries have the lowest youth literacy rates?
# What ages have the lowest youth literacy rates?
# Is there a difference among genders?
# What are some trends in youth literacy rates over time?
# What regions have the highest and lowest average literacy rates?
# Data Dictionary:
# Column Name	Data Type	Description
# Region	object	Region that the country is located
# Country	object	Country
# Year	int	Year
# Age	object	Age range: 15-24: Youth ages 15-24, 15+: youth and adults aged 15 or older, adults over 24
# Gender	object	Gender
# Literacy rate	float	Percentage of individuals who have the ability to both read and write*
# *Literacy rate is calculated by taking the number of literate persons in that age and gender group divided by the total number of persons in that group. Literate is defined as the ability to both read and write, with understanding, a short, simple statement about everyday life. Literacy rates at or near 100% indicate that (nearly) every individual in that age range and gender group is able to read and write, at least at a basic level.

# Requirement:
# Code used for data cleaning and analysis along with a slide deck for a presentation that you could present to executives.

# Scoring:
# Make sure you include the following:

# Missing values handled
# Explanation of method used to handle missings
# Any incorrect or odd values are handled
# Column datatypes are converted to match the data dictionary
# 3 useful visualizations
# Interpretations of all 3 visualizations
# Slide deck
# Code is well organized, structured, indented correctly, and commented
# No unnecessary code or files are included


df = pd.read_csv('data/literacy_rates.csv')

print(df.head())

print(df.info())

print(df.isnull().sum())

df = df.dropna()
print(df.info())

print(df['Literacy rate'].value_counts())
print(df['Literacy rate'].sort_values())

# df.loc[18,['Literacy rate']] = .45384



df['Literacy rate'] = df['Literacy rate'].apply(lambda x: float(x.split()[0].replace('%', '')))
df.loc[18,['Literacy rate']] = .45384 
print(df['Literacy rate'].sort_values())
# df['Literacy rate'] = df['Literacy rate'].astype(float)
print(df.info())





# print(df['Literacy rate'].sort_values())


# df = df['Literacy rate'].astype(float)
# print(df.info())