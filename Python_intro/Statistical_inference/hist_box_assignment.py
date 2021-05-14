import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_excel('data/catsvdogs.xlsx')

print(df.head())

#1 . Plot a histogram of the Percentage of Dog Owners.
plt.subplot(1,2,1)
df['Percentage of Dog Owners'].hist()
plt.title('Dog Ownership %')



#2 Plot a histogram of the Percentage of Cat Owners.
plt.subplot(1,2,2)
df['Percentage of Cat Owners'].hist()
plt.title('Cat Ownership %')

plt.show()

# 3a. What can we see by comparing these two histograms? What information does this tell us?

#The cat ownership percentage is highly concentrated b/t 25-35%ish with some outlier states on either side. The dog ownership is more evenly distibuted but at a higher ownership percentage thatn that of cats. More households own dogs then cats is also a pretty clear insight from this histogram.

# 3b. Optional/Bonus: plot these two histograms on the same plot. An example of what this could look like is given below:

df['Percentage of Cat Owners'].hist(alpha = .7, label = 'Cats')
df['Percentage of Dog Owners'].hist(alpha = .7, label = 'Dogs')
plt.xlabel("Percentage of Animal Owners")
plt.ylabel("Count")
plt.legend()
plt.show()

#4 4. Create two boxplots on the same plot: one of the mean number of cats per household and the other of the mean number of dogs. An example of what this might look like is given below:

plt.boxplot([df['Mean Number of Cats'], df['Mean Number of Dogs per household']], labels=['Cats', 'Dogs'], notch=True, widths = .7, patch_artist=True)
plt.ylabel("Mean Number of Animals per State")


plt.show()

#5. What can we see by comparing these two boxplots? What information does this tell us?

#the biggest take away is that cat owners are more likely to have multiple cats per household than dog owners are to have multiple dogs per household