import matplotlib.pyplot as plt
import pandas as pd

x = [10,20,30,40,50]
y = [5,10,15,20,25]

# plt.show()
# plt.plot([10,20,30,40,50], [100,400,900,1600,2500])
print(plt.style.available)
plt.style.use('seaborn')
plt.plot(x,y)
plt.show()