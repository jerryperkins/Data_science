import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rows = ['Actually 0', 'Actually 1']
columns = ['Predicted 0', 'Predicted 1']
data = np.array([[87,16],
                [17,59]])

fig, ax  = plt.subplots()
im = ax.imshow(data, cmap='Blues')
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
for i in range(len(rows)):
    for j in range(len(columns)):
        if data[i,j] > 50:
            c = 'w'
        else:
            c='b'
        text = ax.text(j, i, data[i, j], ha='center', va='center', color=c)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('', rotation=90, va='bottom')


fig.tight_layout()
plt.show()