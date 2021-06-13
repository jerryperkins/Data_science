import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, plot_roc_curve


df= pd.read_csv('data/bank_modified.csv')
print(df.head())
print(df.info())
print(df['pdays'].value_counts()) # after reading this I am making the assumption that 999 is now -1 for this data set and represents not contacted

X = df.loc[:, df.columns[(df.columns != 'y_yes')]]
y = df.loc[:,'y_yes'].values

print(X.shape)
print(y.shape)

print(df['y_yes'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', multi_class='ovr',)
log_reg.fit(X_train, y_train)
print('training accuracy:', log_reg.score(X_train, y_train)) #training accuracy: 0.8881163084702908
print('testing accuracy', log_reg.score(X_test, y_test)) #testing accuracy 0.8931466470154753
preds = log_reg.predict(X_test)
print(preds)
print(X_test)

print("sensitivity score built in", precision_recall_fscore_support(y_test, preds)) # Sensitvity = 0.23076923076923078

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print(tn,fp,fn,tp)

sensitivity = tp / (fn + tp)
print(sensitivity) #0.23076923076923078

specificity = tn / (tn+fp)
print(specificity) # 0.9791840133222315

print(f'Training AUC: {roc_auc_score(y_train, log_reg.predict_proba(X_train)[:,1])}')
print(f'Testing AUC: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1])}')

plot_roc_curve(log_reg, X_train, y_train)
plt.plot([0,1],[0,1], ls = '--', label = "Baseline (AUC = 0.5")
plt.legend()
plt.show()

#1.Which metric is going to be best to evaluate your classification model for this dataset and why? Hint: think about what each metric means in this specific scenario. What metric is going to help the bank the most?

# Specificity is the best metric for the bank because it is most important that people not default on their loans. Despite the bad rep banks get, everyone loses when someone who borrows money defaults

# Compare your model's accuracy with the baseline model. Do you see any problems here?

# The baseline comes in at roughly 87% of No's so when our model gives us an accuracy of 88.7% we are not doing much better than just assigning everyone a no and saying we got 87% right.

# What are other ways you could improve this model?
# put greather epmhasis on finding false positives as those are the results that are very bad for the bank