#coding=utf-8
'''
@author: DMao
@time: 08.07.20    16:27
'''
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import seaborn as sns
sns.set()

digit = load_digits()
print(digit.keys())

'''
# show the digit pics
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8,8, i + 1, xticks=[], yticks=[])
    ax.imshow(digit.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 5, str(digit.target[i]))

plt.show()
'''
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digit.data, digit.target, random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

from sklearn import metrics
res = metrics.classification_report(ypred, ytest)
print(res)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel( 'true label' )
plt.ylabel( 'predicted label ')
plt.show()
