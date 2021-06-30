#coding=utf-8
'''
@author: DMao
@time: 30.06.20    13:25
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print(digits.images.shape) # 3D data 1797, 8, 8


'''
fig, axes = plt.subplots(10, 10, figsize= (8,8),
                        subplot_kw= {'xticks':[], 'yticks':[]},
                        gridspec_kw= dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap= 'binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')

'''



X = digits.data # 2D

y = digits.target # 1D

# dimension reduction of 2D for visualization (unsupervised learning)

from sklearn.manifold import Isomap

iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
print(data_projected.shape) # 1797, 2    2D

'''
plt.scatter(data_projected[:, 0], data_projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral',10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
'''



# classification on digits

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
diff = accuracy_score(y_test, y_pred)

# confusion matrix
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, y_pred)
print(mat)

'''
import seaborn as sns
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
'''

fig, axes = plt.subplots(10, 10, figsize = (8,8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap= 'binary', interpolation='nearest')
    
    ax.text(0.05, 0.05, str(y_pred[i]),
            transform=ax.transAxes,
            color='green' if (y_test[i] == y_pred[i]) else 'red')

plt.show()
