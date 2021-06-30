#coding=utf-8
'''
@author: DMao
@time: 03.07.20    08:54
'''
# Gaussian Naive Bayes

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


'''
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X,y)
lim = plt.axis()
# generate new data and predict the label
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(200,2)
ynew = model.predict(Xnew)
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.3)
plt.axis(lim)
plt.show()

yprob = model.predict_proba(Xnew)
print(yprob[-5:].round(3))
'''

# multinomial naive bayes

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
print(data.target_names)
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# print(train.data[5])
# convert str into vector of numbers via creating pipeline TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

# confusion matrix

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predict label')

plt.show()

def pred_cat(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

x = pred_cat('sending a ')
print(x)

'''
p = model.predict(['discussing abcdefg '])
y = train.target_names[p[0]]
print(y)
'''

