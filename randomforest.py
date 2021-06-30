#coding=utf-8
'''
@author: DMao
@time: 08.07.20    13:17
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# creating a decision tree
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)
# print(X)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=30)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)

# for visualization

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=30, clim=(y.min(), y.max()), zorder=3 )
    ax.axis('tight')
    ax.axis('off')
    xlim= ax.get_xlim()
    ylim= ax.get_ylim()

    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes +1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()), zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


# visualize_classifier(DecisionTreeClassifier(), X, y)


# ensembles of estimators : random forests

'''
# decision trees + bagging
from sklearn.ensemble import BaggingClassifier

tre = DecisionTreeClassifier()
bag = BaggingClassifier(tre, n_estimators=100, max_samples=0.8, random_state=1)
bag.fit(X, y)

visualize_classifier(bag, X, y)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y)

plt.show()
'''

rng = np.random.RandomState(42)
x = 10 * rng.rand(200) # 1D list, single feature
# print(x)
def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))
    return slow_oscillation + fast_oscillation + noise
y = model(x)
print(y) # 1D list
plt.errorbar(x, y, 0.3, fmt='*', alpha=0.5)

# random forest regression
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=200)
forest.fit(x.reshape(-1, 1), y)
xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])  # or reshape(-1, 1)
ytrue = model(xfit, sigma=0)
plt.plot(xfit, yfit, '-r', label= 'predict')
plt.plot(xfit, ytrue, '-k', alpha=0.5, label='true')
plt.legend()
plt.show()
