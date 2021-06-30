#coding=utf-8
'''
support vector machine
@author: DMao
@time: 06.07.20    13:22
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set()

from sklearn.datasets import make_blobs
from sklearn.svm import SVC

''' 
X, y = make_blobs(n_samples= 50, centers = 2, random_state=0, cluster_std= 0.6)
# X : [[], [] ] y: []

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')

xfit = np.linspace(-1, 3.5)



plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
# print(xfit)

for m, b in [(1, 0.65), (0.5, 1.6), ( -0.2, 2.9)]:
    plt.plot(xfit, m*xfit + b, '-k')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), ( -0.2, 2.9, 0.2)]:
    yfit = xfit * m + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit -d, yfit + d, edgecolor='none', color='gray', alpha=0.3)
plt.xlim(-1, 3.5)
plt.show()


# SVM with linear kernel support vector classifier
from sklearn.svm import SVC

model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

# plot the decision boundary of SVM
def plot_svc_decision(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca() # get current axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create a grid
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # plt support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plot_svc_decision(model)
plt.show()
print(model.support_vectors_)

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.06)

    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    #ax = ax or plt.gca()
    if ax is None:
        plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    #ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision(model, ax)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
#fig.subplots_adjust(left= 0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))


plt.show()



from sklearn.datasets import make_circles

X, y = make_circles(100, factor= 0.1, noise=0.1)
clf = SVC(kernel='rbf', C=1E6).fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision(clf, plot_support=False)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=300, lw=1, facecolors='none')

'''

# soft margin
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

for axi, C in zip(ax, [ 10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', s=50)
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, lw=1, facecolors='none')
    axi.set_title("C={0:.1f}".format(C), size=14)



plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()
