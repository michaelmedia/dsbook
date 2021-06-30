#coding=utf-8
'''
for non circular clusters --> Gaussian mixture models
@author: DMao
@time: 15.07.20    13:46
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
X = X[:, ::-1] # flip the x, y axes

from sklearn.cluster import KMeans

'''
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()
'''

from scipy.spatial.distance import cdist
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    centers = kmeans.cluster_centers_
    radii = [ cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    # add circle of clusters
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

'''
kmeans = KMeans(4, random_state=0)
plot_kmeans(kmeans, X)
plt.show()
'''
# Gaussian mixture model
from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)
'''
plt.scatter(X[:, 0], X[:, 1], s=10, c=labels, cmap='viridis')
plt.show()
'''


probs = gmm.predict_proba(X) # [n_samples, n_clusters]
#print(probs[:5].round(3))

size = 50 * probs.max(1) **2
'''
plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, cmap='viridis')
plt.show()
'''
from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

# GMM as Density Estimation
from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=0.05, random_state=0)
#plt.scatter(Xmoon[:, 0], Xmoon[:, 1])


gmms = GMM(n_components=22, covariance_type='full', random_state=0).fit(Xmoon)

#plot_gmm(gmms, Xmoon, label=False)
n_compoents = np.arange(1,21)
models = [ GMM(n_components=n, covariance_type='full', random_state=0).fit(Xmoon) for n in n_compoents]
plt.plot(n_compoents, [ m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_compoents, [ m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()


