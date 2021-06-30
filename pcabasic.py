#coding=utf-8
'''
unsupervised learning
dimension reduction, noise filtering, feature extraction
@author: DMao
@time: 09.07.20    16:02
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 200 data points

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T


from sklearn.decomposition import PCA

'''
pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_) # direction of vectors
print(pca.explained_variance_) # squared length of vectors


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.axis('equal')
for vector, length in zip(pca.components_, pca.explained_variance_ ):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)

plt.show()
'''

'''
# dimension reduction
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print('original shape:    ', X.shape)
print('transformed shape:  ', X_pca.shape)

X_new = pca.inverse_transform(X_pca)

plt.scatter(X_new[:, 0], X_new[:, 1], c='red')
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.show()
'''
from sklearn.datasets import load_digits
digits = load_digits()
print('original data shape:    ', digits.data.shape)
# project from 8 X 8 into 2 D

'''
pca = PCA(n_components=2)
projected = pca.fit_transform(digits.data)
print('pca data shape:    ', projected.shape)

plt.scatter(projected[:, 0], projected[:, 1], c=digits.target, alpha=0.5, edgecolors='none', cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()
'''

'''
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
'''



def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4), subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8), cmap='binary', interpolation='nearest', clim=(0, 16))

np.random.seed(42)
noisy = np.random.normal(digits.data, 4)  # digits with noise

pca = PCA(0.5).fit(noisy)
print(pca.n_components_)

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)
plt.show()
