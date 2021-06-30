#coding=utf-8
'''
Manifold Learning
@author: DMao
@time: 10.07.20    15:28
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# create data points in HELLO shape

def make_hello(N=100, rseed=0):
    fig, ax = plt.subplots(figsize=(4,1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)

    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]

X = make_hello(1000)
colorized = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
#plt.scatter(X[:,0], X[:, 1], **colorized)
#plt.axis('equal')


# rotated HELLO via rotaion matrix
def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)

X2 = rotate(X, 20) + 5
'''

plt.scatter(X2[:,0], X2[:, 1], **colorized)
plt.axis('equal')
plt.show()

'''
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
print(D.shape)



D2 = pairwise_distances(X2)
print(np.allclose(D, D2))

'''
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.show()
'''
# MDS multi dimention scaling
from sklearn.manifold import  MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
'''
plt.scatter(out[:, 0], out[:,1], **colorized)
plt.axis('equal')
plt.show()
'''

# MDS as Manifold Learning
def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])
X3 = random_projection(X, 3)
print(X3.shape)

model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorized)
plt.axis('equal')
plt.show()
