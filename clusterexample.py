#coding=utf-8
'''
@author: DMao
@time: 14.07.20    18:19
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data) # np ndarray
print(kmeans.cluster_centers_.shape) # 10 clusters in 64 Dimensions

'''
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, centers in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(centers, interpolation='nearest', cmap=plt.cm.binary)

plt.show()
'''


# matching learned label with the true label
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0] # true label

from sklearn.metrics import accuracy_score
score = accuracy_score(digits.target, labels)
print(score)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
'''
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predict label')
plt.show()
'''
#print(mat)


# using t-SNE / t distributed stochastic neighbor embedding (none linear) before clustering for better results
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init ='pca', random_state=0)
digits_proj = tsne.fit_transform(digits.data)
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
acc = accuracy_score(digits.target, labels)
print('after using t-SNE:   ', acc)


from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
print('original pic shape:  ', china.shape) # height, width, RBG
'''
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
plt.show()
'''
# rescale pic into 0-1
data = china / 255.0
# reshape pic into n_sample X n_features
data = data.reshape(427*640, 3)
print('reshaped pic shape', data.shape)
# visualize these pixels in color space
def plot_pixel(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel= 'Red', ylabel= 'Green', xlim= (0, 1), ylim= (0, 1))
    ax[1].scatter(R, B, color=colors, marker= '.')
    ax[1].set(xlabel= 'Red', ylabel='Blue', xlim= ( 0, 1), ylim= ( 0, 1))
    fig.suptitle(title, size=20)

'''
plot_pixel(data, title= 'Input color space')
plt.show()
'''
# reduce colors via mini batch clustering
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
'''
plot_pixel(data, colors=new_colors, title='Reduced color space 16 colors')
plt.show()
'''
recolored = new_colors.reshape(china.shape)
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('original pic', size=16)
ax[1].imshow(recolored)
ax[1].set_title('16-color pic', size=16)
plt.show()

