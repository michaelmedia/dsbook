#coding=utf-8
'''
Manifold learning
Isomap on Faces
visualizing structure in Digits
@author: DMao
@time: 13.07.20    14:22
'''
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

faces = fetch_lfw_people(min_faces_per_person=30)
# print(faces.data.shape)

'''
fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')

plt.show()
'''

# plot low dimensional embedding of original high dimension via PCA

from sklearn.decomposition import PCA as RandomizedPCA
'''
model = RandomizedPCA(n_components=100, svd_solver='randomized').fit(faces.data)
ratio = model.explained_variance_ratio_
plt.plot(np.cumsum(ratio))
plt.xlabel('n components')
plt.ylabel('cumulative variance')
plt.show()
'''


from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)
print(proj.shape) # 2D projection of inputs

from matplotlib import offsetbox
def plot_components(data, model, images=None, ax=None, thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    if images is not None:
        min_dis_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) **2, 1)
            if np.min(dist) < min_dis_2: # points too closed wont be shown
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
            ax.add_artist(imagebox)
'''
fig, ax = plt.subplots(figsize=(10, 10))
plot_components(faces.data, model= Isomap(n_components=2), images=faces.images[:, ::2, ::2])
plt.show()
'''

# visualizing structure in Digits
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
#print(mnist.data.shape)

'''
fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(mnist.data[1250 * i].reshape(28, 28), cmap='gray_r')

plt.show()
'''
# manifold learning with 1/30 data volume
data = mnist.data[::30]
target = mnist.target[::30].astype(int)

model = Isomap(n_components=2)
proj = model.fit_transform(data)

'''
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
'''


