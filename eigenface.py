#coding=utf-8
'''
Eigenface with PCA
large datasets --> RandomizedPCA
https://stackoverflow.com/questions/54494785/sklearn-0-20-2-import-error-with-randomizedpca
pca = RandomizedPCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

@author: DMao
@time: 10.07.20    14:45
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

'''
# show the original faces
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel = faces.target_names[faces.target[i]])
plt.show()
'''
from sklearn.decomposition import PCA as RandomizedPCA
pca = RandomizedPCA(n_components=150, whiten=False, svd_solver='randomized',
                    copy=True, iterated_power=3, random_state=None)
# pca with eigenvectors
pca.fit(faces.data)
'''
fig, axes = plt.subplots(3, 8, figsize=(9, 4), subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
plt.show()
'''

'''
# cumulative variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
'''
pca = RandomizedPCA(n_components=150, svd_solver='randomized').fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62,47), cmap='binary_r')

ax[0,0].set_ylabel('full-dim\ninput')
ax[1,0].set_ylabel('150-dim\nreconstruction')
plt.show()
