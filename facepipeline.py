#coding=utf-8
'''
Face Detection Pipeline
HOG: Histogram of Oriented Gradients
@author: DMao
@time: 21.07.20    17:50
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from skimage import data, color, feature
import skimage.data


'''
image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualize=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')
ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG feature')
plt.show()
'''
# a simple face dector
# 1. obtain positive training set (containing faces)
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
#print(positive_patches.shape) #13233, 62, 47

# 2. obtain negative training set ( non faces)
from skimage import data, transform

imgs_to_use = [ 'camera', 'text', 'coins', 'moon', 'page', 'clock',
                'immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field']
images = [ color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]

from sklearn.feature_extraction.image import PatchExtractor

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches

negative_patches = np.vstack([ extract_patches(im, 1000, scale)
                               for im in images for scale in [0.5, 1.0, 2.0]])
#print(negative_patches.shape)  # 30000, 62, 47

'''
fig, ax = plt.subplots(6, 10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500 * i], cmap='gray')
    axi.axis('off')

plt.show()
'''
# 3. combine sets and extract HOG feature
from itertools import chain
X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[: positive_patches.shape[0]] = 1

#print(X_train.shape) # 43233, 1215  43k samples in 1215 Dimension

# 4. train SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
'''
score = cross_val_score(GaussianNB(), X_train, y_train)
#print('score of NB classifier', score)
'''


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
'''
score = grid.best_score_ # find the best estimator
print(score)
print(grid.best_params_)
'''


model = grid.best_estimator_
model.fit(X_train, y_train)

# 5. find faces in new datasets
test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = test_image[:290, 60:340]
'''
plt.imshow(test_image, cmap='gray')
plt.axis('off')
plt.show()
'''


# create a window iterates over patches and compute HOG features
def sliding_window(img, patch_size=positive_patches[0].shape, istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale*s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([ feature.hog(patch) for patch in patches])
print(patches_hog.shape)

labels = model.predict(patches_hog)
print(labels.sum())

fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')
Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
plt.show()
