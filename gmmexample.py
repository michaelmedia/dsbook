#coding=utf-8
'''
@author: DMao
@time: 16.07.20    17:19
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

# show original data
def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8,8), cmap='binary')
        im.set_clim(0, 16)

'''
plot_digits(digits.data)
plt.show()
'''

# build GMM on dimensional reduced original data
# PCA to preserve 99% of the variance
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
print(data.shape)

# use AIC Akaike information criterion to get n_components of GMM
n_components = np.arange(50, 210, 10)
models = [GMM(n, covariance_type='full', random_state=0)
          for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
'''
plt.plot(n_components, aics)
plt.show()
'''
gmm = GMM(n_components=110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)
data_newX, data_newy = gmm.sample(100) # return X, y
print(data_newX.shape)

digits_new = pca.inverse_transform(data_newX)
plot_digits(digits_new)
plt.show()


