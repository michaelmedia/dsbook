#coding=utf-8
'''
@author: DMao
@time: 23.06.20    20:54
'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

''' 
iris = sns.load_dataset('iris')
print(iris.head())
sns.set()
sns.pairplot(iris, hue='species', size=1.5)

X_iris = iris.drop('species', axis=1)
print(X_iris)
print(X_iris.shape)
y_iris = iris['species']
print(y_iris.shape)



plt.show()

# supervised learning simple linear regression
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = x * 2 - 1 + rng.randn(50)


# choose a model and hyperparameters
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
# arrange data into a features matrix 2D and target vector 1D
X = x[:, np.newaxis]
print(X.shape)
print(y.shape)
# fit model to your data
model.fit(X, y)
# predict label for unknown data

xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.plot(xfit, yfit)


plt.scatter(x, y)
plt.show()
'''
# supervised learning Iris classification
iris = sns.load_dataset('iris') # iris dataset
print(iris.head())

X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
print(X_iris.shape)
print(y_iris.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# unsupervised learning Iris dimensionality
from sklearn.decomposition import PCA
model = PCA(n_components=2) # model returns 2 components = reducing data into 2D
model.fit(X_iris)
X_2D = model.transform(X_iris)  # transform the data to 2D
iris['PCA1'] = X_2D[:, 0]  # insert new data into original dataset
iris['PCA2'] = X_2D[:, 1]

# sns.lmplot('PCA1', 'PCA2', hue= 'species', data=iris, fit_reg=False)

# unsupervised learning Iris clustering
from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris)
y_gmm = model.predict(X_iris) # determine cluster labels
iris['cluster'] = y_gmm
sns.lmplot('PCA1', 'PCA2', data=iris, hue='species', col='cluster', fit_reg=False)
plt.show()
