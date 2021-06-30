#coding=utf-8
'''
@author: DMao
@time: 03.07.20    15:15
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# simple linear regression

rng = np.random.RandomState(1)
x = 10 * rng.rand(50) # 1D
y = 2 * x - 5 + rng.randn(50)

plt.scatter(x, y)


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)  # change into 2D
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.plot(xfit, yfit)

print('model slope: ', model.coef_[0])
print('model intercept: ', model.intercept_)

# multi dimensional linear regression  y = a + bx1 + cx2 + ...
X = 10 * rng.rand(100, 3) # 2D
y = 0.5 + np.dot(X, [1.5, -2., 1.]) # np matrix mulitplication
model.fit(X, y)
print(model.coef_)
print(model.intercept_)

# polynomia basis function   y = a + bx + cxˆ2 + dx^3 + ...
from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4]) # 1D
poly = PolynomialFeatures(3, include_bias=False)  # transform into 3D
new = poly.fit_transform(x[:, None])
'''
None is an alias for NP.newaxis. It creates an axis with length 1. 
This can be useful for matrix multiplcation etc.
'''

print(new)

from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

rng = np.random.RandomState(1)
x = 10 * rng.rand(50) # 1D
y = np.sin(x) + 0.1 * rng.randn(50) # 1D
poly_model.fit(x[:, np.newaxis], y)
xfitm = np.linspace(0, 10, 1000)
yfit = poly_model.predict(xfitm[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
