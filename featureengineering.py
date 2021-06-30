#coding=utf-8
'''
@author: DMao
@time: 02.07.20    09:44
'''
# Categorical Features

data = [{'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
        {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
        {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
        {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}]

# old methods 1
temp_list = []
for i in data:
    for key, value in i.items():
        if key == 'neighborhood':
            temp_list.append(value)
print(temp_list)
temp_dic = {}
for key in temp_list:
    temp_dic[key] = temp_dic.get(key, 0)  + 1

print(temp_dic)
# Counter api
from collections import Counter
result = Counter(temp_list)
print(result)

# one hot encoding via sklearn
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)  # sparse = True very efficient solution only when many 0
x= vec.fit_transform(data)
print(x)
print(vec.get_feature_names())  # get columns names of vec

# text features

sample = ['problem of evil',
          'evil queen',
          'horizon problem']

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)


import pandas as pd
a = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(a)

# Term frequency - inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer()
X = v.fit_transform(sample)

b = pd.DataFrame(X.toarray(), columns=v.get_feature_names())
print(b)
'''
# derived features
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5]) # 1D array
y = np.array([4, 2, 1, 3, 7])
from sklearn.linear_model import LinearRegression

X = x[:, np.newaxis] # 2D array
model = LinearRegression().fit(X, y) # need 2D array
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit, label='line')


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)

model2 = LinearRegression().fit(X2, y)

yfit2 = model2.predict(X2)
plt.plot(x, yfit2, label='poly')
plt.legend()
plt.show()
'''


# imputation of missing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy import nan
Xm = np.array([[ nan, 0,   3  ],
               [ 3,   7,   9  ],
               [ 3,   5,   2  ],
               [ 4,   nan, 6  ],
               [ 8,   8,   1  ]])
ym = np.array([14, 16, -1,  8, -5])


from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
X3 = imp.fit_transform(Xm)
print(X3)
model3 = LinearRegression().fit(X3, ym) # need 2D array
yfit3 = model3.predict(X3)
print(yfit3)

# feature pipeline

from sklearn.pipeline import  make_pipeline

modelp = make_pipeline(SimpleImputer(strategy='mean'),
                       PolynomialFeatures(degree=2),
                       LinearRegression())
modelp.fit(Xm, ym) # here with missing value Xm
print(ym)
print(modelp.predict(Xm))
