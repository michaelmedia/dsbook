#coding=utf-8
'''
@author: DMao
@time: 25.05.20    16:56
'''
import numpy as np
'''   
for i in [2,4,6]:
    print(i)

x = np.array([ range(i, i+10) for i in [2,4,6]])

print(x)
a1 = np.random.randint(100, size=(3,4))
print(a1.nbytes)

b = np.array([ 11,22,33])
print(b)
b1 = b.reshape((3,1))
print(b1)
b2 = b[:, np.newaxis]
print(b2)

b3 = np.array([11,22,33])
grid = np.array([[1,2,3],
                [4,5,6]])

d1 = np.array([1,2,3])
d2 = np.array([4,5,6])
d = np.dstack([d1, d2])
print(d)
ds = np.dsplit(d, [1])
print(ds)


x  = np.arange(5)
print(x[1:])
y = np.random.randint(100, size=10)
print(y)
inx = [ 0,2,3]
print(y[inx])
'''

x = np.zeros(5)
print(x)

name = [ 'abc', 'bcd', 'cde', 'efg']
age = [10, 15,17,19]
weight = [ 55, 65, 70, 72]
data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                          'formats': ( 'U10', 'i4', 'f8')})
data['name'] = name
data[ 'age'] = age
data[ 'weight'] = weight
print(data[-1]['age'])

tp = np.dtype([( 'id', 'i8'), ( 'mat', 'f8', ( 3,3))])
a = np.zeros(1, dtype=tp)
print(a['mat'][0])

data_rec = data.view(np.recarray)
print(data_rec)
print(data_rec['age'])
print(data_rec.age)
