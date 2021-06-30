#coding=utf-8
'''
@author: DMao
@time: 21.07.20    17:00
'''

'''
import numpy as np
d = {'mean_fit_time1': [0.00244441, 0.00226107, 0.00215297, 0.00229616, 0.00216675],
     'mean_fit_time2': [0.00244441, 0.00226107, 0.00215297, 0.00229616, 0.00216675],
     'mean_fit_time3': [0.00244441, 0.00226107, 0.00215297, 0.00229616, 0.00216675]}

for i in d.values():
    print(np.mean(i))

import skimage.data
from skimage import data, transform
from skimage import data, color, feature
import matplotlib.pyplot as plt

test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = test_image[:290, 60:340]
plt.imshow(test_image, cmap='gray')
plt.axis('off')
plt.show()

'''

liste = [ 2, 5, 6,8, 90]
#print(sum(liste))

old = ['old: 20', 'new: 22', 'old: 100', 'new: 25', 'old: 34', 'old: 56']
new = []
for p in old:
    print(type(p.split(':')))
    #print(int(p.split(':')[1]))
    price =  int(p.split(':')[1])
    if 'old' in p:
        if price < 20:
            price = price * 0.8
        elif price < 30:
            price = price * 0.7
        else:
            price = price * 0.5
    new.append(price)

print(new)
