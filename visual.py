#coding=utf-8
'''
@author: DMao
@time: 10.06.20    11:38
'''
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-white')
''' 
# scatt plot

iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.2, s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.colorbar()
plt.show()

# error bar
plt.style.use('seaborn-whitegrid')

x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='o', color='red', elinewidth=3, ecolor='lightgray', capsize=2)

# contour plots

def f(x,y):
    return np.sin(x)**10 + np.cos(10 + y*x)*np.cos(x)
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
# plt.contourf(X, Y, Z, 30, cmap='RdGy')
contours = plt.contour(X, Y, Z, 5, color='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin= 'lower', cmap='RdGy', alpha=0.5)
plt.colorbar()
plt.axis(aspect= 'image')

# 1D histogram
data1 = np.random.randn(1000)
data2 = np.random.normal(-5, 5, 1000)
data3 = np.random.normal(-9, 9, 1000)
# plt.hist(data1, bins= 25, density=True, edgecolor='black', alpha= 0.4, label='data1')
# plt.hist(data2, bins= 25, density=True, edgecolor='black', alpha= 0.4, label='data2')
kwargs = dict(alpha=0.4, density=True, bins=25, edgecolor='black')
plt.hist(data1, **kwargs, label='data1')
plt.hist(data2, **kwargs, label='data1')
plt.hist(data3, **kwargs, label='data1')
count, bin_edges = np.histogram(data1, bins=3)
print(count, bin_edges)

# 2D Histogram
mean = [0, 0]
cov = [[1,1], [1,2]]
x , y = np.random.multivariate_normal(mean, cov, 10000).T
print(x, y)
# plt.hist2d(x, y, bins=50, cmap='Blues')
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
count, xedge, yedge = np.histogram2d(x, y, bins=3)
print(count)

# KDE
from scipy.stats import gaussian_kde
mean = [0, 0]
cov = [[1,1], [1,2]]
x , y = np.random.multivariate_normal(mean, cov, 10000).T
data = np.vstack([x, y])  # n dim, n samples
kde = gaussian_kde(data) # kernel density estimation
xgrid = np.linspace(-3, 3, 40)
ygrid = np.linspace(-7, 7, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto', extent= [-3, 3, -7, 7], cmap= 'Blues')
cb = plt.colorbar(label='density')
plt.show()

# multi legends
fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']
colors = ['black', 'red', 'blue', 'green']
x = np.linspace(0, 10, 1000)
for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi/2), styles[i], color=colors[i] )

ax.axis('equal')
ax.legend(lines[:2], [ 'lineA', 'lineB'], loc='upper right', frameon=False)
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['lineC', 'lineD'], loc='lower right', frameon=False)
ax.add_artist(leg)
plt.show()

from matplotlib.colors import LinearSegmentedColormap

def grayscale_cmap(cmap):
    # return a grayscale version of a given colormap
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    RGB_weight = [ 0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3]**2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

def view_colormap(cmap):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    fig, ax = plt.subplots(2, figsize=(6, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1] )
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    plt.show()
view_colormap('viridis')


from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])




from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)
plt.scatter(projection[:,0], projection[:,1], lw=0.1, c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)
plt.show()

# arrow annotation
x = np.linspace(0, 20, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.cos(x))
ax.axis('equal')
ax.annotate('local min', xy=(5*np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->", connectionstyle='angle3, angleA=0, angleB=-90'))

# hiding ticks or labels

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images
fig, ax = plt.subplots(5,5, figsize=(5,5))
fig.subplots_adjust(hspace=0, wspace=0)

for i in range(5):
    for j in range(5):
        ax[i,j].xaxis.set_major_locator(plt.NullLocator())
        ax[i,j].yaxis.set_major_locator(plt.NullLocator())
        ax[i,j].imshow(faces[30*i + 10*j ], cmap="bone")
plt.show()

# style module
print(plt.style.available[:5])
'''

# 3D plot
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(0, 15, 1000)
y = np.sin(x)
z = np.cos(x)
ax.plot3D(x, y, z, 'blue')

xdata = 15 * np.random.random(100)
ydata = np.sin(xdata) + 0.1 * np.random.randn(100)
zdata = np.cos(xdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
xm = np.linspace(-6, 6, 30)
ym = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(xm, ym)
def f(x,y):
    return np.sin(np.sqrt(x **2 + y **2))
Z = f(X, Y)
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(60, 35)
plt.show()
