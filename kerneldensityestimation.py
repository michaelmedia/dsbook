#coding=utf-8
'''
Kernel Density Estimation
@author: DMao
@time: 17.07.20    15:27
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f*N):] += 5
    return x

x = make_data(20)

hist = plt.hist(x, bins=30, density=True)
# total area under histogram is 1
density, bins, patches = hist
width = bins[1:] - bins[:-1]
print((density * width).sum())

'''
bins = np.linspace(-5, 10, 10)
fig, ax = plt.subplots(1, 2, figsize = (12, 4), sharex=True, sharey=True,
                       subplot_kw={'xlim': (-4, 9), 'ylim': (-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)

for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins = bins + offset, density=True)
    ax[i].plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
'''

'''
fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1, alpha=0.5))
    ax.set_xlim(-4, 8)
    ax.set_ylim(-0.2, 8)
'''

'''
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)
'''
from scipy.stats import norm
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)
'''
plt.fill_between(x_d, density, alpha = 0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
plt.axis([ -4, 8, -0.2, 8])

plt.show()
'''

from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])
#print(x_d[:, None].shape)

loggrob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(loggrob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
grid.fit(x[:, None])
print(grid.best_params_)
