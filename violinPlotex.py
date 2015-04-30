# Import modules
import pylab as pl
from scipy import stats
import numpy as np

# Function for Violin Plot

def violin_plot(ax,data,groups,bp=False):
    '''Create violin plot along an axis'''
    dist = max(groups) - min(groups)
    w = min(0.15*max(dist,1.0),0.5)
    for d,p in zip(data,groups):
        k = stats.gaussian_kde(d) #calculates the kernel density
        m = k.dataset.min() #lower bound of violin
        M = k.dataset.max() #upper bound of violin
        x = np.arange(m,M,(M-m)/100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v/v.max()*w #scaling the violin to the available space
        ax.fill_betweenx(x,p,v+p,facecolor='y',alpha=0.3)
        ax.fill_betweenx(x,p,-v+p,facecolor='y',alpha=0.3)
    if bp:
        ax.boxplot(data,notch=1,positions=pos,vert=1)

groups = range(2)
a = np.random.normal(size=100)
b = np.random.normal(size=100)
c = np.random.normal(size=100)

data = np.vstack((a,b,c))
print type(data)
fig = pl.figure()
ax = fig.add_subplot(111)
violin_plot(ax,data,groups,bp=0)
pl.show()