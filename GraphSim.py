import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import collections as mc


plt.style.use('dark_background')

fig, ax = plt.subplots()

pointsNum = 10
radius = 1

for i in range(pointsNum):
    x = radius * np.cos(2 * np.pi * i / pointsNum)
    y = radius * np.sin(2 * np.pi * i / pointsNum)
    for j in range(pointsNum):
        x2 = radius * np.cos(2 * np.pi * j / pointsNum)
        y2 = radius * np.sin(2 * np.pi * j / pointsNum)
        ax.plot([x2,x],[y2,y])

plt.show()


