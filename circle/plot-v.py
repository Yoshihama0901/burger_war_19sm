#!/usr/bin/env python
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv(sys.argv[1])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(df['circle_y'], df['circle_r'], df['v'])
ax.set_xlabel('circle_y')
ax.set_ylabel('circle_r')
ax.set_zlabel('v')

plt.show()
