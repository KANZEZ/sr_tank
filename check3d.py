import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path0 = "/home/grasp/sr_tank/P1.txt"
path1 = "/home/grasp/sr_tank/phi1.txt"
path2 = "/home/grasp/sr_tank/T1.txt"
df0 = pd.read_csv(path0, sep='\t', header=None)
df1 = pd.read_csv(path1, sep='\t', header=None)
df2 = pd.read_csv(path2, sep='\t', header=None)
P = df0.values.reshape(-1)
phi = df1.values.reshape(-1)
T = df2.values.reshape(-1)


mask2 = (-P>=0) & (-P<=50000) & (phi>=0) & (phi <=1) & (T>=-0.2) & (T<=1)
P = np.ma.masked_where(~mask2,P)
phi = np.ma.masked_where(~mask2,phi)
T = np.ma.masked_where(~mask2,T)

fig=plt.figure(figsize=(8.49 / 2.54, 8.49 / 2.54))
ax1 = Axes3D(fig)
fig.add_axes(ax1)
ax1.scatter3D(P / 1000, phi/np.pi * 180, T, c = T, cmap='rainbow', label='Experimental Data')
ax1.set_ylim(0,60)
ax1.set_xlim(-50,0)
ax1.set_zlim(-0.2, 1)
plt.show()
#ax1.scatter3D(X.ravel(), Y.ravel(), Z.ravel(), c=Z.ravel(), cmap='magma')


