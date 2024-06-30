from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')

r=1;
u=np.linspace(-2,2,200);
v=np.linspace(0,2*np.pi,60);
[u,v]=np.meshgrid(u,v);

a = 1
b = 1
c = 1.2



x = a*np.cosh(u)*np.cos(v)
y = b*np.cosh(u)*np.sin(v)
z = c*np.sinh(u)

ax.plot_surface(x, y, z,  rstride=4, cstride=8, cmap=cm.coolwarm, linewidth=1)


ax.set_xlim3d([-5, 5])
ax.set_ylim3d([-5, 5])
ax.set_zlim3d([-5, 5])
ax.set_box_aspect([1,1,1])
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


plt.savefig('./segre.png', dpi=400)
plt.show()