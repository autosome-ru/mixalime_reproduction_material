import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def calc_r(r, k, p=0.5):
    return r * ((1 - p) * k - 1) / (k * (1 - p))

n = 1000
r = np.linspace(2, 200, n)
k = np.linspace(2, 200, n)
R, K = np.meshgrid(r, k)
z = calc_r(R, K)
fig = plt.figure(dpi=200, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(R, K, z, cmap=cm.copper,)
plt.xlabel('r')
plt.ylabel(r'$\kappa$')
ax = plt.subplot(1, 2, 2, projection='3d', )
ax.plot_surface(R, K, z, cmap=cm.copper, linewidth=0)
plt.xlabel('r')
plt.ylabel(r'$\kappa$')
plt.tight_layout()

plt.savefig('reparam.pdf')