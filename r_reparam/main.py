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


# kappas = [2, 5, 10, 50]
rs = [5, 10, 20, 40]
kappas = np.linspace(2, 200, n)
plt.figure(figsize=(6, 6), dpi=200)
legend = list()
its = []
for r in rs:
    rt = calc_r(r, kappas)
    its.append(plt.plot(kappas, rt)[0]);plt.axhline(r, linestyle='--', color=its[-1].get_color(), linewidth=1)
    legend.append(r'$r = ' + str(r) + r'$')
plt.legend(labels=legend, handles=its)
plt.xlabel('$\kappa$')
plt.ylabel(r'$r_{BetaNB}\left(r, \kappa, \frac{1}{2}\right)$')
plt.tight_layout()
plt.savefig('reparam_1d.pdf')