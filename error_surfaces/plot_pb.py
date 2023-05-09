from scipy.special import betainc as binc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from jax.config import config


config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from betanegbinfit.betainc import _betainc as bnew

a = 100
as_ = np.arange(10, 200 + 1)
bps = np.arange(1, 200 + 1)
ps = np.linspace(0.0001, 0.99999, len(bps))
B, P = np.meshgrid(bps, ps)

def test(a, B, P):
    F_scipy = binc(B, a, P)
    F_my = np.zeros_like(F_scipy)
    F_my_inv = np.zeros_like(F_my)
    for x in range(B.shape[0]):
        for y in range(B.shape[1]):
            F_my[x, y] = bnew(B[x, y], a, P[x, y])
            F_my_inv[x, y] = 1 - bnew(a, B[x, y], 1 - P[x,  y])
    E1 = np.abs(F_scipy - F_my)
    E2 = np.abs(F_scipy - F_my_inv)
    E1r = E1 / F_scipy
    E2r = E2 / F_scipy
    E1r[np.isclose(F_scipy, 0.0, atol=1e-100) & np.isclose(E1, 0.0, atol=1e-100) ] = 0.0
    E1r[np.isnan(E1r) | (E1r > 1.0)] = 1.0
    E2r[np.isclose(F_scipy, 0.0, atol=1e-100) & np.isclose(E2, 0.0, atol=1e-100) ] = 0.0
    E2r[np.isnan(E2r) | (E2r > 1.0)] = 1.0
    return E1r, E2r


levels = np.linspace(0, 1, 60)

def plot2(i, anim=False):
    if anim:
        a = as_[i]
        if i and not i % 5:
            print(f'{i} / {len(as_)}')
    else:
        a = i
    E1, E2 = test(a, B, P)
    a = '{:3}'.format(a)
    ax1 = plt.subplot(1, 2, 1)
    plt.contourf(P, B, E1, vmin=0.0, vmax=1.0, levels=levels)
    plt.ylabel('x')
    plt.xlabel('p')
    plt.title(f'$I_p(x, {a})$')
    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel('p')
    plt.contourf(P, B, E2, vmin=0.0, vmax=1.0, levels=levels)
    plt.yticks([])
    plt.ylabel(str())
    plt.title(f'$1 - I_{{1-p}}({a}, x)$')
    plt.tight_layout()
    plt.colorbar(ticks=[0, 0.5, 1], ax=[ax1, ax2], pad=0.015).ax.set_ylabel('Relative absolute error', rotation=270,
                                                                            labelpad=12)

print('Plotting single image...')
fig = plt.figure(figsize=(10, 4.5), dpi=300)
plot2(100)
plt.savefig('ibeta_pb.pdf')

animate = lambda i, *args: plot2(i, True)
fig = plt.figure(figsize=(10, 4.5), dpi=150)
fun = FuncAnimation(fig, animate, frames=len(as_) , interval=200, repeat=False)

print(f'Plotting animation (will take x{len(as_)} as long)...')
fun.save("ibeta_pb.mp4")
