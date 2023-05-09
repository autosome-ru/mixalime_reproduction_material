from scipy.special import betainc as binc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from jax.config import config


config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from betanegbinfit.hyp import cdf, _cdf, _cdfc, calc_cond
from betanegbinfit.distributions import BetaNB

p = 0.5
xs = np.arange(1, 200)
rs = np.arange(1, 200 + 1, 10)
ks = np.linspace(5, 200, len(xs))
K, X = np.meshgrid(ks, xs)



def test(X, r, p, K):
    true_fun = lambda x, k: np.array(list(map(float, BetaNB.long_cdf(x, p, k, r))))
    F_true = np.zeros_like(X, dtype=float)
    F_my1 = np.array(_cdf(X, r, p, K))
    F_my2 = np.array(_cdfc(X, r, p, K))
    # F_all = np.array(cdf(X, R, p, k))
    for i in range(X.shape[0]):
        x = X[:, i].copy()
        k = K[0, i].copy()
        F_true[:, i] = true_fun(x, k)
    E1 = np.abs(F_true - F_my1) / F_true
    E1 = np.clip(E1, 0, 1)
    E2 = np.abs(F_true - F_my2) / F_true
    E2 = np.clip(E2, 0, 1)
    # E1[np.isclose(F_true, 0.0, atol=1e-100) & np.isclose(E1, 0.0, atol=1e-100) ] = 0.0
    # E1[np.isnan(E1) | (E1 > 1.0)] = 1.0
    # E2[np.isclose(F_true, 0.0, atol=1e-100) & np.isclose(E2, 0.0, atol=1e-100) ] = 0.0
    # E2[np.isnan(E2) | (E2 > 1.0)] = 1.0
    # E3 = np.abs(F_true - F_all) / F_true
    return E1, E2


levels = np.linspace(0, 1, 60)

def plot2(i, anim=False):
    if anim:
        r = rs[i]
        if i and not i % 5:
            print(f'{i} / {len(rs)}')
    else:
        r = i
    E1, E2 = test(X, r, p, K)
    print(E1.shape, E2.shape)
    print(np.mean(E1), np.max(E1))
    ax1 = plt.subplot(1, 2, 1)
    plt.contourf(K, X, E1, vmin=0.0, vmax=1.0, levels=levels)
    plt.ylabel('x')
    plt.xlabel('$\kappa$')
    plt.title(f'$G_{{BetaNB}}$(x, {r}, {p}, $\kappa$)')
    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel('$\kappa$')
    plt.contourf(K, X, E2, vmin=0.0, vmax=1.0, levels=levels)
    plt.yticks([])
    plt.ylabel(str())
    plt.title(f'$1-G_{{BetaNB}}$({r}-1, x + 1, {p}, $\kappa$)')
    plt.tight_layout()
    plt.colorbar(ticks=[0, 0.5, 1], ax=[ax1, ax2], pad=0.015).ax.set_ylabel('Relative absolute error', rotation=270,
                                                                            labelpad=12)

print('Plotting single image...')
fig = plt.figure(figsize=(10, 4.5), dpi=300)
plot2(100)
plt.savefig('hyp_kx.pdf')

animate = lambda i, *args: plot2(i, True)
fig = plt.figure(figsize=(10, 4.5), dpi=150)
fun = FuncAnimation(fig, animate, frames=len(rs) , interval=200, repeat=False)

print(f'Plotting animation (will take x{len(ks)} as long)...')
fun.save("hyp_kx.mp4")

