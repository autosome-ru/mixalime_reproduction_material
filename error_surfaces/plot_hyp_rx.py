from scipy.special import betainc as binc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from jax.config import config


config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from betanegbinfit.hyp import cdf, _cdf, _cdfc, calc_cond
from betanegbinfit.distributions import BetaNB

a = 100
k = 50
rs = np.arange(1, 200 + 1)
xs = np.arange(1, 200 + 1)
ps = np.linspace(0.0001, 0.99999, 100)
R, X = np.meshgrid(rs, xs)



def test(X, R, p, k):
    true_fun = lambda x, r: np.array(list(map(float, BetaNB.long_cdf(x, p, k, r))))
    F_true = np.zeros_like(X, dtype=float)
    F_my1 = np.array(_cdf(X, R, p, k))
    F_my2 = np.array(_cdfc(X, R, p, k))
    # F_all = np.array(cdf(X, R, p, k))
    for i in range(X.shape[0]):
        x = X[:, i].copy()
        r = R[0, i].copy()
        F_true[:,i] = true_fun(x, r)
    E1 = np.abs(F_true - F_my1) / F_true
    E1 = np.clip(E1, 0, 1)
    E2 = np.abs(F_true - F_my2) / F_true
    E2 = np.clip(E2, 0, 1)
    # E3 = np.abs(F_true - F_all) / F_true
    return E1, E2


levels = np.linspace(0, 1, 60)

def plot2(i, anim=False):
    if anim:
        p = ps[i]
        if i and not i % 5:
            print(f'{i} / {len(ps)}')
        pinv = '{:.2f}'.format(1-p)
        pst = '{:.2f}'.format(p)
    else:
        p = i
        pst = p
        pinv = 1 - p
    E1, E2 = test(X, R, p, k)
    print(E1.shape, E2.shape)
    print(np.mean(E1), np.max(E1))
    ax1 = plt.subplot(1, 2, 1)
    plt.contourf(X, R, E1, vmin=0.0, vmax=1.0, levels=levels)
    plt.ylabel('x')
    plt.xlabel('r')
    plt.title(f'$G_{{BNB}}$(x, r, {pst}, {k})')
    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel('x')
    plt.contourf(X, R, E2, vmin=0.0, vmax=1.0, levels=levels)
    plt.yticks([])
    plt.ylabel(str())
    plt.title(f'$1-G_{{BNB}}$(r-1, x + 1, {pinv}, {k})')
    plt.tight_layout()
    plt.colorbar(ticks=[0, 0.5, 1], ax=[ax1, ax2], pad=0.015).ax.set_ylabel('Relative absolute error', rotation=270,
                                                                            labelpad=12)

print('Plotting single image...')
fig = plt.figure(figsize=(10, 4.5), dpi=300)
plot2(0.5)
plt.savefig('hyp_xr.pdf')

# animate = lambda i, *args: plot2(i, True)
# fig = plt.figure(figsize=(10, 4.5), dpi=150)
# fun = FuncAnimation(fig, animate, frames=len(ps) , interval=200, repeat=False)

# print(f'Plotting animation (will take x{len(ps)} as long)...')
# fun.save("hyp_xr.mp4")

