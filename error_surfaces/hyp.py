from scipy.special import betainc as binc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from jax.config import config


config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
from gmpy2 import get_context
from mpmath import mp


from betanegbinfit.hyp import cdf, _cdf, _cdfc, calc_cond
from betanegbinfit.distributions import BetaNB

k = 100
m = 100
xs = np.arange(1, m + 1)
rs = np.arange(1, m + 1)
p = 0.5
X, R = np.meshgrid(xs, rs)

def test(X, R, p, k):
    true_fun = lambda x, r: np.array(list(map(float, BetaNB.long_cdf(x, p, k, r))))
    F_true = np.zeros_like(X, dtype=float)
    F_my1 = np.array(_cdf(X, R, p, k))
    F_my2 = np.array(_cdfc(X, R, p, k))
    F_all = np.array(cdf(X, R, p, k))
    for i in range(X.shape[0]):
        x = X[i].copy()
        r = R[i][0].copy()
        F_true[i, :] = true_fun(x, r)
    E1 = np.abs(F_true - F_my1) / F_true
    E1 = np.clip(E1, 0, 1)
    E2 = np.abs(F_true - F_my2) / F_true
    E2 = np.clip(E2, 0, 1)
    E3 = np.abs(F_true - F_all) / F_true
    return E1, E2, E3
E1, E2, E3 = test(X, R, p, k)
levels = np.linspace(0, 1, 60)


plt.figure(figsize=(6 * 3 + 1, 5), dpi=200)
plt.subplot(1, 3, 1)
plt.title('Relative error: 3f2')
plt.contourf(X, R, E1, vmin=0.0, vmax=1.0, levels=levels)
f = np.clip(calc_cond(xs, p, rs), 1, rs.max())
plt.plot(xs, f, 'r')
plt.xlabel('X')
plt.ylabel('R')

# # pt = 1 / (1 + np.exp(k))
# # a, b, c = cf[0]
# plt.plot(xs, np.clip(xs * (1 - p) / p +  (1 - p) / p * 100 * xs / k  + 5 , 0, rs.max()), 'r', linewidth=4)
# plt.plot(xs, np.clip((1 - p) / p  * xs  + 5 , 0, rs.max()), 'y', linewidth=4)

plt.subplot(1, 3, 2)
plt.contourf(X, R, E2, vmin=0.0, vmax=1.0, levels=levels)
plt.plot(xs, f, 'r')
plt.xlabel('X')
# plt.ylabel('R')
plt.title('Complement 3f2')
plt.subplot(1, 3, 3)
plt.contourf(X, R, E3,  levels=levels)
plt.xlabel('X')
plt.title('Combined 3f2')
plt.suptitle(f'k = {k}, p = {p}')
plt.tight_layout()


# k = 200
# xs = np.arange(1, 200 + 1)
# ps = np.linspace(0.01, 0.99, len(xs))
# r = 100
# X, P = np.meshgrid(xs, ps)

# def test2(X, r, P, k):
#     true_fun = lambda x, p: np.array(list(map(float, BetaNB.long_cdf(x, p, k, r))))
#     F_true = np.zeros_like(X, dtype=float)
#     F_my1 = np.array(cdf(X, r, P, k))
#     F_my2 = np.array(cdf(X, r, P, k, switch=True))
#     for i in range(X.shape[0]):
#         x = X[i]
#         p = P[i][0]
#         F_true[i, :] = true_fun(x, p)
        
#     E1 = np.abs(F_true - F_my1) / F_true
#     # E1[(np.isclose(F_true, 0.0)) & (np.isclose(F_my1, 0.0))] = 0
#     E1 = np.clip(E1, 0, 1)
#     E2 = np.abs(F_true - F_my2) / F_true
#     # E2[(np.isclose(F_true, 0.0)) & (np.isclose(F_my2, 0.0))] = 0
#     E2 = np.clip(E2, 0, 1)
#     return E1, E2, F_true, F_my1, F_my2

# E1, E2, F_true, F_my1, F_my2 = test2(X, r, P, k)
# levels = np.linspace(0, 1, 60)


# plt.figure(figsize=(6 * 3 + 1, 5), dpi=200)
# plt.subplot(1, 3, 1)
# plt.title('Relative error: 3f2')
# plt.contourf(X, P, E1, vmin=0.0, vmax=1.0, levels=levels)
# plt.xlabel('X')
# plt.ylabel('p')
# plt.subplot(1, 3, 2)
# plt.contourf(X, P, E2, vmin=0.0, vmax=1.0, levels=levels)
# plt.xlabel('X')
# # plt.ylabel('R')
# plt.title('Relative error: complement 3f2')
# plt.subplot(1, 3, 3)
# plt.contourf(X, P, -np.log10(np.abs(F_true)), levels=20)
# plt.xlabel('p')
# # plt.ylabel('R')
# plt.title('CDF value')
# plt.suptitle(f'k = {k}, r = {r}')
# plt.tight_layout()


# plt.figure(figsize=(6 * 2 + 1, 5), dpi=200)
# plt.subplot(1, 2, 1)
# plt.contourf(X, R, F_my, vmin=0.0, vmax=1.0, levels=levels)
# plt.xlabel('X')
# plt.ylabel('R')
# plt.subplot(1, 2, 2)
# plt.contourf(X, R, F_true, vmin=0.0, vmax=1.0, levels=levels)
# plt.xlabel('X')
# plt.ylabel('R')
# plt.tight_layout()


# def plot2(i, anim=False):
#     if anim:
#         a = as_[i]
#         if i and not i % 5:
#             print(f'{i} / {len(as_)}')
#     else:
#         a = i
#     E1, E2 = test(a, B, P)
#     print(np.mean(E1), np.max(E1))
#     a = '{:3}'.format(a)
#     ax1 = plt.subplot(1, 2, 1)
#     plt.contourf(P, B, E1, vmin=0.0, vmax=1.0, levels=levels)
#     plt.ylabel('r')
#     plt.xlabel('p')
#     plt.title(f'$I_p({a}, r)$')
#     ax2 = plt.subplot(1, 2, 2)
#     plt.xlabel('p')
#     plt.contourf(P, B, E2, vmin=0.0, vmax=1.0, levels=levels)
#     plt.yticks([])
#     plt.ylabel(str())
#     plt.title(f'$1 - I_{{1-p}}(r, {a})$')
#     plt.tight_layout()
#     plt.colorbar(ticks=[0, 0.5, 1], ax=[ax1, ax2], pad=0.015).ax.set_ylabel('Relative absolute error', rotation=270,
#                                                                             labelpad=12)

# print('Plotting single image...')
# fig = plt.figure(figsize=(10, 4.5), dpi=300)
# plot2(100)
# plt.savefig('ibeta_pb.pdf')

# # animate = lambda i, *args: plot2(i, True)
# # fig = plt.figure(figsize=(10, 4.5), dpi=150)
# # fun = FuncAnimation(fig, animate, frames=len(as_) , interval=200, repeat=False)

# # print(f'Plotting animation (will take x{len(as_)} as long)...')
# # # from matplotlib.animation import FFMpegFileWriter as FM
# # # fm = FM(fps=20)
# # fun.save("ibeta_pb.mp4")
