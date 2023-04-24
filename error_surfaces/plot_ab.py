from scipy.special import betainc as binc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from jax.config import config


config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from betanegbinfit.betainc import _betainc as bnew

a = 100
bps = np.arange(1, 200 + 1)
as_ = np.arange(1, 200 + 1)
ps = np.linspace(0.0001, 0.99999, 100)
A, B = np.meshgrid(bps, as_)

def test(A, B, p):
    F_scipy = binc(A, B, p)
    F_my = np.zeros_like(F_scipy)
    F_my_inv = np.zeros_like(F_my)
    for x in range(B.shape[0]):
        for y in range(B.shape[1]):
            F_my[x, y] = bnew(A[x, y], B[x, y], p)
            F_my_inv[x, y] = 1 - bnew(B[x, y], A[x, y], 1 - p)
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
        p = ps[i]
        if i and not i % 5:
            print(f'{i} / {len(ps)}')
        pinv = '{:.2f}'.format(1-p)
        pst = '{:.2f}'.format(p)
    else:
        p = i
        pst = p
        pinv = 1 - p
    E1, E2 = test(A, B, p)
    print(E1.shape, E2.shape)
    print(np.mean(E1), np.max(E1))
    ax1 = plt.subplot(1, 2, 1)
    plt.contourf(A, B, E1, vmin=0.0, vmax=1.0, levels=levels)
    plt.ylabel('r')
    plt.xlabel('x')
    plt.title(f'$I_{{{pst}}}(x, r)$')
    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel('x')
    plt.contourf(A, B, E2, vmin=0.0, vmax=1.0, levels=levels)
    plt.yticks([])
    plt.ylabel(str())
    plt.title(f'$1 - I_{{{pinv}}}(r, x)$')
    plt.tight_layout()
    plt.colorbar(ticks=[0, 0.5, 1], ax=[ax1, ax2], pad=0.015).ax.set_ylabel('Relative absolute error', rotation=270,
                                                                            labelpad=12)

print('Plotting single image...')
fig = plt.figure(figsize=(10, 4.5), dpi=300)
plot2(0.5)
plt.savefig('ibeta_ab.pdf')

animate = lambda i, *args: plot2(i, True)
fig = plt.figure(figsize=(10, 4.5), dpi=150)
fun = FuncAnimation(fig, animate, frames=len(ps) , interval=200, repeat=False)

print(f'Plotting animation (will take x{len(ps)} as long)...')
fun.save("ibeta_ab.mp4")






# from scipy.special import betainc as binc
# import numpy as np
# from jax.config import config
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
# from betanegbinfit.distributions import NB
# import jax.numpy as jnp

# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

# from betanegbinfit.betainc import _betainc as bnew
# from betanegbinfit.betainc import _betaincc as bnewc
# from betanegbinfit.betainc import betainc

# xs = np.arange(1, 100 + 1)
# rs = np.arange(1, 100 + 1)
# p = 1 / 2
# X, R = np.meshgrid(xs, rs)

# def test(X, R, p):
#     true_fun = lambda x, r: np.array(list(map(float, NB.long_cdf(x, r, p))))
#     cdf = jnp.vectorize(bnew)
#     cdfc = jnp.vectorize(bnewc)
#     F_true = np.zeros_like(X, dtype=float)
#     F_my1 = np.array(cdf(R, X + 1, 1-p))
#     F_my2 = np.array(cdfc(R, X + 1, 1-p))
#     for i in range(X.shape[0]):
#         x = X[i]
#         r = R[i][0]
#         F_true[i, :] = true_fun(x, r)
#     E1 = np.abs(F_true - F_my1) / F_true
#     E1 = np.clip(E1, 0, 1)
#     E2 = np.abs(F_true - F_my2) / F_true
#     E2 = np.clip(E2, 0, 1)
#     F_all = betainc(R, X + 1, 1 - p)
#     F_scipy = binc(R, X + 1, 1 - p)
#     E_t = np.clip(np.abs(F_true - F_all) / F_true, 0, 1)
#     eps = -float('inf')
#     print(np.mean(E_t[F_scipy > eps]))
#     print(np.mean(E_t[F_true > eps]))
#     return E1, E2, F_true

# E1r, E2r, F_true = test(X, R, p)


# levels = np.linspace(0, 1, 60)

# # def plot2(E1, E2, F_true):
# #     global p
# #     p = 1 - p
# #     ax1 = plt.subplot(1, 3, 1)
# #     plt.contourf(X, R, E1, vmin=0.0, vmax=1.0, levels=levels)
# #     f = np.clip(p / ( 1 - p ) * xs + (0.5 - p) * 5, 1, rs.max())
# #     plt.plot(xs, f, 'r--')
# #     plt.ylabel('r')
# #     plt.xlabel('x')
# #     p1 = '{:.2f}'.format(p)
# #     p2 = '{:.2f}'.format(1 - p)
# #     plt.title(f'$I_{{{p1}}}(r, x)$')
# #     ax2 = plt.subplot(1, 3, 2)
# #     plt.xlabel('x')
# #     plt.contourf(X, R, E2, vmin=0.0, vmax=1.0, levels=levels)
# #     plt.plot(xs, f, 'r--')
# #     plt.yticks([])
# #     plt.ylabel(str())
# #     plt.title(f'$1 - I_{{{p2}}}(x, r)$')
# #     ax3 = plt.subplot(1, 3, 3)
# #     plt.xlabel('x')
# #     plt.contourf(X, R, F_true, vmin=0.0, vmax=1.0, levels=levels)
# #     plt.yticks([])
# #     plt.ylabel(str())
# #     plt.title(r'$log_{10}[groud_truth]')
# #     plt.tight_layout()
# #     plt.colorbar(ticks=[0, 0.5, 1], ax=[ax1, ax2, ax3], pad=0.015)

# def plot2(E1, E2):
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
    
# plt.figure(figsize=(15, 4.5), dpi=300)
# plot2(100)
# plt.savefig('t.pdf')
# # plt.tight_layout()
# # fig = plt.figure(figsize=(8, 5), dpi=150)
# # fun = FuncAnimation(fig, animate, frames=len(res) , interval=140, repeat=True)
# # print('Plotting...')
# # from matplotlib.animation import FFMpegFileWriter as FM
# # fm = FM(fps=20)
# # fun.save("error_surfaces.mp4")
# # fun.save('error_surfaces.gif')



# # bps = np.linspace(0.1, 2000, 500)
# # # aps = np.linspace(0, 5000, 100)
# # ps = np.linspace(0.01, 0.999, 30)
# # B, P = np.meshgrid(bps,ps)
# # F_my = np.zeros_like(P)
# # F_sc = np.zeros_like(P)
# # F_scipy = np.zeros_like(P)
# # E = np.zeros_like(P)
# # E2 = np.zeros_like(P)
# # aps = np.linspace(1, 2000, 500)
# # res2 = dict()
# # from tqdm import tqdm
# # i = 0
# # for a in tqdm(aps):
# #     i += 1
# #     for x in range(len(ps)):
# #         for y in range(len(bps)):
# #             # a = X[x, y, z]
# #             b = B[x, y]
# #             p = P[x, y]
# #             sc = binc(a, b, p)
# #             F_my[x, y] = np.exp(logbetainc(a, b, p))
# #             F_sc[x, y] = logbetaincc(b, a, 1-p)
# #             E[x, y] = abs(F_sc[x, y] - sc)
# #             E2[x, y] = abs(F_my[x, y] - sc)
# #     res2[i] = (E.copy(), E2.copy())


# # # for x in np.linspace(0, 1, 20):
# #     # bb = jnp.exp(logbetainc(a, b, x))
# #     # bsc = binc(a, b, x)
# #     # print('{:.2f} {:.1f} {:.2f} {:.2f}'.format(x, abs(bsc - bb), bsc, bb))
