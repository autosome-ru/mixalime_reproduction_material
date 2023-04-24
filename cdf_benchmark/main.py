from betanegbinfit.betainc import betainc as betainc_my
from betanegbinfit.distributions import NB as dist
from scipy.special import betainc as betainc_scipy
from mpmath import betainc as betainc_mp
import gmpy2
import numpy as np
from jax.config import config
from matplotlib import pyplot as plt
from jax import numpy as jnp
from tqdm import tqdm



logpdf = jnp.vectorize(dist.logprob)

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
gmpy2.get_context().precision = 256

print('Testing I_0.5(x,x) = 0.5...')
xs = np.arange(1, 10000, 1)
rs = np.arange(1, 10000, 1)
bad = 1
p = bad / (bad + 1)
t0 = betainc_scipy(xs, rs, p)
t1 = betainc_my(xs, rs, p)
diff_scipy = np.abs(t0 - 0.5)
diff_our = np.abs(t1 - 0.5)
print('scipy:', f'Max error: {diff_scipy.max()}, Mean error: {diff_scipy.mean()}')
print('We:', f'Max error: {diff_our.max()}, Mean error: {diff_our.mean()}')

plt.figure(figsize=(5, 4), dpi=200)
scipy, = plt.plot(np.abs(t0 - 0.5), 'bo', markersize=1.5)
my, = plt.plot(np.abs(t1 - 0.5), 'ro', markersize=1.5, alpha=1)
plt.legend(handles=[scipy, my], labels=['scipy', 'Our implementation'])
plt.title('Absolute error: x=r, p=0.5')
plt.xlabel('x')
plt.ylabel(r'$|I_{0.5}(x, x) - 0.5|$')
plt.tight_layout()
plt.savefig('betainc_xx.pdf')


print('Testing I_p(x,1) = p^x')
xs = np.arange(1, 10000, 10)
ps = np.linspace(0.001, 0.999, len(xs))
X, P = np.meshgrid(xs, ps)
t0 = betainc_scipy(X, 1, P)
t1 = betainc_my(X, 1, P)
truth = np.zeros_like(P, dtype=float)
for ind in np.ndindex(truth.shape):
    p = gmpy2.mpfr(float(P[ind]))
    x = gmpy2.mpfr(float(X[ind]))
    truth[ind] = float(p ** x)
truth = P ** X
diff_scipy = np.abs(t0 - truth) / truth
diff_our = np.abs(t1 - truth) / truth
print('scipy:', f'Max error: {np.nanmax(diff_scipy)}, Mean error: {np.nanmean(diff_scipy)}')
print('We:', f'Max error: {np.nanmax(diff_our)}, Mean error: {np.nanmean(diff_our)}')



xs = np.arange(1, 1000, 10)
rs = np.arange(1, 1000, 10)
X, R = np.meshgrid(xs, rs)
res = dict()

bad = 1
max_bad = 6
for bad in range(1, max_bad + 1):
    print(f'Computing for BAD={bad} (out of {max_bad} BADs in total)...')
    for p in (bad / (bad + 1), 1 / (bad + 1)):
        if p in res:
            break
        print(f'p = {p:.3f}')
        scipy_pred = betainc_scipy(X, R, p)
        our_pred = np.array(betainc_my(X, R, p))
        diff = np.abs(our_pred - scipy_pred) / scipy_pred
        fails_scipy = scipy_pred == 0
        fails_our = our_pred == 0
        diff[fails_scipy | fails_our] = np.nan
        res[p] = (np.nanmean(diff), np.nanmax(diff))

