#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 01:55:09 2022

@author: georgy
"""
from mixalime.utils import get_init_file, openers
from itertools import product
import numpy as np
import dill
from betanegbinfit.distributions import Binom, NB, _TruncatedBinom, _TruncatedNB
from functools import partial
from scipy.stats import power_divergence
from scipy.optimize import minimize
import jax
from collections import defaultdict

class Model():
    def __init__(self, dist: str, mask_size=0):
        self.dist = dist
        self.mask = np.zeros(mask_size, dtype=bool)
        self.grad = jax.jit(jax.jacfwd(self.negloglik, argnums=0))
        self.p = 0.5
    
    @partial(jax.jit, static_argnums=(0, ))
    def fun(self, r: float, data: jax.numpy.ndarray, w:jax.numpy.ndarray,
            mask: jax.numpy.ndarray, left: float, right: float):
        p = self.p
        if self.dist == 'NB':
            logl = _TruncatedNB.logprob(data, r, p, left, right)
        else:
            logl = _TruncatedBinom.logprob(data, r, p, left, right)
        logl *= w
        return jax.numpy.where(mask, 0.0, logl)
    
    @partial(jax.jit, static_argnums=(0, ))
    def negloglik(self, r: jax.numpy.ndarray,  data: jax.numpy.ndarray, w:jax.numpy.ndarray,
                  mask: jax.numpy.ndarray, left: float, right: float):
        return -self.fun(r, data, w, mask, left, right).sum()
    
    def update_mask(self, data,  w):
        mask = self.mask
        n = len(data)
        m = len(mask)
        if n > m:
            mask = np.zeros(n, dtype=bool)
            self.mask = mask
            m = n
        mask[:n] = False
        mask[n:] = True
        c = self.allowed_const
        v = max(0, m - len(data)); data = np.pad(data, (0, v), constant_values=c)
        v = max(0, m - len(w)); w = np.pad(w, (0, v), constant_values=c)
        return data, w
    
    def power_divergence(self, uqs, cnts, probs, lambda_=0, max_tr=None,
                         count_tr=None, normalize=False):
        
        if max_tr is not None:
            inds = uqs < max_tr
            cnts = cnts[inds]
            uqs = uqs[inds]
        if count_tr is not None:
            inds = cnts > count_tr
            cnts = cnts[inds]
            uqs = uqs[inds]
        n = cnts.sum()
        expected = n * probs
        if normalize:
            cnts *= expected.sum() / cnts.sum()
        if lambda_ == 0:
            r = 2 * (cnts * (np.log(cnts) - np.log(expected))).sum()
        else:
            r = 2 / (lambda_ * (lambda_ + 1)) * (cnts * ((cnts / expected) ** lambda_ - 1)).sum()
        return r / n


    def fit(self, data: np.ndarray):
        data, w = data.T
        left = data.min() - 1
        right = data.max()
        n = len(data)
        x0 = [data.mean()] if self.dist == 'NB' else [2 * right - left]
        bounds = None if self.dist == 'NB' else [(right, None)]
        self.allowed_const = left + 1
        data, w = self.update_mask(data, w)
        mask = self.mask
        f = lambda x: partial(self.negloglik, data=data, w=w, mask=mask, left=left, right=right)(x[0])
        grad = lambda x: [partial(self.grad, data=data, w=w, mask=mask, left=left, right=right)(x[0])]
        res = minimize(f, x0=x0, jac=grad, method='SLSQP', bounds=bounds)
        probs = np.exp(self.fun(res.x[0], data=data, w=jax.numpy.ones_like(w), mask=mask, left=left, right=right)[:n])
        data = data[:n]
        w = w[:n]
        pd = self.power_divergence(data, w, probs)
        # probs = np.exp(self.fun(res.x[0], data=data, w=jax.numpy.ones_like(w), mask=mask, left=left, right=right)[:n])
        # freqs = probs * w[:n]
        # freqs /= freqs.sum() / w[:n].sum()
        # pd = power_divergence(w[:n], freqs)[0]
        return res, pd
    
name = 'demo'
init_filename = get_init_file(name)
compression = init_filename.split('.')[-1]
open = openers[compression]
with open(init_filename, 'rb') as f:
    counts = dill.load(f)['counts'][1]
counts = {(x, y): n for x, y, n in counts}
m = 60
def get_samples(counts, x, y, min_n=5, min_r=2):
    res = dict()
    r = 0
    while r <= min_r or sum(res.values()) <= min_n:
        for sx in range(1 + 2 * (r - 1)):
            for t in product([x + sx, x - sx], [y + r, y - r]):
                try:
                    res[t] = counts[t]
                except KeyError:
                    continue
        for sy in range(r + 1):
            for t in product([x + r, x - r], [y + sy, y - sy]):
                try:
                    res[t] = counts[t]
                except KeyError:
                    continue
        
        r += 1
    lt = list()
    for t in sorted(res):
        lt.append(tuple((*t, res[t])))
    return np.array(lt, dtype=int)

def split_circle(circle):
    res_a = defaultdict(int)
    res_b = defaultdict(int)
    for a, b, n in circle:
        res_a[a] += n
        res_b[b] += n
    lt_a = list()
    lt_b = list()
    for res, lt in [(res_a, lt_a), (res_b, lt_b)]:
        for t in sorted(res):
            lt.append((t, res[t]))
    return np.array(lt_a), np.array(lt_b)
max_sz = 0


from tqdm import tqdm
items = list(product(range(5, m + 1), range(5, m + 1)))
for x, y in items:
    t_a, t_b = split_circle(get_samples(counts, x, y))
    max_sz = max(max_sz, max(len(t_a), len(t_b)))
# print(max_sz)
model_nb = Model('NB', max_sz)
model_b = Model('Binom', max_sz)

res = dict()
w = np.zeros((m - 4, m - 4), dtype=float)
w2 = w.copy()
w_a = np.zeros((m - 4, m - 4), dtype=float)
w_b = np.zeros((m - 4, m - 4), dtype=float)
r_a = np.zeros((m - 4, m - 4), dtype=float)
r_b = np.zeros((m - 4, m - 4), dtype=float) 
r_a2 = np.zeros((m - 4, m - 4), dtype=float)
r_b2 = np.zeros((m - 4, m - 4), dtype=float) 
# items = [(19, 57)]
for x, y in tqdm(items):
    t_a, t_b = split_circle(get_samples(counts, x, y))
    r_nb_a, pd_nb_a = model_nb.fit(t_a)
    r_nb_b, pd_nb_b = model_nb.fit(t_b)
    r_b_a, pd_b_a = model_b.fit(t_a)
    r_b_b, pd_b_b = model_b.fit(t_b)
    w[x - 5, y - 5] = (r_nb_a.x[0] - r_nb_b.x[0]) / r_nb_a.x[0]
    w2[x - 5, y - 5] = (r_b_a.x[0] - r_b_b.x[0]) / r_b_a.x[0]
    w_a[x - 5, y - 5] = (pd_nb_a - pd_b_a) #/ pd_nb_a
    w_b[x - 5, y - 5] = (pd_nb_b - pd_b_b) #/ pd_nb_b
    r_a[x-5, y-5] = r_nb_a.x[0]
    r_b[x-5, y-5] = r_nb_b.x[0]
    r_a2[x-5, y-5] = r_b_a.x[0]
    r_b2[x-5, y-5] = r_b_b.x[0]
    # if pd_nb_a > pd_b_a:
    #     print(x, y, pd_nb_a - pd_b_a)
    # if pd_nb_b > pd_b_b:
    #     print(x, y, pd_nb_b - pd_b_b)
    res[(x, y)] = {'NB': ((r_nb_a, r_nb_b), (pd_nb_a, pd_nb_b)), 
                   'Binom': ((r_b_a, r_b_b), (pd_b_a, pd_b_b))
                   }
    
import matplotlib.pyplot as plt
plt.figure(dpi=200, figsize=(8,6))
plt.title(r'$(r_{ref} - r_{alt}) / r_{ref}$')
plt.contourf(range(5, m + 1), range(5, m + 1), w, levels=20);plt.xlabel('ref'); plt.ylabel('alt'); plt.colorbar()
plt.tight_layout()

plt.figure(dpi=200, figsize=(6,6))
plt.title(r'$(r_{ref} - r_{alt}) > 0$')
plt.contourf(range(5, m + 1), range(5, m + 1), np.sign(w2), levels=20);plt.xlabel('ref'); plt.ylabel('alt')
plt.tight_layout()
plt.savefig('ref_sub_alt.pdf', dpi=200)

mx = max(r_a2.max(), r_b2.max())

plt.figure(dpi=200, figsize=(15,6))
plt.subplot(1, 2, 1)
plt.title(r'$r_{ref}$')
plt.contourf(range(5, m + 1), range(5, m + 1), r_a2, levels=20, vmin=0, vmax=mx);plt.xlabel('ref'); plt.ylabel('alt'); plt.colorbar()
plt.subplot(1, 2, 2)
plt.title(r'$r_{alt}$')
plt.contourf(range(5, m + 1), range(5, m + 1), r_b2, levels=20, vmin=0, vmax=mx);plt.xlabel('ref'); plt.colorbar()
plt.tight_layout()
plt.savefig('ref_alt.pdf', dpi=200)


plt.figure(dpi=200, figsize=(14,6))
plt.subplot(1, 2, 1)
plt.title(r'REF: $NB_{rmsea} - Binom_{rmsea} > 0$')
plt.contourf(range(5, m + 1), range(5, m + 1), np.sign(w_a), levels=20, );plt.xlabel('ref'); plt.ylabel('alt');
plt.subplot(1, 2, 2)
plt.title(r'ALT: $NB_{rmsea} - Binom_{rmsea} > 0$')
plt.contourf(range(5, m + 1), range(5, m + 1), np.sign(w_b), levels=20,);plt.xlabel('ref');
plt.tight_layout()
plt.savefig('rmsea_ref_sub_alt.pdf', dpi=200)

mx = max(w_a.max(), w_b.max())
mn = min(w_a.min(), w_b.min())
# w_a[w_a > 10] = np.nan
# w_b[w_b > 10] = np.nan
# w_a[w_a < -10] = np.nan
# w_b[w_b < -10] = np.nan
plt.figure(dpi=200, figsize=(16,6))
plt.subplot(1, 2, 1)
plt.title(r'REF: $(NB_{rmsea} - Binom_{rmsea}) / NB_{rmsea}$')
plt.contourf(range(5, m + 1), range(5, m + 1), w_a, levels=20, );plt.xlabel('ref'); plt.ylabel('alt'); plt.colorbar()
plt.subplot(1, 2, 2)
plt.title(r'ALT: $(NB_{rmsea} - Binom_{rmsea}) / NB_{rmsea}$')
plt.contourf(range(5, m + 1), range(5, m + 1), w_b, levels=20, );plt.xlabel('ref'); plt.colorbar()
plt.savefig('rmsea_ref_sub_alt_rel.pdf', dpi=200)
plt.tight_layout()