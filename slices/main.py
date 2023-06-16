from mixalime.utils import get_init_file, openers
from mixalime.plot import plot_heatmap, plot_histogram, _ref, _alt
from betanegbinfit import distributions as dists
from matplotlib.ticker import FormatStrFormatter
from itertools import product
import numpy as np
import dill
import matplotlib.pyplot as plt

name = 'demo'
init_filename = get_init_file(name)
compression = init_filename.split('.')[-1]

def pdf1(x, y):
    return np.exp(dists.LeftTruncatedNB.logprob(x, y - 1, 0.5, 5))

def pdf2(x, y):
    return np.exp(dists.Binom.logprob(x, y, 0.5))


open = openers[compression]
with open(init_filename, 'rb') as f:
    counts = dill.load(f)['counts'][1]

n = 60
n_slice = 45
h_slice = 15

    
# plt.figure(figsize=(13.3, 6), dpi=200)
# plt.subplot(1, 2, 1)
# plot_heatmap(counts, n, n_slice)
# plt.xlim([5, n])
# plt.ylim([5, n])
# plt.subplot(1, 2, 2)
# ax1 = plot_histogram(counts, n, n_slice, pdf2, slc_sum=True, c='k')
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# plt.tight_layout()
# plt.savefig('sum_slice.pdf')


tn = counts[counts[:, :-1].sum(axis=1) == n_slice, :-1].max()
plt.figure(figsize=(20, 6), dpi=200)
plt.subplot(1, 3, 1)
plot_heatmap(counts, n, (h_slice, h_slice))
plt.xlim([5, n])
plt.ylim([5, n])
plt.subplot(1, 3, 2)
ax1 = plot_histogram(counts, tn, h_slice, pdf1, c=_ref)
plt.subplot(1, 3, 3)
ax2 = plot_histogram(counts, tn, h_slice, pdf1, c=_alt, s=1)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.tight_layout()
plt.savefig('hor_slice.pdf')


tn = 40#counts[(counts[:, 0] == n_slice) | (counts[:, 1] == n_slice), :-1].max()
plt.figure(figsize=(20, 6), dpi=200)
plt.subplot(1, 3, 1)
plot_heatmap(counts, n, n_slice)
plt.xlim([5, n])
plt.ylim([5, n])
plt.subplot(1, 3, 2)
ax1 = plot_histogram(counts, tn, n_slice, pdf2, c='k', slc_sum=True,)
plt.subplot(1, 3, 3)
tn = 35
ax2 = plot_histogram(counts, tn, n_slice, pdf2, c='k', s=1, slc_sum=True,)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.tight_layout()
plt.savefig('sum_slice.pdf')

