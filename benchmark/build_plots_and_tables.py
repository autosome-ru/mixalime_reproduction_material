import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.integrate import simpson
import pandas as pd
import numpy as np
import dill
import os
import mixalime 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import lzma
import dill

main_paper = False

_fontsize = 26 if main_paper else 18
_markersize = 8


def update_style():
    font_files = font_manager.findSystemFonts(fontpaths=os.path.join(os.path.dirname(mixalime.__file__), 'data'))
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.weight'] = "medium"
    plt.rcParams['axes.labelweight'] = 'medium'
    plt.rcParams['figure.titleweight'] = 'medium'
    plt.rcParams['axes.titleweight'] = 'medium'
    plt.rcParams['font.family'] = 'Lato'
    plt.rcParams['font.size'] = _fontsize

update_style()
figures_folder = 'figures/benchmark'
results_folder = 'result'
os.makedirs(os.path.join(results_folder, figures_folder), exist_ok=True)

colors = {'NB': '#e69f00', 'NB_p': '#e69e00', 'BetaNB': '#0072b2', 'BetaNB_p': '#0072a2', 'MCNB': '#f0e442', 'MCNB_p': '#f0e432',
          'binom': '#999999', 'binom_beta': '#009e73'}
markers = {'NB': 'o', 'NB_p': 'x', 'BetaNB': '*', 'BetaNB_p': 's', 'MCNB': 'v', 'MCNB_p': '+', 'binom': '>', 'binom_beta': '<'}
markersizes = defaultdict(lambda: 10)
markersizes['MCNB_p'] = 14
markersizes['BetaNB'] = 14
alphas = {k: 0.8 if k.endswith('_p') else 1.0 for k in colors}
ordering = ['NB', 'NB_p', 'BetaNB', 'BetaNB_p', 'MCNB', 'MCNB_p', 'binom', 'binom_beta']
# markers = ['o', '+', 'v', '*', 'x', '^', 's', '>', '<']

xlabels = {'coverage': r'$ase\_coverage$', 'samples': r'$n\_samples$', 'bias': r'$bias$', 'pmult': r'$ase\_es$',
           'bad': r'BAD', 'kappa': r'$\kappa$',
           'r0noise': r'$r0\_noise$', 'ro': r'$r_0$', 'r0w': r'$\tau$', 'left': 'left', 'nobias': 'i'}
if main_paper:
    xlabels.update({'samples': 'Number of samples', 'bias': 'Bias'})

items = ['nb_nobias_bad1_pmult', 'nb_nobias_bad1_samples', 'bnb_nobias_kappa', 'nb_nobias_bad', 'nb_bad2_bias', 'nb_bias', 'nb_nobias_bad1_coverage',
         'nb_r0noise', 'n_ro']
items = ['nb_nobias_bad1_pmult', 'nb_nobias_bad1_samples', 'bnb_nobias_kappa', 'nb_nobias_bad', 'nb_bad2_bias', 'nb_bias',
         'nb_nobias_bad1_coverage', 'nb_r0noise', 'nb_left', 'nb_mixture_nobias', 'nb_mixture_bias', 'nb_mixture_r0w',
         'nb_mixture_bias_r0w']
items = ['nb_bias', 'nb_mixture_bias_2_r0w', 'nb_mixture_bias_r0_bias', 'bnb_mixture_bias_r0w', 'bnb_mixture_bias_03_kappa', 'nb_r0noise']
groupings = {'A': 'nb_nobias_bad1_samples', 'B': 'nb_nobias_bad1_pmult', 'C': 'nb_bias', 'D': 'nb_nobias_bad',  'E': 'bnb_nobias_kappa',
             'F': 'nb_nobias_bad1_coverage', 'G': 'nb_bad2_bias', 'H': 'nb_r0noise', 'I': 'nb_mixture_bias_2_r0w', 'J': 'bnb_mixture_bias_r0w' ,
             'K': 'nb_mixture_bias_r0_bias', 'L': 'bnb_mixture_bias_03_kappa' }

param_of_interest = {'A': 4, 'B': 0.5, 'C': 1.25, 'D': 2, 'E': 128, 'F': 100, 'G': 1.2, 'H': 4,
                     'I': 0.0, 'J': 0.0, 'K': 1.2, 'L': 128}

# models_group = defaultdict(lambda: ('binom', 'binom_beta', 'NB', 'BetaNB', 'MCNB'))

# models_group = {'A': ('NB', 'binom'), 'B': ('binom', 'NB'), 'C': ('NB', 'binom', 'NB_p'),
#                 'D': ('binom', 'binom_beta', 'NB'), 'E': ('binom_beta', 'BetaNB', 'binom', 'NB', 'MCNB'),
#                 'F': ('NB', 'binom'), 'G': ('NB_p', 'NB', 'binom', 'binom_beta'), 
#                 'H': ('NB', 'NB_p', 'MCNB', 'MCNB_p', 'binom', 'binom_beta'),
#                 'I': ('NB', 'NB_p', 'binom', 'binom_beta', 'BetaNB', 'BetaNB_p'),
#                 'J': ('BetaNB', 'BetaNB_p', 'NB', 'binom', 'binom_beta'),
#                 'K': ('NB_p', 'NB', 'binom', 'binom_beta'),
#                 'L': ('binom_beta', 'BetaNB', 'BetaNB_p')}

keep_p = {'C', 'G'}

def translate_name(n: str):
    if n == 'binom':
        return 'Binomial'
    if n == 'binom_beta':
        return 'Beta-binomial'
    return n

def table(stat, std):
    old_ind = stat.columns[0]
    if old_ind in ('samples', 'coverage', 'left', 'kappa'):
        fmt_str = '{:.0f}'
    elif old_ind in ('pmult', 'bad', 'r0noise', 'r0w'):
        fmt_str = '{:.1f}'
    else:
        fmt_str = '{:.4f}'
    new_ind = xlabels[old_ind]
    stat = stat.rename(columns={old_ind: new_ind})
    std = std.rename(columns={old_ind: new_ind})
    for t in (stat, std):
        t[new_ind] = [fmt_str.format(float(v)) for v in t[new_ind]]
        ncols = list()
        for c in t.columns[1:]:
            t[c] = ['{:.4f}'.format(float(v)) for v in t[c]]
            ncols.append(c.replace('_', r'\_'))
        t.columns = [new_ind] + ncols
    stat = stat.set_index(new_ind)
    std = std.set_index(new_ind)
    res = stat + r' $\pm$ ' + std
    return res

base_table = '''
\\begin{{table}}[H]
\\centering
\\begin{{adjustbox}}{{max width=\\linewidth}}
{body}
\end{{adjustbox}}
\\caption{{Group {group}, {metric}, values after $\\pm$ are standard deviations.}}
\\end{{table}}
'''

if main_paper:
    row_figure = '''
\\raisebox{{6.5em}}
    				\\begin{{adjustbox}}{{max width=0.925\\textwidth}}
    					\\includegraphics{{figures/benchmark/{group}}}
    				\\end{{adjustbox}}
    '''
else:
    row_figure = '''
\\raisebox{{6.5em}}{{\\Huge {group}}} &
    				\\begin{{adjustbox}}{{max width=0.925\\textwidth}}
    					\\includegraphics{{figures/benchmark/{group}}}
    				\\end{{adjustbox}}
    '''

tabulars = str()
figures = list()
figures_2 = list()

# items = ['n_ro']
for grouping, name in groupings.items():
    if main_paper and grouping not in ('A', 'D', 'K'):
        continue
    power_mean = pd.read_csv(os.path.join(results_folder, f'recall_mean_{name}.tsv'), sep='\t')
    power_std = pd.read_csv(os.path.join(results_folder, f'recall_std_{name}.tsv'), sep='\t')
    spec_mean = pd.read_csv(os.path.join(results_folder, f'specificity_mean_{name}.tsv'), sep='\t')
    spec_std = pd.read_csv(os.path.join(results_folder, f'specificity_std_{name}.tsv'), sep='\t')
    auc_mean = pd.read_csv(os.path.join(results_folder, f'auc_mean_{name}.tsv'), sep='\t')
    auc_std = pd.read_csv(os.path.join(results_folder, f'auc_std_{name}.tsv'), sep='\t')
    cols = list(power_mean.columns[:1]) + sorted(power_mean.columns[1:], key=ordering.index)
    power_mean = power_mean[cols]
    power_std = power_std
    spec_mean = spec_mean[cols]
    spec_sttd = spec_std[cols]
    auc_mean = auc_mean[cols]
    auc_std = auc_std[cols]
    poi = param_of_interest[grouping]
    with lzma.open(os.path.join(results_folder, f'prc_{name}.lzma'), 'rb') as f:
        prc = dill.load(f)[poi]
    with lzma.open(os.path.join(results_folder, f'auc_{name}.lzma'), 'rb') as f:
        auc = dill.load(f)[poi]
    
    x = power_mean.iloc[:, 0]
    models = cols[1:]
    if main_paper:
        models_to_plot =  ('binom', 'binom_beta', 'NB', 'BetaNB', 'MCNB')
    else:
        if grouping not in keep_p:
            models_to_plot = list(filter(lambda x: not x.endswith('_p'), models))
        else:
            models_to_plot = list(filter(lambda x: x != 'MCNB_p', models))
        # models_to_plot = models_group.get(grouping, models)
        # models_to_plot = [m for m in models if not m.endswith('_p')]
    plt.figure(dpi=200, figsize=(20, 7))
    plt.subplot(1, 3, 1)
    for i, model in enumerate(models_to_plot):
        plt.plot(x, auc_mean[model], color=colors[model], marker=markers[model], markersize=markersizes[model], alpha=alphas[model])
    if not main_paper or grouping == 'A':
        plt.legend(labels=list(map(translate_name, models_to_plot)))
    plt.axvline(x=poi, linestyle='dashed', color='k')
    plt.xlabel(xlabels[power_mean.columns[0]])
    plt.ylabel('AUPRC')
    if not main_paper:
        plt.title(f'AUPRC at different {xlabels[power_mean.columns[0]]}')
    xticks = sorted(list(plt.xticks()[0]))[1:-1]
    if main_paper:
        plt.xticks(xticks + [poi])
    plt.subplot(1, 3, 2)
    for i, model in enumerate(models_to_plot):
        prc_x, prc_y = prc[model]
        plt.plot(prc_x, prc_y, color=colors[model], alpha=alphas[model])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    if not main_paper:
        plt.title(f'PR curve at {xlabels[power_mean.columns[0]]} = {poi}')
    
    plt.subplot(1, 3, 3)
    j = list(x.values).index(poi)
    # std = 
    auc = pd.DataFrame(auc)[list(models_to_plot)]
    std = auc.std().values
    means = auc.mean().values
    lq = means - std
    hq = means + std
    parts = plt.violinplot(auc, showmeans=False, showextrema=False)
    inds = np.array(list(range(1, len(auc.columns) + 1)))
    for i, pc in enumerate(parts['bodies']):
        pc.set_color(colors[auc.columns[i]])
        pc.set_alpha(1)
    # parts['cbars'].set_color('black')
    # parts['cmins'].set_color('black')
    # parts['cmaxes'].set_color('black')
    # parts['cmeans'].set_color('black')
    plt.vlines(inds, lq, hq, color='k', lw=4)
    plt.hlines(lq,  inds - 0.15, inds + 0.15, color='k', lw=4)
    plt.hlines(hq,  inds - 0.15, inds + 0.15, color='k', lw=4)
    plt.scatter(inds, means, marker='o', color='white', s=30, zorder=3)
    labels = ['Beta-\nbinomial' if '-' in m else m for m in map(translate_name, auc.columns)]
    plt.xticks(inds, labels=labels, rotation=60)
    # plt.xlabel('Model')
    # for i, model in enumerate(models_to_plot):
    #     plt.violinplot(auc[model], model,)
        # plt.bar(model, auc_mean[model].values[j], color=colors[model])
        # plt.errorbar(model, auc_mean[model].values[j], 2 * auc_std[model].values[j], 
        #              color='Black', elinewidth=2, capthick=10, errorevery=1, ms=4, capsize = 2)
        # if color_significant:
        #     for name, pval, pval_fdr, v in zip(groups, pvals, pvals_fdr, values):
        #         if pval < 0.05:
        #             pval_bar_ = plt.bar(name, v, width=0.9, color='orange')
        #             if pval_fdr < 0.05:
        #                 fdr_bar = plt.bar(name, v, width=0.9, color='green')
        #             else:
        #                 pval_bar = pval_bar_
        # if variance is not None:
        #     plt.errorbar(groups, values, np.array(variance) ** 0.5,  fmt='.', color='Black', elinewidth=2, capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
        # plt.xticks(rotation=45)
    plt.ylabel('AUPRC')
    if not main_paper:
        plt.title(f'AUPRC at {xlabels[power_mean.columns[0]]} = {poi}')
    plt.tight_layout()
    path = os.path.join(results_folder, figures_folder, f'{grouping}.pdf')
    plt.savefig(path)
    os.popen(f'pdfcrop {path} {path}').read()
    if grouping in keep_p and not main_paper:
        r = row_figure
        figures_2.append(r)
    else:
        r = row_figure.format(group=grouping)
        figures.append(r)
    tabulars += f'\\subsubsection*{{Group {grouping}}}'
    for mean, std, metric in ((auc_mean, auc_std, 'PR AUC'), (power_mean, power_std, 'Sensitivity'),
                              (spec_mean, spec_std, 'Specificity')):
        tabulars += base_table.format(body=table(mean, std).to_latex(), 
                                      group=grouping, metric=metric)
rows = r'\\'.join(figures)
if main_paper:
    figures = f'''
    \\begin{{longtable}}{{c}}
    {rows}
    \\end{{longtable}}
    '''
else:
    figures = f'''
    \\begin{{longtable}}{{cc}}
    {rows}
    \\end{{longtable}}
    \\captionof{{figure}}{{PR AUC, sensitivity and specificity metrics as evaluated for various datasets. We evaluated all models present in \\textbf{{MIXALIME}}, 
                           but for some figures only models whose performance we found relevant to the particular dataset are shown. For a complete comparison, see tables in
                           Appendix~\\ref{{app:benchmark_tables}}. }}
    \\label{{fig:benchmark}}
    '''
with open(os.path.join(results_folder, 'benchmark_figures.tex'), 'w') as f:
    f.write(figures)
    

rows = r'\\'.join(figures_2)
figures = f'''
\\begin{{longtable}}{{cc}}
{rows}
\\end{{longtable}}
\\captionof{{figure}}{{Groups C, K togher with conventional MIXALIME models where $p$ was estimated and $b$ is fixed to $1$. }}
\\label{{fig:benchmark_bias}}
'''
if not main_paper:
    with open(os.path.join(results_folder, 'benchmark_figures_2.tex'), 'w') as f:
        f.write(figures)

with open(os.path.join(results_folder, 'benchmark_tabulars.tex'), 'w') as f:
    f.write(tabulars)
    
    