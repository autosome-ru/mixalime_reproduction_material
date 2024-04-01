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
_fontsize = 16
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
results_folder = 'results_final'
os.makedirs(os.path.join(results_folder, figures_folder), exist_ok=True)

colors = {'NB': 'blue', 'NB_p': 'darkblue', 'BetaNB': 'red', 'BetaNB_p': 'darkred', 'MCNB': 'violet', 'MCNB_p': 'darkviolet', 'binom': 'grey', 'binom_beta': 'black'}
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

items = ['nb_nobias_bad1_pmult', 'nb_nobias_bad1_samples', 'bnb_nobias_kappa', 'nb_nobias_bad', 'nb_bad2_bias', 'nb_bias', 'nb_nobias_bad1_coverage',
         'nb_r0noise', 'n_ro']
items = ['nb_nobias_bad1_pmult', 'nb_nobias_bad1_samples', 'bnb_nobias_kappa', 'nb_nobias_bad', 'nb_bad2_bias', 'nb_bias',
         'nb_nobias_bad1_coverage', 'nb_r0noise', 'nb_left', 'nb_mixture_nobias', 'nb_mixture_bias', 'nb_mixture_r0w',
         'nb_mixture_bias_r0w']
items = ['nb_bias', 'nb_mixture_bias_2_r0w', 'nb_mixture_bias_r0_bias', 'bnb_mixture_bias_r0w', 'bnb_mixture_bias_03_kappa', 'nb_r0noise']
groupings = {'A': 'nb_nobias_bad1_samples', 'B': 'nb_nobias_bad1_pmult', 'C': 'nb_bias', 'D': 'nb_nobias_bad',  'E': 'bnb_nobias_kappa',
             'F': 'nb_nobias_bad1_coverage', 'G': 'nb_bad2_bias', 'H': 'nb_r0noise', 'I': 'nb_mixture_bias_2_r0w', 'J': 'bnb_mixture_bias_r0w' ,
             'K': 'nb_mixture_bias_r0_bias', 'L': 'bnb_mixture_bias_03_kappa' }

models_group = {'A': ('NB', 'binom'), 'B': ('binom', 'NB'), 'C': ('NB', 'binom', 'NB_p'),
                'D': ('binom', 'binom_beta', 'NB'), 'E': ('binom_beta', 'BetaNB', 'binom', 'NB', 'MCNB'),
                'F': ('NB', 'binom'), 'G': ('NB_p', 'NB', 'binom', 'binom_beta'), 
                'H': ('NB', 'NB_p', 'MCNB', 'MCNB_p', 'binom', 'binom_beta'),
                'I': ('NB', 'NB_p', 'binom', 'binom_beta', 'BetaNB', 'BetaNB_p'),
                'J': ('BetaNB', 'BetaNB_p', 'NB', 'binom', 'binom_beta'),
                'K': ('NB_p', 'NB', 'binom', 'binom_beta'),
                'L': ('binom_beta', 'BetaNB', 'BetaNB_p')}

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

row_figure = '''
\\raisebox{{6.5em}}{{\\Huge {group}}} &
    				\\begin{{adjustbox}}{{max width=0.925\\textwidth}}
    					\\includegraphics{{figures/benchmark/{group}}}
    				\\end{{adjustbox}}\\
'''
row_figure_f = '''
\\raisebox{{6.5em}}{{\\Huge {group}}} &
    				\\begin{{adjustbox}}{{max width=0.925\\textwidth}}
    					\\includegraphics{{figures/benchmark/{group}_full}}
    				\\end{{adjustbox}}\\
'''

tabulars = str()
figures = list()
figures_full = list()

# items = ['n_ro']
for grouping, name in groupings.items():
        
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
    
    x = power_mean.iloc[:, 0]
    models = cols[1:]
    models_to_plot = models_group.get(grouping, models)
    for models_to_plot, full in ((models_to_plot, False), (models, True)):
        plt.figure(dpi=200, figsize=(20, 6))
        plt.subplot(1, 3, 1)
        for i, model in enumerate(models_to_plot):
            plt.plot(x, auc_mean[model], color=colors[model], marker=markers[model], markersize=markersizes[model], alpha=alphas[model])
        plt.legend(labels=models_to_plot)
        plt.xlabel(xlabels[power_mean.columns[0]])
        plt.ylabel('PR AUC')
        plt.subplot(1, 3, 2)
        for i, model in enumerate(models_to_plot):
            plt.plot(x, power_mean[model], color=colors[model], marker=markers[model], markersize=markersizes[model], alpha=alphas[model])
        plt.ylabel('Sensitivity')
        plt.xlabel(xlabels[power_mean.columns[0]])
        
        plt.subplot(1, 3, 3)
        for i, model in enumerate(models_to_plot):
            ax = plt.plot(x, spec_mean[model], color=colors[model], marker=markers[model], markersize=markersizes[model], alpha=alphas[model])[0]
        plt.xlabel(xlabels[power_mean.columns[0]])
        plt.ylabel('Specificity')
        
        plt.tight_layout()
        if full:
            plt.savefig(os.path.join(results_folder, figures_folder, f'{grouping}_full.pdf'))
            figures_full.append(row_figure_f.format(group=grouping))
        else:
            plt.savefig(os.path.join(results_folder, figures_folder, f'{grouping}.pdf'))
            figures.append(row_figure.format(group=grouping))
    tabulars += f'\\subsubsection*{{Group {grouping}}}'
    for mean, std, metric in ((auc_mean, auc_std, 'PR AUC'), (power_mean, power_std, 'Sensitivity'),
                              (spec_mean, spec_std, 'Specificity')):
        tabulars += base_table.format(body=table(mean, std).to_latex(), 
                                      group=grouping, metric=metric)
rows = r'\\'.join(figures)
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
    

rows = r'\\'.join(figures_full)
figures = f'''
\\begin{{longtable}}{{cc}}
{rows}
\\end{{longtable}}
\\captionof{{figure}}{{Uncensored 18+ version of the Figure~\\ref{{fig:benchmark}}. }}
\\label{{fig:benchmark_full}}
'''
with open(os.path.join(results_folder, 'benchmark_figures_full.tex'), 'w') as f:
    f.write(figures)

with open(os.path.join(results_folder, 'benchmark_tabulars.tex'), 'w') as f:
    f.write(tabulars)
    
    