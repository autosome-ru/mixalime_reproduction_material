import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

res_folder = 'result_29'
seeds = list(filter(lambda x: os.path.isdir(os.path.join(res_folder, x)), os.listdir(res_folder)))

def mean_and_std(d: defaultdict, name: str):
    mean = defaultdict(lambda: defaultdict(list))
    std = defaultdict(lambda: defaultdict(list))
    for param, models in d.items():
        for model in models:
            mean[param][model] = np.mean(d[param][model])
            std[param][model] = np.std(d[param][model])
    mean = pd.DataFrame.from_dict(mean, orient='index')
    mean.index.name = name
    std = pd.DataFrame.from_dict(std, orient='index')
    std.index.name = name
    return mean.sort_index(), std.sort_index()


items = ['nb_bias', 'nb_mixture_bias_2_r0w', 'nb_mixture_bias_r0_bias', 'bnb_mixture_bias_r0w', 'bnb_mixture_bias_03_kappa', 'nb_r0noise', 
         'nb_nobias_bad1_pmult', 'nb_nobias_bad1_samples', 'nb_nobias_bad1_coverage', 'nb_nobias_bad', 'nb_bad2_bias', 'bnb_nobias_kappa']

def doit(name: str):
    recall, specificity = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    auc = defaultdict(lambda: defaultdict(list))
    for seed in tqdm(seeds):
        result_folder = os.path.join(res_folder, seed)
        folders = filter(lambda x: os.path.isdir(os.path.join(result_folder, x)) and x.startswith(name+'_'), os.listdir(result_folder))
        for folder in folders:
            try:
                param = float(folder[len(name) + 1:])
            except:
                continue
            folder = os.path.join(result_folder, folder)
            for model in os.listdir(folder):
                subfolder = os.path.join(folder, model)
                if not os.path.isdir(subfolder):
                    continue
                pval = os.path.join(subfolder, 'pvalues', 'pvals.tsv')
                pval = pd.read_csv(pval, sep='\t').sort_values('fdr_comb_pval')
                pvals = pval.fdr_comb_pval.values.astype(float)
                ases = np.array([n.startswith('ase') for n in pval['#chr']], dtype=bool)
                pval['ase'] = ases
                c_ases = np.cumsum(ases)
                total_ase = c_ases[-1]
                _, inds = np.unique(pvals, return_index=True)
                paucs = [1.0]
                raucs = [0.0]
                for i in inds[1:-1]:
                    correct = c_ases[i - 1]
                    paucs.append(correct / i)
                    raucs.append(correct / total_ase)
                paucs.append(total_ase / len(pvals))
                raucs.append(1.0)
                auc[param][model].append(np.trapz(paucs, raucs))
                pval = pval[pval.fdr_comb_pval < 0.05]
                correct = pval.ase.sum()
                specificity[param][model].append(1.0 - (len(pval) - correct) / (len(pvals) - total_ase))
                recall[param][model].append(correct / total_ase)
    print(name)
    param_name = name.split('_')[-1]
    recall_mean, recall_std = mean_and_std(recall, param_name)
    specificity_mean, specificity_std = mean_and_std(specificity, param_name)
    auc_mean, auc_std = mean_and_std(auc, param_name)
    recall_mean.to_csv(os.path.join(res_folder, f'recall_mean_{name}.tsv'), sep='\t')
    recall_std.to_csv(os.path.join(res_folder, f'recall_std_{name}.tsv'), sep='\t')
    specificity_mean.to_csv(os.path.join(res_folder, f'specificity_mean_{name}.tsv'), sep='\t')
    specificity_std.to_csv(os.path.join(res_folder, f'specificity_std_{name}.tsv'), sep='\t')
    auc_mean.to_csv(os.path.join(res_folder, f'auc_mean_{name}.tsv'), sep='\t')
    auc_std.to_csv(os.path.join(res_folder, f'auc_std_{name}.tsv'), sep='\t')
                
                
with mp.Pool(min(mp.cpu_count(), len(items))) as p:
    list(p.map(doit, items))