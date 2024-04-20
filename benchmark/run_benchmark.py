import numpy as np
from scipy.stats import beta, uniform, binom, expon, gamma
from scipy.special import betainc, beta as betafun
from scipy.optimize import minimize_scalar
from betanegbinfit.distributions import LeftTruncatedNB
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import pandas as pd
import shutil
import os

local_ml = True
data_folder = 'datasets'
res_folder = 'result_window' if local_ml else 'result'

def sample_beta(size: int, a: float, b: float, mode='mu', right_b=1-1e-12):
    if mode == 'mu':
        mu = a
        k = b
        if not np.isfinite(k):
            return mu
        a = mu * k
        b = (1 - mu) * k
    mean = beta.mean(a=a, b=b)
    binc = lambda a, b, x: betainc(a, b, x) * betafun(a, b)
    tr_mean = lambda left: (binc(1 + a, b, left) - binc(1 + a, b, right_b)) / (binc(a, b, left) - binc(a, b, right_b))
    left_b = minimize_scalar(lambda left: np.sum((mean - tr_mean(left)) ** 2), bounds=(0.0, 1.0), tol=1e-16).x
    left_b = beta.cdf(left_b, a=a, b=b)
    right_b = beta.cdf(right_b, a=a, b=b)
    p = beta.ppf(uniform.rvs(size=size) * (right_b - left_b) + left_b, a=a, b=b)
    return p
    

def sample(size: int, r0: int, p0: float, p1: float, k=float('inf'), left=0, single_p=False, right_b=1-1e-12, uqs=True,
           sample_trunc=True, k_old=False, right=float('inf')):
    beta_size = size if not single_p else None
    if not np.all(np.isfinite(k)) or k_old:
        am = p0 * p1 / (1 - (1 - p1) * p0)
        ac = p0 * (1 - p1)
        bm = bc = k
        mode = 'mu'
    else:
        a1 = p1 * k
        a2 = (1 - p1) * k
        a0 = (1 - p0) / p0 * k
        am = a1
        bm = a0
        ac = a2
        bc = a0 + a1
        mode = 'ab'

    pm = sample_beta(beta_size, am, bm, mode=mode, right_b=right_b)
    pc = sample_beta(beta_size, ac, bc, mode=mode, right_b=right_b)
    tleft = -1 if sample_trunc else left
    x = LeftTruncatedNB.sample(r0, pm, left=tleft, size=size)
    y = LeftTruncatedNB.sample(r0 + x , pc, left=tleft, size=size)
    ind = (x <= left) | (y <= left) | (x >= right) | (y >= right)
    while ind.sum():
        x[ind] = LeftTruncatedNB.sample(r0[ind] if np.iterable(r0) else r0, 
                                        pm[ind] if np.iterable(pm) else pm, left=tleft, size=ind.sum())
        y[ind] = LeftTruncatedNB.sample(r0[ind] + x[ind] if np.iterable(r0) else r0 + x[ind], 
                                        pc[ind] if np.iterable(pc) else pc, left=tleft, size=ind.sum())
        ind = (x <= left) | (y <= left)  | (x >= right) | (y >= right)
    z = np.concatenate([x.reshape(1,-1), y.reshape(1, -1)], axis=0).T.astype(int)
    if not uqs:
        return z
    z, c = np.unique(z, return_counts=True, axis=0)
    z = np.concatenate((z, c.reshape(-1,1)),axis=1)
    return z

def sample_n_cond(size: int, r0: int, p0: float, p1: float, n: int, k=float('inf'), left=0, single_p=False, right_b=1-1e-12, uqs=True,
                  sample_trunc=True, k_old=False, right=float('inf')):
    beta_size = size if not single_p else None
    if not np.isfinite(k) or k_old:
        a = p1
        b = k
        mode = 'mu'
    else:
        a1 = p1 * k
        a2 = (1 - p1) * k
        a = a1
        b = a2
        mode = 'ab'
    pc = sample_beta(beta_size, a, b, mode=mode, right_b=right_b)
    x = binom.rvs(p=pc, n=n, size=size)
    y = n - x
    ind = (x <= left) | (y <= left) | (x >= right) | (y >= right)
    while ind.sum():
        x[ind] = binom.rvs(p=pc[ind] if np.iterable(pc) else pc, n=n, size=ind.sum())
        y[ind] = n - x[ind]
        ind = (x <= left) | (y <= left)  | (x >= right) | (y >= right)
    z = np.concatenate([x.reshape(1,-1), y.reshape(1, -1)], axis=0).T.astype(int)
    if not uqs:
        return z
    z, c = np.unique(z, return_counts=True, axis=0)
    z = np.concatenate((z, c.reshape(-1,1)),axis=1)
    return z


def logit_update(p, a: float):
    return p / (p + (1 - p) * 2 ** -a)

def ratio_update(bad: float, a: float):
    return bad * a / (1 + bad * a)

def convert_iterables(type, *args):
    res = [type(item) if np.iterable(item) else item for item in args]
    return res

def sample_data(folder: str, prefix: str, num_snps: int, num_samples: int, left_truncation: int, kappa: float,
                bad: int, bias: float, p0: float, r0: float, w_frac=0.5, p_mult=0, seed=0, right_truncation=float('inf'),
                n_cond=None, r0_noise=0):
    np.random.seed(seed)
    if kappa is None:
        kappa = float('inf')
    r0, bias, p0, kappa = convert_iterables(list, r0, bias, p0, kappa)
    for item in (r0, bias, p0, kappa):
        if np.iterable(item):
            m = len(item)
            step = num_samples // m
            tmp = np.zeros(num_samples)
            for i in range(m):
                tmp[i * step: (i + 1) * step] = item[i]
            item.clear()
            item.extend(tmp)
    r0, bias, p0, kappa = convert_iterables(np.array, r0, bias, p0, kappa)
    p0 = 1 - p0
    beta_trunc = 1e-5
    bam_files = [list() for _ in range(num_samples)]
    tr0 = r0
    for n in tqdm(list(range(num_snps))):
        if r0_noise >  0:
            r0 = gamma.rvs(a=tr0 ** 2 / r0_noise, scale= r0_noise / tr0) + 0.01
        p = ratio_update(bad, bias)
        p = logit_update(p, np.random.choice([-1, 1]) * p_mult)
        if bad > 1:
            if np.random.uniform() <= w_frac:
                p = p
            else:
                p = 1 - p
            # p = np.random.choice([p, 1.0 - p], p=[w_frac, 1.0 - w_frac])
        if not n_cond:
            z = sample(size=num_samples, r0=r0, p0=p0, p1=p, k=kappa, left=left_truncation, single_p=True, uqs=False,
                       right_b=1.0 - beta_trunc, right=right_truncation)
        else:
            z = sample_n_cond(size=num_samples, r0=r0, p0=p0, p1=p, k=kappa, n=n_cond, left=left_truncation, single_p=True, uqs=False,
                       right_b=1.0 - beta_trunc, right=right_truncation)
        for i in range(num_samples):
            bam_files[i].append(tuple(z[i]))
    chr_name = prefix
    pos = list(range(1, num_snps + 1))
    ref = 'G'
    alt = 'T'
    bam_files = map(pd.DataFrame, bam_files)
    os.makedirs(folder, exist_ok=True)
    sample_bias = list()
    filenames = list()
    for i, file in enumerate(bam_files):
        file.columns = ['ref_count', 'alt_count']
        t = file.values
        t = t[t[:,0] != t[:,1]]
        sample_bias.append(np.mean(t[:, 0] > t[:, 1]))
        file['#chr'] = chr_name
        file['ref'] = ref
        file['alt'] = alt
        file['start'] = pos
        file['bad'] = bad
        file = file[['#chr', 'start', 'ref', 'alt', 'ref_count', 'alt_count', 'bad']]
        # filename = os.path.join(folder, f'{prefix}_{i+1}.tsv')
        filename = f'{prefix}_{i + 1}.tsv'
        file.to_csv(os.path.join(folder, filename), sep='\t', index=None)
        filenames.append(filename)
    return filenames, sample_bias

def mixalime_routine(name: str, dataset_folder, output_folder: str, left=0, max_count=1500, only_fits=False, external_fit=False, est_p=True,
                     models_mixalime=('NB', 'MCNB', 'BetaNB'), models_binom=('', '--beta')):
    model_folder = output_folder
    cmd = f'mixalime create {model_folder}/{name} {dataset_folder}/{name}.txt --min-cnt {left+1}'
    lines = [f'echo "{dataset_folder}/{name}"']
    if not only_fits:
        lines.append(cmd)
    if est_p:
        models_mixalime = list(models_mixalime) + [f'{m} --fix-params "b=1" --estimate-p' for m in models_mixalime]# if m != 'BetaNB']
    for i, model in enumerate(models_mixalime): 
        project = f'{name}_{i}' if only_fits else name
        lines.append(f'echo "{dataset_folder}/{model}"')
        if only_fits:
            cmd = f'mixalime create {model_folder}/{project} {dataset_folder}/{name}.txt --min-cnt {left+1}'
            lines.append(cmd)
        if not external_fit:
            window_size = '--window-size 100000000' if not local_ml else str()
            cmd = f'mixalime fit {model_folder}/{project} {model} --max-count {max_count}' + window_size
            lines.append(cmd)
        if not only_fits:
            cmd = f'mixalime test {model_folder}/{project}'
            if external_fit:
                t = '_'.join(name.split('_')[:-2])
                cmd += f' --fit {model_folder}/{t}_{i}.fit.lzma'
            lines.append(cmd)
            cmd = f'mixalime combine {model_folder}/{project}'
            lines.append(cmd)
            if est_p:
                model = model.replace(' --fix-params "b=1" --estimate-p --r-transform none', '_pt')
                model = model.replace(' --fix-params "b=1" --estimate-p', '_p')
            out = os.path.join(output_folder, name, model.replace(' ', '').replace('--', '_'))
            if not external_fit:
                cmd = f'mixalime plot all {model_folder}/{project} {out}'
                lines.append(cmd)
            cmd = f'mixalime export all {model_folder}/{project} {out}'
            lines.append(cmd)
    if only_fits:
        return lines
    for model in models_binom:
        lines.append(f'echo "binom {model}"')
        cmd = f'mixalime test_binom {model_folder}/{name} {model} --estimate-p --max-cover {max_count * 2}'
        lines.append(cmd)
        cmd = f'mixalime combine {model_folder}/{name}'
        lines.append(cmd)
        out = os.path.join(output_folder, name, 'binom' + model.replace(' ', '').replace('--', '_'))
        cmd = f'mixalime export all {model_folder}/{name} {out}'
        lines.append(cmd)
    return lines



def gen(seed: int):
    results_folder = os.path.join(res_folder, str(seed))
    datasets_folder = os.path.join(data_folder, str(seed))
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(datasets_folder, exist_ok=True)
    script = list()
    ############## 
    ###########    BASE. BAD = 1, NB, No bias
    name = 'nb_nobias_bad1'
    folder = os.path.join(datasets_folder, name)
    # shutil.rmtree(folder, ignore_errors=True)
    num_snps = 10000
    num_ases = 1000
    num_samples = 10
    r0 = 1
    p0 = 1e-2
    w_frac = 0.5
    bias = 1
    bad = 1
    left = 0
    kappa = None
    print('Sampling BASE, NB, No bias, BAD = 1')
    filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                              left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed)
    
    
    # ###
    # ### 1000 ASEs, varying p_mult
    num_ases = 1000
    for i, p_mult in enumerate(np.linspace(0, 1, num=11)[1:]):
        print(f'Sampling ASEs, p_mult = {p_mult}')
        filenames_ase, sample_bias = sample_data(folder, prefix=f'asepm{i}', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed)
        filenames = filenames_base + filenames_ase
        subname = f'{name}_pmult_{p_mult}'
        out_filename = f'{subname}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(subname, datasets_folder, results_folder, left=left)
    
    ###
    ### Varying number of samples
    p_mult = 0.4
    num_samples = 64
    print(f'Sampling ASEs, p_mult = {p_mult} (for varying number of samples)')
    filenames_ase, sample_bias = sample_data(folder, prefix='asesamp', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                              left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed)
    
    for n in [1, 2, 4, 8, 16, 25, 32, 48, 64]:
        filenames = filenames_base + filenames_ase[:n]
        subname = f'{name}_samples_{n}'
        out_filename = f'{subname}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(subname, datasets_folder, results_folder, left=left)
    
    # ###
    # ### ASEs of particular coverages
    p_mult = 0.4
    num_samples = 10
    num_ases = 1000
    print(f'Sampling ASEs, p_mult = {p_mult} (for varying number of samples)')
    
    out_filename = f'{name}.txt'
    with open(os.path.join(datasets_folder, out_filename), 'w') as f:
        f.write('\n'.join([os.path.join(name, f) for f in filenames_base]))
    
    script += mixalime_routine(name, datasets_folder, results_folder, left=left, only_fits=True)
    
    for n in [20, 30, 40, 50, 60, 70, 80, 100, 160, 200]:
        filenames_ase, sample_bias = sample_data(folder, prefix=f'asecov_{n}', num_snps=num_ases, num_samples=num_samples, bad=bad, n_cond=n,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed)
        filenames = filenames_base + filenames_ase
        subname = f'{name}_coverage_{n}'
        out_filename = f'{subname}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(subname, datasets_folder, results_folder, left=left, external_fit=True)
        
        
    
    # # # # ############## 
    # # # # ###########    Varying BADs
    # print('Sampling varying BADs...')
    num_samples = 10
    for bad in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
        print(f'Sampling BAD={bad}...')
        name = f'nb_nobias_bad_{bad}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left)
    
    # # # ############## 
    # ###########    Varying refbias
    bad = 1
    p_mult = 0.4
    for bias in np.linspace(1.0, 1.5, num=9):
        print(f'Sampling bias={bias}...')
        name = f'nb_bias_{bias}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left)
        print(seed, name)
    # ############## 
    # ##########    Varying refbias for BAD=2
    bad = 2
    for bias in [1.0, 1.1, 1.2, 1.3, 1.5]:
        print(f'Sampling bias={bias}...')
        name = f'nb_bad2_bias_{bias}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left, est_p=True)
    
    # # # ############# 
    # # ##########    Varying kappas
    bias = 1.0
    p_mult = 0.4
    num_ases = 1000
    bad = 1
    num_samples = 20
    for kappa in [10, 16, 32, 64, 128, 256, 512]:
        print(f'Sampling kappa={kappa}...')
        name = f'bnb_nobias_kappa_{kappa}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0,
                                                  right_truncation=10000, seed=seed)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult,
                                                  right_truncation=10000, seed=seed)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left)
        
    #  # # ############## 
    #  # # ###########    Varying r0 noise
    print('Sampling varying r0 noise...')
    num_samples = 10
    bad = 1
    kappa = None
    p_mult = 0.4
    r0 = 2.5
    for r0_noise in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        print(f'Sampling r0_noise={r0_noise}...')
        name = f'nb_r0noise_{r0_noise}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed,
                                                  r0_noise=r0_noise)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed,
                                                  r0_noise=r0_noise)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left)
        print(seed, name)
    
    ####### MIXTURE OF R0S
    num_samples = 4

    bias = np.array([1.0, 1.2])
    r0a = 0.1
    r0b = 1.9
    mid = (r0b - r0a) / 2
    p_mult = 0.4
    bias = 1
    kappa = None
    r0_noise = 0
    for t in np.linspace(1.0, 0.0, num=6):
        print(f'Sampling biased r0 mixture ra = {r0a}, rb = {r0b}, t = {t:.2f}...')
        r0 = np.array([r0a + mid* t, r0b - mid * t])
        name = f'nb_mixture_bias_2_r0w_{t:.2}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed,
                                                  r0_noise=r0_noise)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed,
                                                  r0_noise=r0_noise)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left, est_p=True)
    
    r0a = 0.1
    r0b = 1.9
    p_mult = 0.4
    r0_noise = 0
    kappa = None
    r0 = [r0a, r0b]
    for b in np.linspace(1.0, 1.5, num=6):
        bias = [1.0, b]
        print(f'Sampling biased r0 mixture ra = {r0a}, rb = {r0b}, bias = [1, {b:.2f}...')
        name = f'nb_mixture_bias_r0_bias_{b:.2}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed,
                                                  r0_noise=r0_noise, right_truncation=7500)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed,
                                                  r0_noise=r0_noise, right_truncation=7500)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left,  est_p=True)
        print(seed, name)
    
    num_samples = 4
    r0a = 0.1
    r0b = 1.9
    mid = (r0b - r0a) / 2
    p_mult = 0.4
    r0_noise = 0
    kappa = [128, 256]
    bias = [1.0, 1.2]
    for t in np.linspace(0.0, 1.0, num=6):
        print(f'Sampling biased r0 mixture ra = {r0a}, rb = {r0b}, kappa_a = {kappa[0]:.2f}, kappa_b = {kappa[1]:.2f} t = {t:.2f}...')
        r0 = np.array([r0a + mid* t, r0b - mid * t])
        name = f'bnb_mixture_bias_r0w_{t:.2f}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed,
                                                  r0_noise=r0_noise, right_truncation=7500)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed,
                                                  r0_noise=r0_noise, right_truncation=7500)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left)
        print(seed, name)
    num_samples = 4
    r0a = 0.1
    r0b = 1.9
    p_mult = 0.4
    r0_noise = 0
    bias = np.array([1.0, 1.2])
    for kappa in [16, 32, 64, 128, 256, 512, 1024]:
        print(f'Sampling biased 0.3 r0 mixture ra = {r0a}, rb = {r0b}, kappa = {kappa:.2f}...')
        r0 = np.array([r0a, r0b])
        name = f'bnb_mixture_bias_03_kappa_{kappa}'
        folder = os.path.join(datasets_folder, name)
        shutil.rmtree(folder, ignore_errors=True)
        filenames_base, sample_bias = sample_data(folder, prefix='noase', num_snps=num_snps, num_samples=num_samples, bad=bad,
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=0, seed=seed,
                                                  r0_noise=r0_noise, right_truncation=7500)
        filenames_ase, sample_bias = sample_data(folder, prefix='ase', num_snps=num_ases, num_samples=num_samples, bad=bad, 
                                                  left_truncation=left, kappa=kappa, bias=bias, p0=p0, r0=r0, w_frac=w_frac, p_mult=p_mult, seed=seed,
                                                  r0_noise=r0_noise, right_truncation=7500)
        filenames = filenames_base + filenames_ase
        out_filename = f'{name}.txt'
        with open(os.path.join(datasets_folder, out_filename), 'w') as f:
            f.write('\n'.join([os.path.join(name, f) for f in filenames]))
        script += mixalime_routine(name, datasets_folder, results_folder, left=left)
        print(seed, name)

    
    ############################
    with open(f'run_mixalime_{seed}.sh', 'w') as f:
        f.write('\n'.join(script))
    os.popen(f'chmod +x run_mixalime_{seed}.sh')
    p = os.popen(f'sh run_mixalime_{seed}.sh')
    with open(f'log_{seed}.txt', 'w') as f:
        f.write(p.read()) 
seeds = list(range(20))
with mp.Pool(min(mp.cpu_count(), len(seeds))) as p:
    list(p.map(gen, seeds))