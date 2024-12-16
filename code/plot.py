import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
import h5py
import data_generation as dg

# fix rcParams for plotting
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'lines.linewidth': 2, 'lines.markersize': 10})
plt.rcParams.update({'figure.autolayout': True})

def h5_to_dict(h5_group): # This function is Chat gpt output
    '''
    Recursively convert an h5py group or file into a nested dictionary.
    '''
    result = {}
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):  # If the item is a group, recurse
            result[key] = h5_to_dict(item)
        elif isinstance(item, h5py.Dataset):  # If the item is a dataset, read the data
            result[key] = item[()]  # Read the dataset's content
    return result

def file_to_dict( file ):
    '''
    Import data from file and use h5_to_dict function to convert the h5py file into a dictionary.
    '''
    data_dict = None
    with h5py.File(file, 'r') as f:
        data_dict = h5_to_dict(f)
    return data_dict

def import_data():
    '''
    Import data from ../output/sequential_performance.h5 and ../output/sequential_performance.h5.
    Use the function h5_to_dict to convert the h5py file into a dictionary.
    '''
    seq_data_dict = {}
    par_data_dict = {}
    seq_file = f'../output/sequential_performance.h5'
    par_file = f'../output/parallel_performance.h5'
    
    with h5py.File(seq_file, 'r') as f:
        seq_data_dict = h5_to_dict(f)
    with h5py.File(par_file, 'r') as f:
        par_data_dict = h5_to_dict(f)

    return seq_data_dict, par_data_dict

def l_variation( seq_data_dict ):
    '''
    Plot the average runtimes for the sequential algorithms as a function of l.
    '''
    ls = seq_data_dict['parameters']['ls']
    n = seq_data_dict['parameters']['n']
    k = seq_data_dict['parameters']['k']
    algos = list(seq_data_dict.keys())
    algos.remove('parameters')
    fig, ax = plt.subplots()
    for algo in algos:
        sketch_means = np.zeros((len(ls)))
        sketch_stds = np.zeros((len(ls)))
        krank_means = np.zeros((len(ls)))
        krank_stds = np.zeros((len(ls)))
        #print(seq_data_dict[algo].keys())
        for i,l in enumerate(ls):
            lab = f'n={n}_l={l}_k={k}'
            sketch_means[i] = seq_data_dict[algo][f'sketch_ts_{lab}_mean']
            sketch_stds[i] = seq_data_dict[algo][f'sketch_ts_{lab}_std']
            krank_means[i] = seq_data_dict[algo][f'k_rank_ts_{lab}_mean']
            krank_stds[i] = seq_data_dict[algo][f'k_rank_ts_{lab}_std']
        ax.errorbar(ls, sketch_means, yerr=sketch_stds, fmt='o', label=f'{algo}')
        ax.errorbar(ls, krank_means, yerr=krank_stds, fmt='s', label=f'{algo} rank k')
    ax.set_xlabel('$l$')
    ax.set_ylabel('Runtime [s]')
    ax.legend()
    plt.savefig('../figures/runtimes_l_variation.png')
    return

def n_variation( seq_data_dict ):
    '''
    Plot the average runtimes for the sequential algorithms as a function of n.
    '''
    ns = seq_data_dict['parameters']['ns']
    l = seq_data_dict['parameters']['l']
    k = seq_data_dict['parameters']['k']
    algos = list(seq_data_dict.keys())
    algos.remove('parameters')
    fig, ax = plt.subplots()
    for algo in algos:
        sketch_means = np.zeros((len(ns)))
        sketch_stds = np.zeros((len(ns)))
        krank_means = np.zeros((len(ns)))
        krank_stds = np.zeros((len(ns)))
        for i,n in enumerate(ns):
            lab = f'n={n}_l={l}_k={k}'
            sketch_means[i] = seq_data_dict[algo][f'sketch_ts_{lab}_mean']
            sketch_stds[i] = seq_data_dict[algo][f'sketch_ts_{lab}_std']
            krank_means[i] = seq_data_dict[algo][f'k_rank_ts_{lab}_mean']
            krank_stds[i] = seq_data_dict[algo][f'k_rank_ts_{lab}_std']
        ax.errorbar(ns, sketch_means, yerr=sketch_stds, fmt='o', label=f'{algo}')
        ax.errorbar(ns, krank_means, yerr=krank_stds, fmt='s', label=f'{algo} rank k')
    ax.set_xlabel('$n$')
    ax.set_ylabel('Runtime [s]')
    ax.legend()
    plt.savefig('../figures/runtimes_n_variation.png')
    return

def cores_variation( par_data_dict, ns_cores ):
    '''
    Plot the average runtimes for the parallel algorithms as a function of the number of cores.
    '''
    n_small = par_data_dict['parameters']['n_small']
    n_large = par_data_dict['parameters']['n_large']
    l_small = par_data_dict['parameters']['l_small']
    l_large = par_data_dict['parameters']['l_large']
    k_small = par_data_dict['parameters']['k_small']
    k_large = par_data_dict['parameters']['k_large']
    algos = ['SRHT', 'Gaussian']
    fmts = ['o', 's']
    cols = ['tab:blue', 'tab:orange']
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    n_cores = len(ns_cores)
    for j,algo in enumerate(algos):
        sketch_means_small = np.zeros(n_cores)
        sketch_stds_small = np.zeros(n_cores)
        sketch_means_large = np.zeros(n_cores)
        sketch_stds_large = np.zeros(n_cores)
        krank_means_small = np.zeros(n_cores)
        krank_stds_small = np.zeros(n_cores)
        krank_means_large = np.zeros(n_cores)
        krank_stds_large = np.zeros(n_cores)
        for i, n_cores in enumerate(ns_cores):
            lab_small = f'n={n_small}_l={l_small}_k={k_small}'
            lab_large = f'n={n_large}_l={l_large}_k={k_large}'
            sketch_means_small[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'sketch_ts_{lab_small}_mean']
            sketch_stds_small[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'sketch_ts_{lab_small}_std']
            sketch_means_large[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'sketch_ts_{k_large}_mean']
            sketch_stds_large[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'sketch_ts_{k_large}_std']
            krank_means_small[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'k_rank_ts_{lab_small}_mean']
            krank_stds_small[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'k_rank_ts_{lab_small}_std']
            krank_means_large[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'k_rank_ts_{lab_large}_mean']
            krank_stds_large[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'k_rank_ts_{lab_large}_std']
        ax[0].errorbar(ns_cores, sketch_means_small, yerr=sketch_stds_small, fmt=fmts[j], color=cols[0])
        ax[0].errorbar(ns_cores, krank_means_small, yerr=krank_stds_small, fmt=fmts[j], color=cols[1])
        ax[1].errorbar(ns_cores, sketch_means_large, yerr=sketch_stds_large, fmt=fmts[j], color=cols[0])
        ax[1].errorbar(ns_cores, krank_means_large, yerr=krank_stds_large, fmt=fmts[j], color=cols[1])
    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Runtime [s]')
    ax[0].set_suptitle(lab_small.replace('_',' '))
    ax[1].set_suptitle(lab_large.replace('_',' '))
    ax.legend()
    plt.savefig('../figures/runtimes_cores_variation.png')
    return

def plot_synthetic_spectra( n, r ):
    '''
    Plot the spectra of the synthetic_matrix(n,r,decay_rate,decay_type) for different decay rates and types.
    '''
    decay_rates = ['fast', 'medium', 'slow']
    color = ['tab:red', 'tab:orange', 'tab:green']
    decay_types = ['exponential', 'polynomial']
    markers = ['o', 's']
    # create colors increasing in lightness as decay rate increases
    # should use the matplotlib colormaps for this

    fig, ax = plt.subplots()
    for i,decay_rate in enumerate(decay_rates):
        for j, decay_type in enumerate(decay_types):
            A = dg.synthetic_matrix(n, r, decay_rate, decay_type)
            eigs = np.diag(A)
            ax.plot(np.arange(n), eigs, label=f'{decay_type}, {decay_rate}', color=color[i], \
                    marker=markers[j], linestyle='-', markersize=5)
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    plt.savefig('../figures/synthetic_spectra.png')
    return



if __name__ == '__main__':
    #plot_synthetic_spectra( 30+5, 5 )
    seq_data_dict, par_data_dict = import_data()
    l_variation( seq_data_dict )
    n_variation( seq_data_dict )
    cores_variation( par_data_dict, [4,16] )

#  with h5py.File(f'{output_file}.h5', 'r') as f:
#       print(f['Gaussian'].keys())
#        print(f['SRHT'].keys())
        
