import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
import h5py

# fix rcParams for plotting
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'lines.linewidth': 2, 'lines.markersize': 10})
plt.rcParams.update({'figure.autolayout': True})

def h5_to_dict(h5_group): # This function is Chat gpt output
    """
    Recursively convert an h5py group or file into a nested dictionary.
    """
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
    algos = list(seq_data_dict.keys())
    algos.remove('parameters')
    fig, ax = plt.subplots()
    for algo in algos:
        means = np.zeros((len(ls)))
        stds = np.zeros((len(ls)))
        #print(seq_data_dict[algo].keys())
        for i,l in enumerate(ls):
            means[i] = seq_data_dict[algo][f'n={n}_l={l}_mean']
            stds[i] = seq_data_dict[algo][f'n={n}_l={l}_std']
        ax.errorbar(ls, means, yerr=stds, fmt='o', label=f'{algo}')
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
    algos = list(seq_data_dict.keys())
    algos.remove('parameters')
    fig, ax = plt.subplots()
    for algo in algos:
        means = np.zeros((len(ns)))
        stds = np.zeros((len(ns)))
        for i,n in enumerate(ns):
            means[i] = seq_data_dict[algo][f'n={n}_l={l}_mean']
            stds[i] = seq_data_dict[algo][f'n={n}_l={l}_std']
        ax.errorbar(ns, means, yerr=stds, fmt='o', label=f'{algo}')
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
    algos = ['SRHT', 'Gaussian']
    fmts = ['o', 's']
    cols = ['tab:blue', 'tab:orange']
    fig, ax = plt.subplots()
    for j,algo in enumerate(algos):
        means_small = np.zeros((len(ns_cores)))
        stds_small = np.zeros((len(ns_cores)))
        means_large = np.zeros((len(ns_cores)))
        stds_large = np.zeros((len(ns_cores)))
        for i, n_cores in enumerate(ns_cores):
            means_small[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'n={n_small}_l={l_small}_mean']
            stds_small[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'n={n_small}_l={l_small}_std']
            means_large[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'n={n_large}_l={l_large}_mean']
            stds_large[i] = par_data_dict[f'{algo}_cores={n_cores}'][f'n={n_large}_l={l_large}_std']
        ax.errorbar(ns_cores, means_small, yerr=stds_small, fmt=fmts[j], label=f'n={n_small}, l={l_small}', color=cols[0])
        ax.errorbar(ns_cores, means_large, yerr=stds_large, fmt=fmts[j], label=f'n={n_large}, l={l_large}', color=cols[1])
    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Runtime [s]')
    ax.set_title('Gaussian: circles, SRHT: squares')
    ax.legend()
    plt.savefig('../figures/runtimes_cores_variation.png')
    return

if __name__ == '__main__':
    seq_data_dict, par_data_dict = import_data()
    l_variation( seq_data_dict )
    n_variation( seq_data_dict )
    cores_variation( par_data_dict, [4] )