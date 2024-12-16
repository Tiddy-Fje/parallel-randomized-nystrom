from parallel import time_k_rank
import numpy as np
from mpi4py import MPI
import h5py
#from data_generation import synthetic_matrix

def analysis( A_ij, n, l, k, sketch_type, seed_factor, comm, n_rep, output_file ):
    max_sketch_ts, max_k_rank_ts = time_k_rank( A_ij, n, l, k, sketch_type, seed_factor, comm, n_rep) 

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f'Results for {sketch_type} w/ n={n} l={l} k={k}')
        print(f'Max sketch ts : {max_sketch_ts}')
        print(f'Max k_rank ts : {max_k_rank_ts}')
        with h5py.File(f'{output_file}.h5', 'a') as f:            
            data_lab = f'n={n}_l={l}_k={k}'
            # if dataset already exists, skip the writing
            if f'sketch_ts_{data_lab}_mean' in f[f'{sketch_type}'].keys():
                return
            f[f'{sketch_type}'].create_dataset(f'sketch_ts_{data_lab}_mean', data=np.mean(max_sketch_ts))
            f[f'{sketch_type}'].create_dataset(f'sketch_ts_{data_lab}_std', data=np.std(max_sketch_ts))
            f[f'{sketch_type}'].create_dataset(f'k_rank_ts_{data_lab}_mean', data=np.mean(max_k_rank_ts))
            f[f'{sketch_type}'].create_dataset(f'k_rank_ts_{data_lab}_std', data=np.std(max_k_rank_ts))
    return

if __name__ == '__main__':
    n_rep = 2

    log2_l_min = 7
    log2_l_max = 10
    log2_ls = np.arange(log2_l_min, log2_l_max+1).astype(int)
    ls = 2 ** log2_ls
    n = 2 ** (log2_l_max+2)

    log2_n_min = 10
    log2_n_max = 13
    log2_ns = np.arange(log2_n_min, log2_n_max+1).astype(int)
    ns = 2 ** log2_ns
    l = 2 ** (log2_n_min-2)
    k = l // 3

    seed_factor = 1234

    output_file = '../output/sequential_performance'
    with h5py.File(f'{output_file}.h5', 'w') as f:
        f.create_group('Gaussian')
        f.create_group('SRHT')
        f.create_group('parameters')
        f['parameters'].create_dataset('n_rep', data=n_rep)
        f['parameters'].create_dataset('seed_factor', data=seed_factor)
        f['parameters'].create_dataset('l', data=l)
        f['parameters'].create_dataset('k', data=k)
        f['parameters'].create_dataset('n', data=n)
        f['parameters'].create_dataset('ls', data=ls)
        f['parameters'].create_dataset('ns', data=ns)

    comm = MPI.COMM_WORLD  
    for n_ in ns:
        A = np.random.normal(size=(n_, n_))    
        #A = synthetic_matrix(n_, n_//4, 'fast', 'exponential')
        analysis(A, n_, l, k, 'Gaussian', seed_factor, comm, n_rep, output_file)
        analysis(A, n_, l, k, 'SRHT', seed_factor, comm, n_rep, output_file)
    
    #A = synthetic_matrix(n, n//4, 'fast', 'exponential') 
    A = np.random.normal(size=(n, n))
    for l_ in ls:
        analysis(A, n, l_, k, 'Gaussian', seed_factor, comm, n_rep, output_file)
        analysis(A, n, l_, k, 'SRHT', seed_factor, comm, n_rep, output_file)
