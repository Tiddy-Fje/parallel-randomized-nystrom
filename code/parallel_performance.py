from parallel import gaussian_sketching, SRHT_sketching
from sequential_performance import analysis
from parallel_matrix import split_matrix
from data_generation import synthetic_matrix
from mpi4py import MPI
import h5py
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

seed_factor = 1234
n_rep = 3

log2_n_small = 9
n_small = 2 ** log2_n_small
l_small = 2 ** (log2_n_small-2)

log2_n_large = 12
n_large = 2 ** log2_n_large
l_large = 2 ** (log2_n_large-2)

output_file = '../output/parallel_performance'
if rank == 0:
    with h5py.File(f'{output_file}.h5', 'w') as f:
        f.create_group(f'Gaussian_cores={size}')
        f.create_group(f'SRHT_cores={size}')
        if not f'parameters' in f.keys():
            f.create_group('parameters')
            f['parameters'].create_dataset('n_rep', data=n_rep)
            f['parameters'].create_dataset('seed_factor', data=seed_factor)
            f['parameters'].create_dataset('n_small', data=n_small)
            f['parameters'].create_dataset('n_large', data=n_large)
            f['parameters'].create_dataset('l_small', data=l_small)
            f['parameters'].create_dataset('l_large', data=l_large)

A = synthetic_matrix(n_small, n_small//4, 'fast', 'exponential')
A_ij = split_matrix(A, comm)
analysis(A_ij, n_small, l_small, gaussian_sketching, seed_factor, comm, n_rep, output_file)
analysis(A_ij, n_small, l_small, SRHT_sketching, seed_factor, comm, n_rep, output_file)

B = synthetic_matrix(n_large, n_large//4, 'fast', 'exponential')
B_ij = split_matrix(B, comm)
analysis(B_ij, n_large, l_large, gaussian_sketching, seed_factor, comm, n_rep, output_file)
analysis(B_ij, n_large, l_large, SRHT_sketching, seed_factor, comm, n_rep, output_file)



