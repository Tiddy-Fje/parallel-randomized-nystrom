from main import gaussian_sketching, SRHT_sketching, time_sketching
import numpy as np
from mpi4py import MPI
import time
import h5py

def analysis( n, l, algorithm, seed_factor, comm, n_rep, output_file ):
    max_runtimes = time_sketching(n, l, algorithm, seed_factor, comm, n_rep) 

    label = None
    if algorithm == gaussian_sketching:
        label = 'gaussian'
    elif algorithm == SRHT_sketching:
        label = 'SRHT'
    else:
        raise ValueError(f"Unknown algorithm")

    rank = comm.Get_rank()
    if rank == 0:
        print(f'Max runtime: {max_runtimes}')
        with h5py.File(f'{output_file}.h5', 'a') as f:
            f[label].create_dataset(f'n={n}_mean', data=np.mean(max_runtimes))
            f[label].create_dataset(f'n={n}_std', data=np.std(max_runtimes))
    return max_runtimes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_rep = 5

log2_n_min = 10
log2_n_max = 12
log2_ns = np.arange(log2_n_min, log2_n_max+1).astype(int)
ns = 2 ** log2_ns
l = 2 ** (log2_ns[0]-1)
seed_factor = 1234

output_file = '../output/performance'
if rank == 0:
    with h5py.File(f'{output_file}.h5', 'w') as f:
        f.create_group('gaussian')
        f.create_group('SRHT')
        f.create_group('parameters')
        f['parameters'].create_dataset('n_rep', data=n_rep)
        f['parameters'].create_dataset('l', data=l)
        f['parameters'].create_dataset('seed_factor', data=seed_factor)
        
for n in ns:
    temp = analysis(n, l, gaussian_sketching, seed_factor, comm, n_rep, output_file)
    temp = analysis(n, l, SRHT_sketching, seed_factor, comm, n_rep, output_file)

if rank == 0:
    with h5py.File(f'{output_file}.h5', 'r') as f:
        print(f['gaussian'].keys())
        print(f['SRHT'].keys())
        print(f['gaussian']['n=2048_mean'][()])
        print(f['SRHT']['n=2048_mean'][()])

