from main import gaussian_sketching, SRHT_sketching
import numpy as np
from mpi4py import MPI
import time
import h5py

def time_sketching( n, l, algorithm, seed_factor, comm, n_rep, output_file ): 
    label = None
    if algorithm == gaussian_sketching:
        label = 'gaussian'
    elif algorithm == SRHT_sketching:
        label = 'SRHT'
    else:
        raise ValueError(f"Unknown algorithm")

    runtimes = np.empty(n_rep)
    for  i in range(n_rep):
        start = time.perf_counter()
        omega_T, omega = algorithm(n, l, seed_factor, comm)
        end = time.perf_counter()
        runtimes[i] = end - start
        comm.Barrier()

    tot_runtimes = None
    if rank == 0:
        tot_runtimes = np.empty((size,n_rep), dtype=float)
    comm.Gather(runtimes, tot_runtimes, root=0)

    max_runtimes = None
    if rank == 0:
        max_runtimes = np.max(tot_runtimes, axis=0) 
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
    time_sketching(n, l, gaussian_sketching, seed_factor, comm, n_rep, output_file)
    time_sketching(n, l, SRHT_sketching, seed_factor, comm, n_rep, output_file)

if rank == 0:
    with h5py.File(f'{output_file}.h5', 'r') as f:
        print(f['gaussian'].keys())
        print(f['SRHT'].keys())
        print(f['gaussian']['n=2048_mean'][()])
        print(f['SRHT']['n=2048_mean'][()])

