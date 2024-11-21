from parallel import gaussian_sketching, SRHT_sketching, time_sketching
from sequential import sequential_gaussian_sketch, block_SRHT, block_SRHT_bis
import numpy as np
from mpi4py import MPI
import h5py

# TO DO : SEQUENTIAL SRHT SKETCHING FUNCTION SLOWER THAN ALTERNATIVE (TALK W/ AMAL)

def analysis( n, l, algorithm, seed_factor, comm, n_rep, output_file ):
    max_runtimes = time_sketching(n, l, algorithm, seed_factor, comm, n_rep) 

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        label = None
        if algorithm in [gaussian_sketching,sequential_gaussian_sketch]:
            label = 'Gaussian'
        elif algorithm in [SRHT_sketching, block_SRHT, block_SRHT_bis]:
            label = 'SRHT'
        else:
            raise ValueError(f"Unknown algorithm")
        
        print(f'Max runtime for {label} w/ n={n} l={l}: {max_runtimes}')
        with h5py.File(f'{output_file}.h5', 'a') as f:
            cores_lab = ''
            if 'parallel' in output_file:
                cores_lab = f'_cores={size}'
            data_lab = f'n={n}_l={l}'
            # if dataset already exists, skip the writing
            if f'{data_lab}_mean' in f[f'{label}{cores_lab}'].keys():
                return
            print(f'{label}{cores_lab}')
            print(data_lab)
            f[f'{label}{cores_lab}'].create_dataset(f'{data_lab}_mean', data=np.mean(max_runtimes))
            f[f'{label}{cores_lab}'].create_dataset(f'{data_lab}_std', data=np.std(max_runtimes))

    return

if __name__ == '__main__':
    n_rep = 3

    log2_l_min = 8
    log2_l_max = 10
    log2_ls = np.arange(log2_l_min, log2_l_max+1).astype(int)
    ls = 2 ** log2_ls
    n = 2 ** (log2_l_max+1)

    log2_n_min = 10
    log2_n_max = 13
    log2_ns = np.arange(log2_n_min, log2_n_max+1).astype(int)
    ns = 2 ** log2_ns
    l = 2 ** (log2_n_min-1)

    seed_factor = 1234

    output_file = '../output/sequential_performance'
    with h5py.File(f'{output_file}.h5', 'w') as f:
        f.create_group('Gaussian')
        f.create_group('SRHT')
        f.create_group('parameters')
        f['parameters'].create_dataset('n_rep', data=n_rep)
        f['parameters'].create_dataset('seed_factor', data=seed_factor)
        f['parameters'].create_dataset('l', data=l)
        f['parameters'].create_dataset('n', data=n)
        f['parameters'].create_dataset('ls', data=ls)
        f['parameters'].create_dataset('ns', data=ns)


    comm = MPI.COMM_WORLD    
    for n_ in ns:
        analysis(n_, l, sequential_gaussian_sketch, seed_factor, comm, n_rep, output_file)
        #analysis(n, l, block_SRHT, seed_factor, comm, n_rep, output_file)
        analysis(n_, l, block_SRHT_bis, seed_factor, comm, n_rep, output_file)
    for l in ls:
        analysis(n, l, sequential_gaussian_sketch, seed_factor, comm, n_rep, output_file)
        #analysis(n, l, block_SRHT, seed_factor, comm, n_rep, output_file)
        analysis(n, l, block_SRHT_bis, seed_factor, comm, n_rep, output_file)

    with h5py.File(f'{output_file}.h5', 'r') as f:
        print(f['Gaussian'].keys())
        print(f['SRHT'].keys())
        
