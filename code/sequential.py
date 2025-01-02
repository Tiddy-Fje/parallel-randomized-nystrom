from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from data_generation import *
from utility import *

def half_numpy_prod( A, Omega ):
    C = np.zeros((A.shape[0], Omega.shape[1]))
    for i in range(A.shape[0]):
        C[i,:] = A[i,:] @ Omega
    return C

def sequential_gaussian_sketch( A, n, l, random_seed, half_numpy=True ):
    '''Generate Gaussian sketching matrix.'''
    rng = np.random.default_rng(random_seed)
    omega = rng.normal(size=(n, l)) # / np.sqrt(l)
    if not half_numpy:
        C = A @ omega
        B = omega.T @ C
        return B, C
    C = half_numpy_prod( A, omega )
    B = half_numpy_prod( omega.T, C )
    return B, C

def block_SRHT_bis( A, n, l, random_seed):
    '''Generate a block Subsampled Randomized Hadamard Transform (SRHT) sketching matrix.'''
    np.random.seed(random_seed)
    rows = np.concatenate( (np.ones(l), np.zeros(n-l)) ).astype(bool)
    perm = np.random.permutation(n)
    selected_rows = rows[perm]

    def omega_at_A( A_, D_L, D_R ):# this got checked
        temp =  D_R.reshape(-1,1) * A_
        fwht_mat( temp )
        R_temp = temp[selected_rows,:] / np.sqrt( n )
        # we normalise as should use normalised hadamard matrix
        return np.sqrt( n / l ) * D_L.reshape(-1,1) * R_temp
    
    D_L = np.random.choice([-1, 1], size=l, replace=True, p=[0.5, 0.5])
    D_R = np.random.choice([-1, 1], size=n, replace=True, p=[0.5, 0.5])
    C = omega_at_A( A.T, D_L, D_R ).T
    B = omega_at_A( C, D_L, D_R )
    return B, C

if __name__ == '__main__':
    print("This is sequential.py")