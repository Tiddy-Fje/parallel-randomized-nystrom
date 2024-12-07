import parallel_matrix as pm
import numpy as np
from mpi4py import MPI
from scipy.linalg import solve_triangular, hadamard
from data_generation import synthetic_matrix, MNIST_matrix
import time
from utility import fwht_mat
from sequential import sequential_gaussian_sketch, block_SRHT_bis
#from icecream import ic

def time_sketching( A_ij, n, l, algorithm, seed_factor, comm, n_rep ):
    rank = comm.Get_rank()
    size = comm.Get_size() 

    runtimes = np.empty(n_rep)
    for  i in range(n_rep):
        start = time.perf_counter()
        if size == 1:
            B, C = algorithm( A_ij, n, l, seed_factor )
        else:
            B, C = algorithm( A_ij, n, l, seed_factor, comm ) 
        end = time.perf_counter()
        runtimes[i] = end - start
        comm.Barrier()

    if size == 1:
        return runtimes

    tot_runtimes = None
    if rank == 0:
        tot_runtimes = np.empty((size,n_rep), dtype=float)
    comm.Gather(runtimes, tot_runtimes, root=0)

    max_runtimes = None
    if rank == 0:
        max_runtimes = np.max(tot_runtimes, axis=0) 
    return max_runtimes


def gaussian_sketching( A_ij, n, l, seed_factor, comm ): 
    if comm.Get_size() == 1:
        return sequential_gaussian_sketch( A_ij, n, l, seed_factor )
    rank = comm.Get_rank()
    root_blocks = pm.root_blocks_from_comm(comm)

    def get_omega_k( k ):
        row_blocks = np.ceil(n / root_blocks).astype(int)
        np.random.seed(seed_factor*(k+1))
        if k == root_blocks - 1:
            row_blocks = n - (root_blocks-1)*row_blocks
        factor = np.sqrt(l)# CHECK WHICH FACTOR TO USE
        factor = 1.0
        return np.random.randn(row_blocks, l) / factor
    
    i = rank // root_blocks
    j = rank % root_blocks
    omega_j = get_omega_k(j)
    omega_i = get_omega_k(i)
   
    #print(f'rank={rank}, a_ij={A_ij.shape}, omega_i={omega_i.shape}, omega_j={omega_j.shape}')
    return pm.multiply( A_ij, omega_i.T, omega_j, n, l, comm )

def int_check( to_check ):
    assert to_check.is_integer(), 'Value is not an integer'
    return int(to_check)

def SRHT_sketching( A_ij, n, l, seed_factor, comm  ):
    if comm.Get_size() == 1:
        return block_SRHT_bis( A_ij, n, l, seed_factor )
    root_blocks = pm.root_blocks_from_comm(comm)
    n_over_root_p = int_check( n / root_blocks )
    
    def get_samples_for_k( k ):
        np.random.seed(seed_factor*(k+1))
        D_Lk = np.random.choice([-1, 1], size=l, replace=True, p=[0.5, 0.5])
        D_Rk = np.random.choice([-1, 1], size=n_over_root_p, replace=True, p=[0.5, 0.5])
        return D_Lk, D_Rk
    
    assert n_over_root_p >= l, f'l={l} should be < n/sqrt(p)={n_over_root_p}'
    rows = np.concatenate( (np.ones(l), np.zeros(n_over_root_p-l)) ).astype(bool)
    np.random.seed( seed_factor*(root_blocks+2) ) # to avoid seed overlap
    perm = np.random.permutation(n_over_root_p)
    selected_rows = rows[perm]    

    def omega_k_at_A_ks( A_ks, D_Lk, D_Rk ):# this got checked
        temp =  D_Rk.reshape(-1,1) * A_ks
        fwht_mat( temp )
        R_temp = temp[selected_rows,:] / np.sqrt( n_over_root_p )
        # we normalise as should use normalised hadamard matrix
        return np.sqrt( n / (root_blocks*l) ) * D_Lk.reshape(-1,1) * R_temp

    rank = comm.Get_rank()
    i = rank // root_blocks
    j = rank % root_blocks
    
    D_Li, D_Ri = get_samples_for_k( i ) 
    D_Lj, D_Rj = get_samples_for_k( j )

    C_ij = omega_k_at_A_ks( A_ij.T, D_Lj, D_Rj ).T
    B_ij = omega_k_at_A_ks( C_ij, D_Li, D_Ri )
    
    return pm.assemble_B_C( C_ij, B_ij, n, l, comm, only_C=False )   

def seq_rank_k_approx( B, C, n, k ):
    '''
    Compute k-rank approximation of A from B and C.

    Parameters
    B : np.ndarray, the sketch of A
    C : np.ndarray, the sketch of A
    '''
    A_k = None
    U_hat = None
    S_2 = None
    Q = None
    U = None
    try:
        L = np.linalg.cholesky(B)
        Z = solve_triangular(L, C.T, lower=True).T
        Q, R = np.linalg.qr(Z)
        U, s, V = np.linalg.svd(R, full_matrices=False)
        S_2 = s**2 
    except np.linalg.LinAlgError:
        lambdas, U = np.linalg.eigh(B)
        Q, R = np.linalg.qr(C)
        S_2 = lambdas

    U_hat = Q @ U[:,:k] 
    S_2 = S_2[:k]
    S_2.reshape(1,-1)

    return U_hat @ np.diag(S_2) @ U_hat.T


def rank_k_approx( B, C, n, k, comm ):
    rank = comm.Get_rank()

    S_2 = None
    Q = None
    U = None
    flag = False
    if rank == 0:
        try:
            L = np.linalg.cholesky(B)
            Z = solve_triangular(L, C.T, lower=True).T
        except np.linalg.LinAlgError:
            lambdas, U = np.linalg.eigh(B)
        
    if flag:
        if rank==0: 
            Q, R = np.linalg.qr(Z)
            U, s, V = np.linalg.svd(R, full_matrices=False) # only rank 0 has QR results
            S_2 = s**2 
            U = U[:,:k]
    else:
        if rank == 0:    
            Q, R = np.linalg.qr(C)
            S_2 = lambdas 
            U = U[:,:k]

    U_hat = pm.full_multiply( Q, U, comm )
    arg_1 = None
    arg_2 = None
    if rank == 0:
        S_2 = S_2[:k]
        S_2.reshape(1,-1)
        arg_1 = U_hat * S_2
        arg_2 = U_hat.T

    A_k = pm.full_multiply( arg_1, arg_2, comm )
    # this is (n^2)l in complexity, and is usually avoided. Here we need it to get the 
    # Frobenius norm of the error. We therefore do it in parallel.

    return A_k



if __name__ == '__main__':

    seed_factor = 1234
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = 2**11
    r = 2**7
    k = 2**7 + 5
    p = 2**7
    l = k + p

    #mat = synthetic_matrix( n, r, 'fast', 'polynomial' )
    mat = synthetic_matrix( n, r, 'fast', 'exponential' )

    c = 10
    mat_bis = MNIST_matrix( n, c )
    mat_ij = pm.split_matrix( mat_bis, comm )


    def compute_error( mat, n, l, k, seed_factor, comm, seq = True ):
        mat_ij = pm.split_matrix( mat, comm )
        B, C = gaussian_sketching( mat_ij, n, l, seed_factor, comm )
        if seq:
            if rank == 0:
                mat_k = seq_rank_k_approx( B, C, n, k )
        else:
            mat_k = rank_k_approx( B, C, n, k, comm )
        if rank == 0:
            print(np.linalg.norm(mat - mat_k, 'nuc') / np.linalg.norm(mat, 'nuc'))

    compute_error( mat, n, l, k, seed_factor, comm )

