import parallel_matrix as pm
import numpy as np
from mpi4py import MPI
from scipy.linalg import solve_triangular, ldl
from data_generation import synthetic_matrix, MNIST_matrix
import time
from utility import fwht_mat
from sequential import sequential_gaussian_sketch, block_SRHT_bis

def time_function( function, *args, **kwargs ):
    start = time.perf_counter()
    result = function(*args,**kwargs)
    if result is not None:
        result = tuple(result)
    end = time.perf_counter()
    return end - start, result


def time_k_rank( A_ij, n, l, k, sketch_type, seed_factor, comm, n_rep ):
    rank = comm.Get_rank()
    size = comm.Get_size() 

    sketch_fun = {}
    k_rank_fun = {}
    if sketch_type == 'Gaussian':
        sketch_fun['parallel'] = gaussian_sketching
        sketch_fun['sequential'] = sequential_gaussian_sketch
    elif sketch_type == 'SRHT':
        sketch_fun['parallel'] = SRHT_sketching
        sketch_fun['sequential'] = block_SRHT_bis
    else:
        raise ValueError('Unknown sketch type')
    k_rank_fun['parallel'] = rank_k_approx
    k_rank_fun['sequential'] = seq_rank_k_approx

    sketch_runtimes = np.empty(n_rep)
    k_rank_runtime = np.empty(n_rep)
    for  i in range(n_rep):
        BC = None
        comm.Barrier()
        if size == 1:
            sketch_runtimes[i], BC = time_function( sketch_fun['sequential'], A_ij, n, l, seed_factor )
        else:
            sketch_runtimes[i], BC = time_function( sketch_fun['parallel'], A_ij, n, l, seed_factor, comm )
        B, C = BC
        comm.Barrier()
        if size == 1:
            k_rank_runtime[i], _ = time_function( k_rank_fun['sequential'], B, C, n, k, return_A_k=False )
        else:
            k_rank_runtime[i], _ = time_function( k_rank_fun['parallel'], B, C, n, k, comm, return_A_k=False )
        comm.Barrier()

    if size == 1:
        return sketch_runtimes, k_rank_runtime

    tot_sketch_ts = None
    tot_k_rank_ts = None
    if rank == 0:
        tot_sketch_ts = np.empty((size,n_rep), dtype=float)
        tot_k_rank_ts = np.empty((size,n_rep), dtype=float)
    comm.Gather(sketch_runtimes, tot_sketch_ts, root=0)
    comm.Gather(k_rank_runtime, tot_k_rank_ts, root=0)
    
    max_sketch_ts = None
    max_k_rank_ts = None
    if rank == 0:
        max_sketch_ts = np.max(tot_sketch_ts, axis=0) 
        max_k_rank_ts = np.max(tot_k_rank_ts, axis=0)
    return max_sketch_ts, max_k_rank_ts


def gaussian_sketching( A_ij, n, l, seed_factor, comm, half_numpy=True ): 
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

def seq_rank_k_approx( B, C, n, k, return_A_k=True ):
    '''
    Compute k-rank approximation of A from B and C.

    Parameters
    B : np.ndarray, A @ Omega
    C : np.ndarray, Omega.T @ A @ Omega
    '''
    assert len(B) >= k, 'We should have l>=k'

    U_hat = None
    S_2 = None
    Q = None
    U = None

    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        lambdas, U = np.linalg.eigh(B)
        # first solution
        lambdas[lambdas<0.0] = 0.0
        # second solution
        #lambdas = np.sign(lambdas) * np.sqrt( np.sign(lambdas) * lambdas )
        L = U @ np.diag( np.sqrt(lambdas) ) @ U.T
  
    Z = solve_triangular(L, C.T, lower=True).T
    Q, R = np.linalg.qr(Z)
    U, s, V = np.linalg.svd(R, full_matrices=False)
    S_2 = s**2 
    #print(S_2)
    #time.sleep(1)
    U_hat = Q @ U[:,:k] 
    S_2 = S_2[:k]
    S_2.reshape(1,-1)

    if return_A_k:
        return (U_hat * S_2) @ U_hat.T


def rank_k_approx( B, C, n, k, comm, return_A_k=True ):
    rank = comm.Get_rank()
    S_2 = None
    Q = None
    U = None
    Z = None
    lambdas = None

    flag = True
    if rank == 0:
        assert len(B) > k, f'We should have l>k, but l={len(B)} and k={k}'

        try:
            L = np.linalg.cholesky(B)
            Z = solve_triangular(L, C.T, lower=True).T
        except np.linalg.LinAlgError:
            lambdas, U = np.linalg.eigh(B)
            flag = False
        
    if flag:
        Z_l, shape = pm.row_distrib_mat( Z, comm, return_shape=True)
        Ys, R = pm.TSQR( Z_l, comm )
        Q = pm.build_Q_bis( Ys, comm )
        if rank==0: 
            U, s, V = np.linalg.svd(R, full_matrices=False) # only rank 0 has QR results
            S_2 = s**2 
            U = U[:,:k]
    else:
        C_l, shape = pm.row_distrib_mat( C, comm, return_shape=True)
        tsqr_start = time.perf_counter()
        Ys, R = pm.TSQR( C_l, comm )
        tsqr_end = time.perf_counter()
        Q = pm.build_Q_bis( Ys, comm ) 
        if rank == 0:    
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

    if return_A_k:
        A_k = pm.full_multiply( arg_1, arg_2, comm ) 
        # this is (n^2)l in complexity, and is usually avoided. Here we need it to get the 
        # Frobenius norm of the error. We therefore do it in parallel.
        return A_k
    

if __name__ == '__main__':

    seed_factor = 1234
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = 2**10
    r = 2**8
    k = r*2
    p = k
    l = k + p

    #mat = synthetic_matrix( n, r, 'fast', 'polynomial' )
    mat = synthetic_matrix( n, r, 'fast', 'exponential' )
    c = 10
    mat_bis = MNIST_matrix( n, c )


    def compute_error( mat, n, l, k, seed_factor, comm, seq = True ):
        mat_k = None
        if seq:
            B, C = sequential_gaussian_sketch( mat, n, l, seed_factor )
            #B, C = block_SRHT_bis( mat, n, l, seed_factor )
            mat_k = seq_rank_k_approx( B, C, n, k )
        else:
            mat_ij = pm.split_matrix( mat, comm )
            B, C = gaussian_sketching( mat_ij, n, l, seed_factor, comm )
            mat_k = rank_k_approx( B, C, n, k, comm )
        if rank == 0:
            print(np.linalg.norm(mat - mat_k, 'fro') / np.linalg.norm(mat, 'fro'))

    compute_error( mat, n, l, k, seed_factor, comm )

