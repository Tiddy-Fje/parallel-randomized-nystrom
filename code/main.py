from utils import synthetic_data, fwht_mat
import parallel_matrix as pm
import numpy as np
from mpi4py import MPI
from scipy.linalg import solve_triangular

def gaussian_sketching( n, l, seed_factor, comm ):
    rank = comm.Get_rank()
    root_blocks = pm.root_blocks_from_comm(comm)

    def get_omega_k( k ):
        row_blocks = np.ceil(n / root_blocks).astype(int)
        np.random.seed(seed_factor*(k+1))
        if k == root_blocks - 1:
            row_blocks = n - (root_blocks-1)*row_blocks
        factor = np.sqrt(l)
        factor = 1.0
        return np.random.randn(row_blocks, l) / factor
    
    i = rank // root_blocks
    j = rank % root_blocks
    omega_j = get_omega_k(j)
    
    if i == j :
        return omega_j.T, omega_j
    
    omega_i = get_omega_k(i)
    return omega_i.T, omega_j

def SRHT_sketching( A, l ):
    # TO DO : check whether fwht_mat is working correctly (tara adapted from twht)
    m, n = A.shape
    D = np.diag( np.random.choice([-1, 1], m, replace=True, p=[0.5, 0.5]) ).astype(float)
    # applying the fast Walsh-Hadamard transform
    fwht_mat(D)

    rows = np.concatenate( (np.ones(l), np.zeros(m-l)) ).astype(bool)
    perm = np.random.permutation(m)
    selected_rows = rows[perm]

    return np.sqrt(m/l) * D[selected_rows,:] @ A


def rank_k_approx(A_ij, n, l, k, seed_factor, comm ):
    ## What should be parallelized here ? ##
    # QR decompositions are n x l in complexity
    # SVD and EIG are l^3 in complexity
    B_i_T, B_j = gaussian_sketching( n, l, seed_factor, comm )
    B, C = pm.multiply( A_ij, B_i_T, B_j, n, l, comm )

    A_k = None
    U_hat = None
    S_2 = None
    if rank == 0:
        Q = None
        U = None
        try:
            L = np.linalg.cholesky( B )
            Z = solve_triangular(L, C.T, lower=True).T
            Q, R = np.linalg.qr(Z) # should this be done in parallel ?
            U, s, V = np.linalg.svd(R, full_matrices=False)
            S_2 = s**2 
        except np.linalg.LinAlgError:
            lambdas, U = np.linalg.eigh( B )
            Q, R = np.linalg.qr( C ) # should this be done in parallel ?
            S_2 = lambdas

        U_hat = Q @ U[:,:k] # should this be done in parallel ?
        S_2 = S_2[:k]
        S_2.reshape(1,-1)

    # this is (n^2)l in complexity, and is usually avoided
    # here we need it to get the Frobenius norm of the error
    # we therefore do it in parallel 
    arg_1 = None
    arg_2 = None
    if rank == 0:
        arg_1 = U_hat * S_2
        arg_2 = U_hat.T
    A_k_ij = pm.split_matrix( arg_1, comm )
    M_j = pm.get_A_i_to_column_i( arg_2, comm )
    _, A_k = pm.multiply( A_k_ij, M_j, M_j, n, n, comm, only_C=True )
    
    return A_k


if __name__ == '__main__':

    seed_factor = 1234
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = 2**11
    r = 2**7
    k = 2**7
    p = 2**2
    l = k + p

    mat = synthetic_data( n, r, 'fast', 'polynomial' )
    #mat = synthetic_data( n, r, 'fast', 'exponential' )
    mat_ij = pm.split_matrix( mat, comm )

    mat_k = rank_k_approx( mat_ij, n, l, k, seed_factor, comm )
    if rank == 0:
        print('Frobenius norm of the error: ', np.linalg.norm(mat - mat_k, 'fro') / np.linalg.norm(mat, 'fro'))