import numpy as np
from utils import synthetic_data, fwht_mat
from mpi4py import MPI
import scipy as sp
from scipy.linalg import solve_triangular

def gaussian_sketching( n, l, seed_factor, comm ):
    rank = comm.Get_rank()
    root_blocks = root_blocks_from_comm(comm)

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

def root_blocks_from_comm( comm ):
    size = comm.Get_size()
    root_blocks = np.sqrt(size)
    assert root_blocks == int(root_blocks), 'Number of cores must be perfect square'
    return int(root_blocks)

def split_matrix( A, comm ):
    rank = comm.Get_rank()
    root_blocks = root_blocks_from_comm(comm)

    row_blocks, col_blocks, m, n = None, None, None, None
    if rank == 0:
        m, n = A.shape
        row_blocks = np.ceil(m / root_blocks).astype(int)
        col_blocks = np.ceil(n / root_blocks).astype(int)
    row_blocks, col_blocks, m, n = comm.bcast( (row_blocks, col_blocks, m, n), root=0 )

    A_ij = None
    for i in range(root_blocks):
        row_len = row_blocks
        if i == root_blocks - 1:
            row_len = m - i*row_blocks

        for j in range(root_blocks):
            col_len = col_blocks
            if j == root_blocks - 1:
                col_len = n - j*col_blocks

            temp = np.empty((row_len, col_len))
            if rank == 0:
                #print('Sending block ({}, {}) to process {}'.format(i, j, i*root_blocks+j))
                row_end = i*row_blocks + row_len
                col_end = j*col_blocks + col_len
                temp[:,:] = A[i*row_blocks:row_end, j*col_blocks:col_end] # needed for contiguous memory
                if i*root_blocks+j == 0: # rank 0
                    A_ij = np.copy(temp)
                else:
                    comm.Send( temp, dest=i*root_blocks+j ) 
            elif rank == i*root_blocks+j:
                A_ij = np.empty((row_len, col_len))
                #print('Receiving block ({}, {}) from process {}'.format(i, j, 0))
                comm.Recv( A_ij, source=0 )
    return A_ij

def get_A_i_to_column_i( A, comm ):
    rank = comm.Get_rank()

    root_blocks = root_blocks_from_comm(comm)
    row_blocks, m, n = None, None, None
    if rank == 0:
        m, n = A.shape
        row_blocks = np.ceil(m / root_blocks).astype(int)
    row_blocks, m, n = comm.bcast( (row_blocks, m, n), root=0 )

    color_col = rank % root_blocks
    comm_col = comm.Split(color_col, rank)  # Communicator for each column of cores

    A_i = None
    for i in range(root_blocks):
        row_len = row_blocks
        if i == root_blocks - 1:
            row_len = m - i*row_blocks
        
        block_i = np.empty((row_len, n))
        if rank == 0:
            row_end = i*row_blocks + row_len
            block_i[:,:] = A[i*row_blocks:row_end,:] # needed for contiguous memory
            #if i == 0: # rank 0
            #    A_i = np.copy(block_i)
            if i != 0: # we don't need to send to rank 0
                comm.Send( block_i, dest=i ) 
        
        j = rank % root_blocks # column index
        if j == i:
            A_i = np.empty((row_len, n))
            if i == 0: # rank 0 still has the block
                A_i = comm_col.bcast( block_i, root=0 )
            else: 
                if rank == i:
                    comm.Recv( A_i, source=0 )
                A_i = comm_col.bcast( A_i, root=0 )
    return A_i

def multiply( A_ij, B_i_T, B_j, n, l, comm, only_C=False ):
    '''
    Multiplies A and B matrices in parallel to obtain B.T@A@B and A@B. 
    A_ij is the local block of A matrix.
    B_i_T is the transpose of the i-th local block of B matrix.
    B_j is the j-th local block of B matrix.
    n is the size of the A matrix.
    l is the number of columns in the B matrix.
    seed_factor is the seed for the random number generator.
    comm is the MPI communicator.
    '''
    rank = comm.Get_rank()
    root_blocks = root_blocks_from_comm(comm)

    C_ij = A_ij @ B_j
    B_ij = None
    if not only_C:
        B_ij = B_i_T @ C_ij

    ## Found splitting syntax asking GPT (from this line to ##)
    color_row = rank // root_blocks
    color_col = rank % root_blocks

    comm_row = comm.Split(color_row, rank)  # Communicator for each row of cores
    comm_col = comm.Split(color_col, rank)  # Communicator for each column of cores
    ##

    C_i = None
    if color_col == 0:
        C_i = np.empty_like(C_ij)
    comm_row.Reduce(C_ij, C_i, op=MPI.SUM, root=0) # the new 0 rank, not global one

    C = None
    B = None
    if rank == 0:
        C = np.empty((n,l))
        B = np.empty_like(B_ij)

    if color_col == 0:
        if rank == 0:
            print(C_i.shape, C.shape)
        comm_col.Gather(C_i, C, root=0)

    if not only_C:
        comm.Reduce(B_ij, B, op=MPI.SUM, root=0)
    return B, C

def rank_k_approx(A_ij, n, l, k, seed_factor, comm ):
    ## What should be parallelized here ? ##
    # QR decompositions are n x l in complexity
    # SVD and EIG are l^3 in complexity
    B_i_T, B_j = gaussian_sketching( n, l, seed_factor, comm )
    B, C = multiply( A_ij, B_i_T, B_j, n, l, comm )

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
    A_k_ij = split_matrix( arg_1, comm )
    M_j = get_A_i_to_column_i( arg_2, comm )
    _, A_k = multiply( A_k_ij, M_j, M_j, n, n, comm, only_C=True )
    
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
    mat_ij = split_matrix( mat, comm )

    mat_k = rank_k_approx( mat_ij, n, l, k, seed_factor, comm )
    if rank == 0:
        print('Frobenius norm of the error: ', np.linalg.norm(mat - mat_k, 'fro') / np.linalg.norm(mat, 'fro'))