import numpy as np
from utils import synthetic_data, fwht_mat
from mpi4py import MPI

def gaussian_sketching( n, l, seed_factor, comm ):
    rank = comm.Get_rank()
    root_blocks = root_blocks_from_comm(comm)

    def get_omega_k( k ):
        row_blocks = np.ceil(n / root_blocks).astype(int)
        np.random.seed(seed_factor*(k+1))
        if k == root_blocks - 1:
            row_blocks = n - (root_blocks-1)*row_blocks
        return np.random.randn(row_blocks, l) / np.sqrt(l)
    
    i = rank // root_blocks
    j = rank % root_blocks
    omega_j = get_omega_k(j)
    
    if i == j :
        return omega_j.T, omega_j
    
    omega_i = get_omega_k(i)
    return omega_i.T, omega_j

def SRHT_sketching( A, l ):
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

    row_blocks, col_blocks = None, None
    m, n = A.shape
    if rank == 0:
        row_blocks = np.ceil(m / root_blocks).astype(int)
        col_blocks = np.ceil(n / root_blocks).astype(int)
    row_blocks, col_blocks = comm.bcast( (row_blocks, col_blocks), root=0 )

    A_ij = None
    for i in range(root_blocks):
        row_len = row_blocks
        if i == root_blocks - 1:
            row_len = m - i*row_blocks

        for j in range(root_blocks):
            col_len = col_blocks
            if j == root_blocks - 1:
                col_len = n - j*col_blocks

            A_ij = np.empty((row_len, col_len))
            if rank == 0:
                #print('Sending block ({}, {}) to process {}'.format(i, j, i*root_blocks+j))
                #print('Block shape : ', A[i*row_blocks:row_end, j*col_blocks:col_end].shape)
                row_end = i*row_blocks + row_len
                col_end = j*col_blocks + col_len
                temp = np.empty((row_len, col_len))
                temp[:,:] = A[i*row_blocks:row_end, j*col_blocks:col_end] # needed for contiguous memory
                if i*root_blocks+j == 0:
                    A_ij = temp
                else:
                    comm.Send( temp, dest=i*root_blocks+j ) # is isend better?
            elif rank == i*root_blocks+j:
                #print('Receiving block ({}, {}) from process {}'.format(i, j, 0))
                comm.Recv( A_ij, source=0 )
    return A_ij

def sketch( A_ij, n, l, seed_factor, comm ):
    rank = comm.Get_rank()
    root_blocks = root_blocks_from_comm(comm)

    omega_i_T, omega_j = gaussian_sketching( n, l, seed_factor, comm )
    C_ij = A_ij @ omega_j
    B_ij = omega_i_T @ C_ij

    ## Found splitting syntax asking GPT (from this line to ##)
    color_row = rank // root_blocks
    color_col = rank % root_blocks

    #print('Rank : ', rank, 'Color row : ', color_row, 'Color col : ', color_col)
    comm_row = comm.Split(color_row, rank)  # Communicator for each row of processes
    comm_col = comm.Split(color_col, rank)  # Communicator for each column of processes
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
        comm_col.Gather(C_i, C, root=0)

    comm.Reduce(B_ij, B, op=MPI.SUM, root=0)

    return B, C

if __name__ == '__main__':

    seed_factor = 1234
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = 2**11
    r = 2**5
    l = 2**9

    mat = synthetic_data( n, r, 'slow', 'polynomial' )

    if comm.size == 1:
        U, s, V = np.linalg.svd(mat, full_matrices=False)
        print('largest singular value of A: ', s[0])
        print('smallest singular value of A: ', s[-1])
    #if rank == 0:

    #print('Rank : ', rank, 'Input shape : ', mat.shape)
    #comm.barrier()
    mat_ij = split_matrix( mat, comm )
   # print('Split matrix shape : ', mat_ij.shape)

    #a = synthetic_data( n, r, 'fast', 'polynomial' )

    B, C = sketch( mat_ij, n, l, seed_factor, comm )
    if rank == 0:
        U, s, V = np.linalg.svd(B, full_matrices=False)
        print('largest singular value of A: ', s[0])
        print('smallest singular value of A: ', s[-1])
