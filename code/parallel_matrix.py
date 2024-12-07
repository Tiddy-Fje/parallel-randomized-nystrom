import numpy as np
from mpi4py import MPI
#from icecream import ic

def root_blocks_from_comm( comm ):
    '''
    Get the number of blocks in each row and column of the core-grid.
    '''
    size = comm.Get_size()
    root_blocks = np.sqrt(size)
    assert root_blocks == int(root_blocks), 'Number of cores must be perfect square'
    return int(root_blocks)


def split_matrix( A, comm ):
    '''
    Split matrix A into blocks and distribute them to the core-grid.
    '''
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
                row_end = i*row_blocks + row_len
                col_end = j*col_blocks + col_len
                temp[:,:] = A[i*row_blocks:row_end, j*col_blocks:col_end] # needed for contiguous memory
                if i*root_blocks+j == 0: # rank 0
                    A_ij = np.copy(temp)
                else:
                    comm.Send( temp, dest=i*root_blocks+j ) 
            elif rank == i*root_blocks+j:
                A_ij = np.empty((row_len, col_len))
                comm.Recv( A_ij, source=0 )
    return A_ij

def get_A_i_to_column_i( A, comm ):
    '''
    Distribute the i-th row-block of A to the i-th column of cores in core-grid.
    '''
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

def full_multiply( A, B, comm ):
    '''
    Multiply A and B matrices in parallel. Result is returned only to rank 0.
    '''
    n, l = None, None
    rank = comm.Get_rank()
    if rank == 0:
        n = A.shape[0]
        l = B.shape[1]
    A_ij = split_matrix(A, comm)
    B_j = get_A_i_to_column_i(B, comm)
    return multiply( A_ij, None, B_j, n, l, comm, only_C=True )

def multiply( A_ij, B_i_T, B_j, n, l, comm, only_C=False ):
    '''
    Multiplies A and B matrices in parallel to obtain B.T@A@B and A@B. 
    A_ij is the local block of A matrix.
    B_i_T is the transpose of the i-th local block of B matrix.
    B_j is the j-th local block of B matrix.
    A.shape[0].
    B.shape[1].
    seed_factor is the seed for the random number generator.
    comm is the MPI communicator.
    only_C is a flag to indicate if only the C matrix has to be computed. In this case B_i_T is not used.
    '''
    C_ij = A_ij @ B_j
    B_ij = None
    if not only_C:
        B_ij = B_i_T @ C_ij

    return assemble_B_C( C_ij, B_ij, n, l, comm, only_C )
    
def assemble_B_C( C_ij, B_ij, n, l, comm, only_C=False ):
    '''
    Assemble the B and C matrices from the local blocks.
    C_ij is the local block of C matrix.
    B_ij is the local block of B matrix.
    n is the number of rows of the global A matrix.
    l is the number of columns of the global A matrix.
    comm is the MPI communicator.
    only_C is a flag to indicate if only the C matrix has to be computed. In this case B_ij is not used.
    '''
    rank = comm.Get_rank()
    root_blocks = root_blocks_from_comm(comm)
    
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
        # making sure arrays are C_CONTIGUOUS and not F_CONTIGUOUS
        C_i = np.ascontiguousarray(C_i)
        comm_col.Gather(C_i, C, root=0)

    if only_C:
        return C
    
    comm.Reduce(B_ij, B, op=MPI.SUM, root=0)
    return B, C
