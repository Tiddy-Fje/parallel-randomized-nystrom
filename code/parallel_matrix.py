import numpy as np
from mpi4py import MPI
from sequential import half_numpy_prod
#from icecream import ic
from scipy.linalg import solve_triangular, block_diag

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
    return multiply( A_ij, None, B_j, n, l, comm, only_C=True, half_numpy=False )

def multiply( A_ij, B_i_T, B_j, n, l, comm, only_C=False, half_numpy=True ):
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
    C_ij = None
    B_ij = None
    
    if not half_numpy:
        C_ij = A_ij @ B_j
        if not only_C:
            B_ij = B_i_T @ C_ij
    else:    
        C_ij = half_numpy_prod( A_ij, B_j )
        if not only_C:
            B_ij = half_numpy_prod( B_i_T, C_ij )

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


## check if need TSQR too ??
def parallel_CQR( A_l, m, n, comm ):
    rank = comm.Get_rank()
    size = comm.Get_size()

    G = np.empty((n, n))   
    comm.Allreduce( A_l.T@A_l, G, op=MPI.SUM )
    
    R = np.linalg.cholesky(G) # cholesky returns the lower triangular matrix
    #// solve R@Q_l.T=A_l.T as we have Q_l=A_l@(R.T)^-1
    Q_l = solve_triangular(R, A_l.T, lower=True).T
    
    Q = None
    if rank == 0:
        Q = np.empty((m, n), dtype=float)
    Q = comm.gather(Q_l, root=0)

    return Q, R

def row_distrib_mat( mat, comm, return_shape=False ):
    '''
    Distribute the rows of a matrix to the cores in communicator.
    '''
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    m, n = None, None
    if rank == 0:
        m, n = mat.shape
    m, n = comm.bcast( (m, n), root=0 )
    
    rows_per_proc = [m // size + (1 if x < m % size else 0) for x in range(size)]
    counts = np.array([r * n for r in rows_per_proc])
    displs = np.concatenate( ([0], np.cumsum( counts[:-1])) )
    #print(displs, counts.shape)
    #displs = [sum(counts[:i]) for i in range(size)]
    # On all ranks:
    local_rows = rows_per_proc[rank]
    local_A = np.empty((local_rows, n), dtype=np.float64)

    if rank == 0:
        comm.Scatterv([mat, counts, displs, MPI.DOUBLE], local_A, root=0)
    else:
        comm.Scatterv([None, counts, displs, MPI.DOUBLE], local_A, root=0)
    
    if return_shape:
        return local_A, (m, n)
    return local_A


def get_partner_idx( rank:int, k:int ) -> int:
    idx = 0
    if rank % 2**(k+1) == 0:
        idx = rank + 2**k
    else:
        idx = rank - 2**k
    return idx

def int_check( to_check ):
    assert to_check.is_integer(), "Value is not an integer"
    return int(to_check)

def TSQR(A_l, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    Ys = None
    Y_l_kp1, R_l_kp1 = np.linalg.qr(A_l, mode='reduced') 

    Ys = [Y_l_kp1]
    logp_tot = int_check(np.log2(size))
    for q in range(logp_tot): 
        # only keep needed processors 
        if not (rank % 2**q == 0):
            continue
        j = get_partner_idx(rank, q)
        if rank > j:
            comm.Send(R_l_kp1, dest=j)
            break
        else:
            R_j_kp1 = np.empty(R_l_kp1.shape, dtype=float)
            comm.Recv(R_j_kp1, source=j)
            Y_l_k, R_l_k = np.linalg.qr(np.concatenate((R_l_kp1,R_j_kp1), axis=0), mode='reduced')
            R_l_kp1 = R_l_k
            Ys.append(Y_l_k)
    R = None
    if rank == 0:
        R = R_l_k
    return Ys, R

def get_right_mat_shape( m:int, n:int, p:int, logp_tot:int ):
    if p == 2 ** logp_tot:
        return (int_check(m/p), n)
    m = 2 * n
    n = n
    return (m,n)

def build_Q( Y_s, m, n, comm ):
    rank = comm.Get_rank()
    size = comm.Get_size()
    logp_tot = int_check(np.log2(size))

    current_Q = None
    if rank == 0: # set up the initial Q
        current_Q = Y_s[-1]
        Y_s.pop()
    for q in range(logp_tot-1,-1,-1):
        # loop from pen-ultimate stage to first stage
        logp = logp_tot - q
        color = 1
        if not (rank % 2**q == 0):
            color = MPI.UNDEFINED
            new_comm = comm.Split(color, key=rank)
            continue
        new_comm = comm.Split(color, key=rank)

        p = 2**logp
        if new_comm is not None:
            subs_Q_k = None
            if new_comm.Get_rank() == 0:
                shape = get_right_mat_shape( m, n, p, logp_tot )
                subs_Q_k = np.empty((p, *shape), dtype=float) 
            sub_mat = Y_s[-1]
            Y_s.pop()
            new_comm.Gather(sub_mat, subs_Q_k, root=0)

            if new_comm.Get_rank() == 0:
                Q_k = block_diag(*subs_Q_k)
                current_Q = Q_k @ current_Q
            # Gather assembles the object by sorting received data by rank  
    return current_Q

def build_Q_bis( Y_s, comm ):
    # adapted from exercise session number 6
    # https://moodle.epfl.ch/pluginfile.php/3396647/mod_resource/content/1/Series6_solutions.pdf
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = Y_s[0].shape[1]
    Q = None

    if rank == 0:
        Q = np.eye(n, n)
        Q = Y_s[-1]@Q
        Y_s.pop()

    for k in range(int(np.log2(size))-1, -1, -1):
        color = rank%(2**k)
        key = rank//(2**k)
        comm_branch = comm.Split(color = color, key = key)
        if( color == 0):
            Qrows = np.empty((n,n), dtype = 'd')
            comm_branch.Scatterv(Q, Qrows, root = 0)
            
            Qlocal = Y_s[-1]@Qrows
            Y_s.pop()

            Q = comm_branch.gather(Qlocal, root = 0)
            if rank == 0:
                Q = np.concatenate(Q, axis = 0)
        comm_branch.Free()
    return Q

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    import time

    A, B = None, None
    m, n = 2**13, 2**8
    if rank == 0:
        A = np.random.rand(m,n)
        B = np.random.rand(m,n)
    
    A_l = row_distrib_mat(A, comm)

    start = time.perf_counter()
    Y_s, R = TSQR(A_l, comm)
    comm.barrier()
    end = time.perf_counter()
    if rank == 0:
        print("Time for TSQR: ", end-start)
    start = time.perf_counter()
    Q = build_Q_bis(Y_s, comm)
    comm.barrier()
    end = time.perf_counter()
    if rank == 0:
        print("Time for building Q: ", end-start)
    if rank == 0:
        print(np.allclose(Q.T@Q, np.eye(Q.shape[1])))
