from main import  root_blocks_from_comm
import numpy as np
from mpi4py import MPI

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




def simple_vec( n, l, seed_factor, comm ):
    rank = comm.Get_rank()
    root_blocks = root_blocks_from_comm(comm)

    np.random.seed(seed_factor)
    v = np.random.randn(n,l)
    
    def get_omega_k( k ):
        row_blocks = np.ceil(n / root_blocks).astype(int)
        if k == root_blocks - 1:
            row_blocks = n - (root_blocks-1)*row_blocks
        row_end = k*row_blocks + row_blocks
        return v[k*row_blocks:row_end,:]
    
    i = rank // root_blocks
    j = rank % root_blocks
    omega_j = get_omega_k(j)
    
    if i == j :
        return omega_j.T, omega_j
    
    omega_i = get_omega_k(i)
    return omega_i.T, omega_j

# check whether split_matrix is working correctly
# do this by checking whether the blocks compose the original matrix

n = 4

A = np.arange(n**2).reshape(n,n)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
seed_factor = 1234

#A_ij = split_matrix(A, comm)

A_i = get_A_i_to_column_i( A, comm )
print('rank: ', rank, 'A_i: \n', A_i)




'''

B,C,D = multiply( A_ij, n, l, simple_vec, comm )
if rank == 0:
    np.random.seed(seed_factor)
    v = np.random.randn(n,l)
    B_ = v.T @ A @ v
    C_ = A @ v
    D_ = v.T @ A 
    print(B.shape, B_.shape)
    print(C.shape, C_.shape)
    print(D.shape, D_.shape)
    #print(D[:,:3])
    #print(D_[:,:3])
    print('B equal: ', np.allclose(B, B_))
    print('C equal: ', np.allclose(C, C_))
    print('D equal: ', np.allclose(D, D_))
    


A_ij = split_matrix(A, comm)
B,C,D = multiply( A_ij, n, l, gaussian_sketching, comm )
B_,C_,D_ = sketch( A_ij, n, l, seed_factor, comm )

if rank == 0:
    print('B equal: ', np.allclose(B, B_))
    print('C equal: ', np.allclose(C, C_))
    print('D equal: ', np.allclose(D, D_)) 

'''
#omega_i_T, omega_j = gaussian_sketching( n, l, seed_factor, comm )
#print('rank: ', rank, 'Omega_j: \n', omega_j)
#print('rank: ', rank, 'Omega_i_T: \n', omega_i_T)


#A_ij = split_matrix(A, comm)
#print('rank: ', rank, 'A_ij: \n', A_ij)



