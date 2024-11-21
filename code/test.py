import numpy as np
import scipy.sparse as sp
import bz2

# load the sparse matrix from disk
#sparse_matrix = sp.load_npz('../data/mnist.npz')

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



