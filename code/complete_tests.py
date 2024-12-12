# what to test
# - data generation
# - sketching matrices (can test with simple approx for vectors ??) 
# - k-rank approx
# problem is only there for singular matrices  
import numpy as np
import parallel_matrix as pm
import parallel as par
import sequential as seq
from mpi4py import MPI
import data_generation as dg


def test_sketching( A, l, sketch_function, comm ):
    rank = comm.Get_rank()
    A_ij = pm.split_matrix( A, comm )

    if size > 1:
        B, C = sketch_function( A_ij, n, l, 12345, comm)
    else:
        B, C = sketch_function( A, n, l, 12345 )
    
    if rank == 0:
        A_nyst = C @ np.linalg.pinv(B) @ C.T
        print( np.linalg.norm( A - A_nyst, 'nuc' ) / np.linalg.norm(A, 'nuc') )


def test_nystrom( A, l, k, sketch_function, comm ):
    rank = comm.Get_rank()
    A_ij = pm.split_matrix( A, comm )

    if size > 1:
        B, C = sketch_function( A_ij, n, l, 12345, comm)
        A_k = par.rank_k_approx( B, C, n, k, comm )
    else:
        B, C = sketch_function( A, n, l, 12345 )
        A_k = par.seq_rank_k_approx( B, C, n, k, alternative=False )
    
    if rank == 0:
        A_nyst = C @ np.linalg.pinv(B) @ C.T
        print(np.linalg.norm(A_nyst, 'nuc'))
        print(np.linalg.norm(A_k, 'nuc'))

        #print( np.linalg.norm( A_k - A_nyst, 'nuc' ) / np.linalg.norm(A_nyst, 'nuc') )

comm = MPI.COMM_WORLD
size = comm.Get_size()
n = 2 ** 11
r = 3 * n // 4
l = n // 4
k = r // 2

A = dg.synthetic_matrix(n, r, 'fast','exponential') + 0.0 * np.diag(np.ones(n))
#A = dg.synthetic_matrix(n, r, 'fast','polynomial') + 0.0 * np.diag(np.ones(n))
A2 = dg.MNIST_matrix(n, 10)

if size > 1:
    #test_sketching( A, l, par.SRHT_sketching, comm )
    test_nystrom( A, l, k, par.SRHT_sketching, comm )
else:
    #test_sketching( A, l, seq.block_SRHT_bis, comm )
    test_nystrom( A, l, k, seq.block_SRHT_bis, comm )


'''

m = 10000
AA = dg.synthetic_matrix(m, m-1000, 'fast','polynomial') 
print(np.linalg.matrix_rank(AA))

'''
