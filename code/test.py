from parallel_matrix import full_multiply
from mpi4py import MPI
import numpy as np
# init mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

A = np.random.randn(4,4)
B = np.random.randn(4,3)

C = full_multiply( A, B, comm )
print(f'rank={rank}, C={C}')
if rank == 0:
    assert np.allclose( C, A @ B ), 'C is not A@B'