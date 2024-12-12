import parallel_matrix as pm
from mpi4py import MPI
import numpy as np
import time as time
# init mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

A = np.random.randn(8,3)
A_l = pm.row_distrib_mat(A, comm)

if rank == 0:
    print(A)
time.sleep(1)
print(f'rank={rank}, A_l={A_l}')