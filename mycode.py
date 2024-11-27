from mpi4py import MPI

comm = MPI.COMM_WORLD
print('Hi from rank', rank)
