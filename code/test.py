import numpy as np
from scipy.linalg import hadamard
from sequential import block_SRHT_bis, sequential_gaussian_sketch
import time
import matplotlib.pyplot as plt
from cProfile import Profile

profile = False
if profile:
    # profile the execution of block_SRHT_bis for n=2**10
    n = 2**11
    l = 2**8
    seed_factor = 42
    A = np.random.randn(n,n)
    profile = Profile()
    profile.enable()
    block_SRHT_bis( A, n, l, seed_factor )
    profile.disable()

    # print the profile results
    profile.print_stats()
else:

    l = 2**8
    seed_factor = 42
    ns = np.geomspace(2**9, 1.3*2**13, num=7, dtype=int)
#    ns = [2**i for i in range(8, 13)]
    times = []
    times_ = []
    times__ = []
    for n in ns:
        A = np.random.randn(n,n)
        start = time.time()
        B, C = sequential_gaussian_sketch( A, n, l, seed_factor )
        #B, C = block_SRHT_bis( A, n, l, seed_factor )

        end = time.time()
        times.append(end-start)
        #H = hadamard(n) / np.sqrt( n )
        #start = time.time()
        #C = H @ A
        #end = time.time()
        #times_.append(end-start)
       # start = time.time()
       # B__=fwht_mat_bis(A, copy=True)
       # end = time.time()
        #times__.append(end-start)
       # assert np.allclose(B, B__), 'Bs not equal'
        #print(B[:5,:5], '\n', B__[:5,:5])

    fig, ax = plt.subplots()
    ax.plot(ns, times, label='FWHT', marker='o')
    print('Slope', np.polyfit(np.log(ns[2:]), np.log(times[2:]), 1)[0])
    #ax.plot(ns, times_, label='Hadamard')
#    ax.plot(ns, times__, label='FWHT bis')
    ax.set_xlabel('n')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Runtime [s]')
    ax.legend()
    plt.show()



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



