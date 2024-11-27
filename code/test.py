import numpy as np
from scipy.linalg import hadamard
from sympy.discrete import fwht
import time
import matplotlib.pyplot as plt

def fwht_mat(A, copy=False): # adapted from wikipedia page
    '''Apply hadamard matrix using in-place Fast Walsh–Hadamard Transform.'''
    h = 1
    m1 = A.shape[0]
    if copy:
        A = np.copy(A)
    while h < m1:
        for i in range(0, m1, h * 2):
            for j in range(i, i + h):
                A[j,:], A[j + h,:] = A[j,:] + A[j + h,:], A[j,:] - A[j + h,:]
        h *= 2
    if copy:
        return A
    
ns = [2**i for i in range(6, 13)]
times = []
times_ = []
for n in ns:
    A = np.random.randn(n,n)
    H = hadamard(n) / np.sqrt( n )
    start = time.time()
    fwht_mat(A)
    end = time.time()
    times.append(end-start)
    start = time.time()
    C = H @ A
    end = time.time()
    times_.append(end-start)

fig, ax = plt.subplots()
ax.plot(ns, times, label='FWHT')
ax.plot(ns, times_, label='Hadamard')
ax.set_xlabel('n')
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



