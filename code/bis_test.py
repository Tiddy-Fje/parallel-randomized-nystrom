import numpy as np
from scipy.linalg import hadamard
from sequential import block_SRHT_bis, sequential_gaussian_sketch
import time
import matplotlib.pyplot as plt
from cProfile import Profile


def mat_prod( A, B ):
    C = A @ B
    return C

def mix_mat_prod( A, B ):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        C[i,:] = A @ B[:,i]
    return C

seed_factor = 42
ns = np.geomspace(2**8, 2**11, num=20, dtype=int)
#    ns = [2**i for i in range(8, 13)]
times = []
times_ = []
times__ = []
for n in ns:
    print('n', n)
    A = np.random.randn(n,n)
    start = time.time()
    C = mat_prod( A, A )
    end = time.time()
    times.append(end-start)
    #H = hadamard(n) / np.sqrt( n )
    start = time.time()
    C = mix_mat_prod( A, A )
    end = time.time()
    times_.append(end-start)
    # start = time.time()
    # B__=fwht_mat_bis(A, copy=True)
    # end = time.time()
    #times__.append(end-start)
    # assert np.allclose(B, B__), 'Bs not equal'
    #print(B[:5,:5], '\n', B__[:5,:5])

fig, ax = plt.subplots()
ax.plot(ns, times, label='quick', marker='o')
print('Slope', np.polyfit(np.log(ns[2:]), np.log(times[2:]), 1)[0])
ax.plot(ns, times_, label='slower', marker='o')
#    ax.plot(ns, times__, label='FWHT bis')
ax.set_xlabel('n')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Runtime [s]')
ax.legend()
plt.show()