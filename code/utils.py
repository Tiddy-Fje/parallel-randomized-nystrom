import numpy as np


def synthetic_data( n, r, decay_rate, decay_type ) : 
    ones = np.ones(r)
    others = np.arange(1,n-r+1)

    decay = np.empty(others.shape)
    if decay_type == 'exponential' :
        if decay_rate == 'fast' :
            decay_rate = 1.0
        elif decay_rate == 'slow' :
            decay_rate = 0.1
        elif decay_rate == 'medium' :
            decay_rate = 0.25
        decay = 10.0 ** ( -decay_rate*others )
    elif decay_type == 'polynomial' :
        if decay_rate == 'fast' :
            decay_rate = 2.0
        elif decay_rate == 'slow' :
            decay_rate = 0.5
        elif decay_rate == 'medium' :
            decay_rate = 1.0
        decay = (others+1)**(-decay_rate)

    A = np.diag( np.concatenate((ones, decay)) )
    return A

def fwht_mat(A) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a."""
    h = 1
    m1, m2 = A.shape
    while h < m1:
        # perform FWHT
        for i in range(0, m1, h * 2):
            for j in range(i, i + h):
                x = A[j,:]
                y = A[j + h,:]
                A[j,:] = x + y
                A[j + h,:] = x - y
        # normalize and increment
        A /= np.sqrt(2)
        h *= 2

def fwht(a) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a."""
    h = 1
    while h < len(a):
        # perform FWHT
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        # normalize and increment
        a /= np.sqrt(2)
        h *= 2

if __name__ == '__main__':
    np.random.seed(10)
    x = np.random.rand(8)
    print(3**x)