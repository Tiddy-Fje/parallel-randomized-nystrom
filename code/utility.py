import numpy as np

def fwht_mat(A, copy=False): # adapted from wikipedia page with help of GPT
    '''Apply hadamard matrix using in-place Fast Walsh–Hadamard Transform.'''
    h = 1
    m1 = A.shape[0]
    if copy:
        A = np.copy(A)
    while h < m1:
        for i in range(0, m1, h * 2):
            A[i:i+h, :], A[i+h:i+2*h, :] = A[i:i+h, :] + A[i+h:i+2*h, :], A[i:i+h, :] - A[i+h:i+2*h, :]
            # this is done to skip the loop (2-3x faster)
        h *= 2
    if copy:
        return A

if __name__ == '__main__':
    print("This is utility.py")
