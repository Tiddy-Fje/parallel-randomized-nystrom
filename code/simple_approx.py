import numpy as np
import scipy as sp
from utils import synthetic_data, fwht_mat

def buildA(m, sigma_k1, k = 10):
    U = (1/np.sqrt(m))*sp.linalg.hadamard(m)
    V = (1/np.sqrt(2*m))*sp.linalg.hadamard(2*m)
    firstSig = [sigma_k1**(np.floor(j/2)/5) for j in range(1, k+1)]
    sigmas = firstSig + [sigma_k1*(m - j)/(m - 11) for j in range(k+1, m+1)]
    Sigma = np.zeros((m, 2*m))
    np.fill_diagonal(Sigma, sigmas)
    return U@Sigma@np.transpose(V), sigmas

def randomised_svd( A, k, p ):
    m, n = A.shape
    l = p + k
    omega = np.random.randn(n, l)
    Y = A @ omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    U_tilde, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    print('B shape: ', B.shape)
    #assert np.allclose(A, U @ np.diag(S) @ Vh)
    print('Error: ', np.linalg.norm(A - U @ np.diag(S) @ Vh)/ np.linalg.norm(A))
    return U[:, :k], S[:k], Vh[:k, :]

n = 2**11
r = 2**7
l = 2**7
p = 6
k = 10

A, sigmas = buildA(n, 1.0e-5, k)
#A = np.random.randn(n, n)
mat = synthetic_data( n, r, 'slow', 'polynomial' )
#mat = np.random.randn(n, n)
U, S, Vh = randomised_svd(A[:,:n], k, p)