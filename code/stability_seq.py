from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import plot
import matplotlib.pyplot as plt
from data_generation import synthetic_matrix
from sequential import *
from parallel import seq_rank_k_approx


def fwht_mat(X):
    """
    Perform a Fast Walsh-Hadamard Transform (FWHT) using scipy's Hadamard matrix.
    """
    n, m = X.shape
    assert (n & (n - 1)) == 0, "Number of rows must be a power of 2."
    H = hadamard(n)
    return H @ X


def compute_relative_error(A, B,C, ks, normAnuc):
    errors = []
    n=len(A)
    for k in ks:
        if k <= len(B):
            # use the one from in parallel
            A_nystrom=seq_rank_k_approx( B, C, n, k )  
            rel_error = np.linalg.norm(A - A_nystrom, "nuc") / normAnuc
            errors.append(rel_error)
        
    return errors

# Generate and plot results
def generate_plots(A, n, ls, ks, methods, method_names, dataset_name, ax=None):
    normAnuc = np.linalg.norm(A, "nuc")
     
    flag = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        flag = True

    for method, method_name in zip(methods, method_names):
        for l in ls:
            print(f"Processing {method_name} with l={l}...")
            B,C = method(A, n, l, random_seed=1234)  # Use random_seed
            errors = compute_relative_error(A,B,C, ks, normAnuc)
            ax.plot(ks[ks<=l], errors, marker="o", label=f"{method_name}, l={l}")

    ax.set_yscale("log")
    ax.set_xlabel("Approximation-Rank")
    ax.set_ylabel("Relative error (nuclear norm)")
    ax.set_title(f"{dataset_name}")
    ax.legend()
    
    if flag:
        plt.savefig(f"../figures/{dataset_name}_stability.png")


if __name__ == "__main__":
    n = 2 ** 10
    r = 20
   
    methods = [sequential_gaussian_sketch,block_SRHT_bis]
    method_names = ["Gaussian", "BSRHT"]
  
    A_exp_fast = synthetic_matrix(n, r, "fast", "exponential")
    A_exp_slow = synthetic_matrix(n, r, "slow", "exponential")
    ls_exp_fast = [20, 30, 35, 36]
    ls_exp_slow = [30, 70, 161, 162]
    ks_exp_fast = np.arange( 1, 40, 2 ).astype(int) 
    ks_exp_slow = np.concatenate( (np.linspace( 1, r+15, 10).astype(int),np.linspace( r+15, r+80, 10+1).astype(int)[1:] ) )

    print(f"Processing Exp dataset...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    generate_plots(A_exp_slow, A_exp_slow.shape[0], ls_exp_slow, ks_exp_slow, methods, method_names, 'Slow Exp. Decay', ax=ax[0])  
    generate_plots(A_exp_fast, A_exp_fast.shape[0], ls_exp_fast, ks_exp_fast, methods, method_names, 'Fast Exp. Decay', ax=ax[1])
    plt.savefig(f"../figures/exp_decay_stability.png")

    A_poly_fast = synthetic_matrix(n, r, "fast", "polynomial")
    A_poly_slow = synthetic_matrix(n, r, "slow", "polynomial")
    ls_poly = [30, 100, 300]
    ks_poly =  np.concatenate( (np.linspace( 1, r+15, 10).astype(int),np.linspace( r+15, r+150, 10+1).astype(int)[1:] ) )

    print(f"Processing Poly dataset...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    generate_plots(A_poly_slow, A_poly_slow.shape[0], ls_poly, ks_poly, methods, method_names, 'Slow Poly. Decay', ax=ax[0])  
    generate_plots(A_poly_fast, A_poly_fast.shape[0], ls_poly, ks_poly, methods, method_names, 'Fast Poly. Decay', ax=ax[1]) 
    plt.savefig(f"../figures/poly_decay_stability.png")
    
    print(f"Processing MNIST dataset...")
    A_mnist = MNIST_matrix(n)
    generate_plots(A_mnist, A_mnist.shape[0], ls_poly, ks_poly, methods, method_names, 'MNIST')      
