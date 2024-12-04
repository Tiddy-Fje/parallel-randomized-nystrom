from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import math
from scipy.linalg import cholesky, svd, solve_triangular, hadamard
import matplotlib.pyplot as plt
from data_generation import synthetic_matrix
from sequential import *
from parallel import rank_k_approx


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
            A_nystrom=rank_k_approx( B, C, n, k )  
            rel_error = np.linalg.norm(A - A_nystrom, "nuc") / normAnuc
            errors.append(rel_error)
        
    return errors

# Generate and plot results
def generate_plots(A, n, ls, ks, methods, method_names, dataset_name, output_dir):
    normAnuc = np.linalg.norm(A, "nuc")
    plt.figure(figsize=(10, 6))

    for method, method_name in zip(methods, method_names):
        for l in ls:
            print(f"Processing {method_name} with l={l}...")
            if method_name == "Gaussian Sketch":
                B,C = method(A, n, l, seed_factor=1234)  # Use seed_factor
            else:
                B,C = method(A, n, l, random_seed=1234)  # Use random_seed
            errors = compute_relative_error(A,B,C, ks, normAnuc)
            plt.plot(ks[ks<=l], errors, marker="o", label=f"{method_name}, l={l}")

    plt.yscale("log")
    plt.xlabel("Approximation rank (k)")
    plt.ylabel("Relative error (nuclear norm)")
    plt.title(f"Numerical Stability - {dataset_name}")
    plt.legend()
    plt.savefig(f"{output_dir}/{dataset_name}_stability.png")
    plt.close()


if __name__ == "__main__":
    
    log2_l_min, log2_l_max = 6, 10
    log2_n_min, log2_n_max = 10, 13
    ls = 2 ** np.arange(log2_l_min, log2_l_max + 1).astype(int)
    ks = np.array([50, 100, 200, 400, 600, 800, 1000])
    n = 2 ** 10
    output_dir = "./results"

    
    datasets = [
        ("ExpDecay", synthetic_matrix(n, n // 4, "fast", "exponential")),
        ("PolyDecay", synthetic_matrix(n, n // 4, "fast", "polynomial")),
        
    ]

   
    methods = [
        sequential_gaussian_sketch,
        block_SRHT_bis,
    ]
    method_names = ["Gaussian Sketch", "Block SRHT Bis"]

    
    for dataset_name, A in datasets:
        print(f"Processing dataset: {dataset_name}...")
        generate_plots(A, A.shape[0], ls, ks, methods, method_names, dataset_name, output_dir)
