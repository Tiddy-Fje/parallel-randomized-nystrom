from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from data_generation import synthetic_matrix, MNIST_matrix
from sequential import sequential_gaussian_sketch, seq_rank_k_approx
from parallel import block_SRHT_bis
from scipy.linalg import hadamard

def fwht_mat(X):
    """
    Perform a Fast Walsh-Hadamard Transform (FWHT) using scipy's Hadamard matrix.
    """
    n, m = X.shape
    assert (n & (n - 1)) == 0, "Number of rows must be a power of 2."
    H = hadamard(n)
    return H @ X

def compute_relative_error(A, B, C, ks, normAnuc):
    """
    Compute relative errors for rank-k approximations.
    """
    errors = []
    n = len(A)
    for k in ks:
        if k <= len(B):
            A_nystrom = seq_rank_k_approx(B, C, n, k)
            rel_error = np.linalg.norm(A - A_nystrom, "nuc") / normAnuc
            errors.append(rel_error)
    return errors

def generate_plots(A, n, ls_dict, ks, methods, method_names, dataset_name, ax=None):
    """
    Generate plots for stability analysis with method-specific `ls` values.
    """
    normAnuc = np.linalg.norm(A, "nuc")
    flag = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        flag = True

    for method, method_name in zip(methods, method_names):
        ls = ls_dict[method_name]
        for l in ls:
            print(f"Processing {method_name} with l={l}...")
            B, C = method(A, n, l, random_seed=1234)  # Use fixed random_seed for reproducibility
            errors = compute_relative_error(A, B, C, ks, normAnuc)
            ax.plot(ks[ks <= l], errors, marker="o", label=f"{method_name}, l={l}")

    ax.set_yscale("log")
    ax.set_xlabel("Approximation Rank")
    ax.set_ylabel("Relative Error (Nuclear Norm)")
    ax.set_title(dataset_name)
    ax.legend()
    if flag:
        plt.savefig(f"../figures/{dataset_name}_stability.png")

if __name__ == "__main__":
    n = 2 ** 10
    r = 20
    methods = [sequential_gaussian_sketch, block_SRHT_bis]
    method_names = ["Gaussian", "BSRHT"]

    # Adjust `ls` and `ks` values for Gaussian and BSRHT
    ls_bsrht_exp = [50, 100, 150, 200]  # Increased range for better performance
    ks_bsrht_exp = np.concatenate(
        (np.linspace(1, r + 50, 20).astype(int), np.linspace(r + 50, r + 200, 20).astype(int)[1:])
    )

    ls_bsrht_poly = [50, 120, 250, 400]  # Higher values for polynomial decay
    ks_bsrht_poly = np.concatenate(
        (np.linspace(1, r + 30, 15).astype(int), np.linspace(r + 30, r + 300, 15).astype(int)[1:])
    )

    datasets = {
        "Exp. Decay (Fast)": (
            synthetic_matrix(n, r, "fast", "exponential"),
            {"Gaussian": [20, 30, 35, 36], "BSRHT": ls_bsrht_exp},
            np.arange(1, 40, 2),
        ),
        "Exp. Decay (Slow)": (
            synthetic_matrix(n, r, "slow", "exponential"),
            {"Gaussian": [30, 70, 161, 162], "BSRHT": ls_bsrht_exp},
            ks_bsrht_exp,
        ),
        "Poly Decay (Fast)": (
            synthetic_matrix(n, r, "fast", "polynomial"),
            {"Gaussian": [30, 100, 300], "BSRHT": ls_bsrht_poly},
            ks_bsrht_poly,
        ),
        "Poly Decay (Slow)": (
            synthetic_matrix(n, r, "slow", "polynomial"),
            {"Gaussian": [30, 100, 300], "BSRHT": ls_bsrht_poly},
            ks_bsrht_poly,
        ),
        "MNIST": (
            MNIST_matrix(n),
            {"Gaussian": [30, 100, 300], "BSRHT": ls_bsrht_poly},
            ks_bsrht_poly,
        ),
    }

    for dataset_name, (A, ls_dict, ks) in datasets.items():
        print(f"Processing {dataset_name} dataset...")
        fig, ax = plt.subplots(figsize=(14, 5.5))
        generate_plots(A, A.shape[0], ls_dict, ks, methods, method_names, dataset_name, ax=ax)
        plt.tight_layout()
        plt.savefig(f"../figures/{dataset_name.replace(' ', '_').lower()}_stability.png")
