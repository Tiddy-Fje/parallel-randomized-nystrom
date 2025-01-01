from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import plot # for plotting settings to activate
import matplotlib.pyplot as plt
from data_generation import synthetic_matrix
from sequential import *
from parallel import seq_rank_k_approx

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

    # Define methods and method names
    methods_gaussian = [sequential_gaussian_sketch]
    methods_bsrht = [block_SRHT_bis]
    method_names_gaussian = ["Gaussian"]
    method_names_bsrht = ["BSRHT"]

    # Exponential Decay
    A_exp_fast = synthetic_matrix(n, r, "fast", "exponential")
    A_exp_slow = synthetic_matrix(n, r, "slow", "exponential")

    # BSRHT-specific parameters for exponential decay
    ls_exp_fast_bsrht = [15,20,25, 37]
    ls_exp_slow_bsrht = [15,20,25,170]

    # Gaussian-specific parameters for exponential decay
    ls_exp_fast_gaussian = [20, 25, 37]
    ls_exp_slow_gaussian = [30, 70, 75, 160,170]

    ks_exp_fast = np.arange(1, 40, 2).astype(int)
    ks_exp_slow = np.concatenate(
        (np.linspace(1, r + 15, 10).astype(int),
         np.linspace(r + 15, r + 80, 10 + 1).astype(int)[1:])
    )

    # Plot Slow Exponential Decay
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    generate_plots(A_exp_slow, A_exp_slow.shape[0], ls_exp_slow_gaussian, ks_exp_slow, 
                   methods_gaussian, method_names_gaussian, 'Slow Exp. Decay (Gaussian)', ax=ax[0])
    generate_plots(A_exp_slow, A_exp_slow.shape[0], ls_exp_slow_bsrht, ks_exp_slow, 
                   methods_bsrht, method_names_bsrht, 'Slow Exp. Decay (BSRHT)', ax=ax[0])

    # Plot Fast Exponential Decay
    generate_plots(A_exp_fast, A_exp_fast.shape[0], ls_exp_fast_gaussian, ks_exp_fast, 
                   methods_gaussian, method_names_gaussian, 'Fast Exp. Decay (Gaussian)', ax=ax[1])
    generate_plots(A_exp_fast, A_exp_fast.shape[0], ls_exp_fast_bsrht, ks_exp_fast, 
                   methods_bsrht, method_names_bsrht, 'Fast Exp. Decay (BSRHT)', ax=ax[1])

    plt.tight_layout()
    plt.savefig(f"../figures/exp_decay_stability_custom.png")

    # Polynomial Decay
    A_poly_fast = synthetic_matrix(n, r, "fast", "polynomial")
    A_poly_slow = synthetic_matrix(n, r, "slow", "polynomial")
    ls_poly = [30, 100, 300]
    ks_poly = np.concatenate(
        (np.linspace(1, r + 15, 10).astype(int),
         np.linspace(r + 15, r + 150, 10 + 1).astype(int)[1:])
    )

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    generate_plots(A_poly_slow, A_poly_slow.shape[0], ls_poly, ks_poly, methods_gaussian + methods_bsrht,
                   method_names_gaussian + method_names_bsrht, 'Slow Poly. Decay', ax=ax[0])
    generate_plots(A_poly_fast, A_poly_fast.shape[0], ls_poly, ks_poly, methods_gaussian + methods_bsrht,
                   method_names_gaussian + method_names_bsrht, 'Fast Poly. Decay', ax=ax[1])

    plt.tight_layout()
    plt.savefig(f"../figures/poly_decay_stability_custom.png")

    # MNIST Dataset
    A_mnist = MNIST_matrix(n)
    generate_plots(A_mnist, A_mnist.shape[0], ls_poly, ks_poly, methods_gaussian + methods_bsrht,
                   method_names_gaussian + method_names_bsrht, 'MNIST')
