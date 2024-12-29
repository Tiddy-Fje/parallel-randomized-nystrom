import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import scipy.sparse as sp
import plot

# Define synthetic_matrix function
def synthetic_matrix(n, r, decay_rate, decay_type):
    ones = np.ones(r)
    others = np.arange(1, n - r + 1)
    if decay_type == 'exponential':
        if decay_rate == 'fast':
            decay = 10.0 ** (-1.0 * others)
        elif decay_rate == 'medium':
            decay = 10.0 ** (-0.25 * others)
        elif decay_rate == 'slow':
            decay = 10.0 ** (-0.1 * others)
    elif decay_type == 'polynomial':
        if decay_rate == 'fast':
            decay = (others + 1) ** -2.0
        elif decay_rate == 'medium':
            decay = (others + 1) ** -1.0
        elif decay_rate == 'slow':
            decay = (others + 1) ** -0.5
    else:
        raise ValueError("Invalid decay_type")
    return np.diag(np.concatenate([ones, decay]))

# Define MNIST_matrix function
def MNIST_matrix(n, c=10):
    """
    Generate an RBF kernel matrix for MNIST data.
    This assumes the MNIST dataset is stored as a sparse matrix in '../data/mnist_train.npz'.
    """
    mat = sp.load_npz('../data/mnist_train.npz')
    return rbf(mat[:n, :].toarray(), c)

def rbf(data, c):
    data_norm = np.sum(data ** 2, axis=1)
    return np.exp(-(1 / c ** 2) * (data_norm[:, None] + data_norm[None, :] - 2 * np.dot(data, data.T)))

def plot_synthetic_spectra_and_mnist(n, r):
    """
    Plot the spectra of synthetic matrices for different decay rates and types
    alongside MNIST in a single plot with clean styling.
    """
    decay_rates = ['fast', 'medium', 'slow']
    decay_types = ['exponential', 'polynomial']
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown']
    markers = ['o', 's']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot synthetic spectra
    for i, decay_rate in enumerate(decay_rates):
        for j, decay_type in enumerate(decay_types):
            A = synthetic_matrix(n, r, decay_rate, decay_type)
            eigs = np.diag(A)
            ax.plot(
                np.arange(1, len(eigs) + 1), eigs,
                label=f'{decay_type}, {decay_rate}',
                color=colors[i * len(decay_types) + j],
                marker=markers[j], markersize=4, linestyle='-'
            )

    

    # Formatting the plot
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title("Spectra of Synthetic Matrices", fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('../figures/synthetic_spectra.png')
    plt.show()

# Call the function
plot_synthetic_spectra_and_mnist(n=35, r=5)

def plot_mnist_singular_values(n):
    """
    Plot the singular values of the MNIST dataset in a similar style to the synthetic plot.
    """
    mnist_matrix = MNIST_matrix(n)
    singular_values = np.linalg.svd(mnist_matrix, compute_uv=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        np.arange(1, len(singular_values) + 1), singular_values,
        label="MNIST", color='darkblue', linestyle='--', marker='o', markersize=4
    )

    # Formatting the plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Index (log scale)", fontsize=12)
    ax.set_ylabel("Singular value (log scale)", fontsize=12)
    ax.set_title("Singular Values of MNIST Dataset", fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('../figures/mnist_singular_values_darkblue.png')
    plt.show()

# Call the function
plot_mnist_singular_values(n=100)
