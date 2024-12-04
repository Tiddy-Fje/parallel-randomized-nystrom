import numpy as np
from scipy.linalg import hadamard
from sequential import sequential_gaussian_sketch 
import time
import matplotlib.pyplot as plt

# Parameters for the test
l = 2**8
seed_factor = 42
ns = np.geomspace(2**9, 1.3 * 2**13, num=7, dtype=int)  
times = []

# Run the sequential Gaussian sketching for various matrix sizes
for n in ns:
    A = np.random.randn(n, n)  
    start = time.time()
    B, C = sequential_gaussian_sketch(A, n, l, seed_factor)  
    end = time.time()
    times.append(end - start)

# Plot the runtimes
fig, ax = plt.subplots()
ax.plot(ns, times, label="Sequential Gaussian Sketch", marker="o")
slope = np.polyfit(np.log(ns), np.log(times), 1)[0]  
print(f"Slope: {slope}")

# Formatting the plot
ax.set_xlabel("Matrix size (n)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Runtime [s]")
ax.legend()
plt.show()
