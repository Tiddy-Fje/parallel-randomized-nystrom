## README for Randomized Nyström Low-Rank Approximation

### Project Overview
This project was developed as part of the *HPC for Numerical Methods and Data Analysis* course at EPFL. It focuses on the **Randomized Nyström** algorithm for low-rank approximation of symmetric positive semidefinite (SPSD) matrices. The project investigates the numerical stability, scalability, and parallel performance of the algorithm using two sketching methods: **Gaussian Sketching** and **Block Subsampled Randomized Hadamard Transform (BSRHT)**. The goal is to provide efficient low-rank approximations for large-scale matrices, which are essential in applications such as kernel methods, image processing, and principal component analysis (PCA).

See `report/report.pdf` for a detailed analysis and discussion of the project. It covers the theoretical background, implementation details, and results of the numerical stability and runtime investigations.

### How to Run the Analysis
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Set up the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate nystrom
   ```
3. Run the Python scripts to reproduce the experiments and analysis.

### Authors
- Tara Fjellman
- Amal Seddas

For questions or collaborations, please reach out via email.

### Acknowledgments
- **Professor Laura Grigori**: Provided guidance and answered questions related to the project.
- **Original References**: The project builds on the theoretical foundations and algorithms discussed in the course materials and related literature, including the work on randomized algorithms for low-rank matrix approximation.

### Notes
- This project adheres to EPFL guidelines for reproducible research. Ensure all code modifications are documented, and credit is given where due.
- The results and findings are based on numerical experiments conducted on the **Helvetios cluster** at EPFL.

### Key Findings
1. **Numerical Stability**:
   - **Gaussian Sketching** consistently outperforms **BSRHT** in terms of stability across various decay patterns of singular values.
   - Both methods exhibit instability at certain embedding dimensions, particularly when the sketching matrix is not full rank.

2. **Performance**:
   - **Gaussian Sketching** is faster than **BSRHT** for small embedding dimensions $l$ and scales better with the matrix size $n$.
   - **BSRHT** scales better with larger embedding dimensions $l$, making it more suitable for information-dense data.
   - Parallelization of the algorithm was successful, with quasi-linear speed-ups observed for large computations.

3. **Trade-offs**:
   - The choice between Gaussian Sketching and BSRHT involves a trade-off between stability, scalability, and computational efficiency, depending on the specific task and data at hand.

### References
- [1] L. Grigori, 'Randomized algorithms for low-rank matrix approximation'.
- [2] J. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, 'Fixed-Rank Approximation of a Positive-Semidefinite Matrix from Streaming Data'.
- [3] O. Balabanov, M. Beaupere, L. Grigori, and V. Lederer, 'Block Subsampled Randomized Hadamard Transform for Low-Rank Approximation on Distributed Architectures'.