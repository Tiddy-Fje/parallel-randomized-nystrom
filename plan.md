## General tasks
- [ ] email professor about composition of our group by December 3, 2024
- [ ] decide work repartition 
- [ ] ...

## Work repartition reflectio
* the numerical stability and performance analysis require a working implementation
* a working implementation requires
  * data pre-processing (potential normalisation, $A$ construction etc.) 
  * coding of the sequential algorithm ??
    * if we want to do it separately from the parallel one
    * could help have something to compare the parallel one with for debugging
    * k-rank approx too 
  * coding of the parallel algorithm 
    * this is probably (in Tara's opinion) the single longest part 
    * someone could work at this (on the synthetic dataset) while the other works out 
      * how to use the real data efficiently 
      * deals with pre-processing 
      * potentially starts to describe the algorithm in the report ??   
* once this first batch of tasks are done, we could split the stability and performance analysis, as well as the report writing ??? 

## Goal and context 
* this  project should allow you to identify a randomized algorithm that is numerically stable for the considered data sets and scales reasonably well in parallel.
* we limit ourselves to rank-k truncation of $A_{Nyst}$
* we should use a synthetic dataset (described in a paper), as well as a real dataset (MNIST or YearPredictionMSD)
  * the description of the synthetic dataset is found in section 5 of the paper that is downloaded
  * MNIST is 70'000 times 780 = 54'600'000
  * YearPredictionMSD is (463'715+51,630) times	90 = 46'381'050
  * YearPredictionMSD seems badly compressed (really heavy files, >200MB), while MNIST is ~15MB, reason for which Tara only downloaded MNIST for now 
* the $n\times n$ matrix $A$ on which we work is obtained by applying the radial basis function $e^{-\|x_i-x_j\|^2/c^2}$ from $n$ rows of input data. The parameter $c$ should be varied and can be chosen as 100 for the MNIST dataset and $10^4$ as well as $10^5$ for the YearPredictionMSD dataset. The dimension $n$ should be taken depending on what your code can support in terms of memory consumption

## Useful tools
* A Julia code for generating the $A$ matrix from the  data can be found for example at https://github.com/matthiasbe/block_srht as well as potentially other useful codes for the project. Some of those matrices will be generated and used during the exercise sessions.
* we can rely on optimized libraries for operations as matrix-vector multiplication, matrix-matrix multiplication, Cholesky factorization, sequential Householder QR, eigenvalue decomposition or singular value decomposition, Walsh-Hadamard transform

## Technical tasks
### Algorithm description 
* should present 
  * the randomized Nyström low rank approximation algorithm considered in the project and the algebra on which the algorithm relies. 
  * a short description of the oblivious subspace embedding property and the sketching matrices $\Omega$ used in the project : Gaussian and block SRHT

### Numerical stability investigation 
* for both data-sets 
* what should we provide ?
  * graphs displaying the error of the low rank approximation in terms of trace norm
    * should actually use $\|A-[A_{Nyst}]_k\|_\star / \|A\|_\star$ to ease interpretation
  * comparison of the accuracy obtained for the two different sketching matrices by taking into consideration the sketching dimension 

### Parallellisation presentation 
* can assume $P=q^2$ to make matrix distribution easier 
  * does she actually mean a perfect square ???
* should :
  * provide the pseudo-code 
  * describe it 

### Performance (runtime) analysis 
* run for cores = 1, ..., 64 and average over 3-5 runs
* should do the comparisons of performance with the two different sketching matrices
* should include :
  * scaling when increasing the number of processors
  * advantages and disadvantages in terms of parallel performance and numerical stability
  * explanation of if we were expecting these results and why
  * if we observe any advantage in using a faster sketching operator with respect to the sketch dimension $l$ that you might need in order to obtain an accurate low rank approximation
    * what is meant by this ???
    * maybe just analysis of how big $l$ needs to be in both cases ???