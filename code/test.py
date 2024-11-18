import numpy as np
import scipy.sparse as sp
import bz2

def parse_MNIST(file_path):
    '''
    Parse the MNIST dataset in LIBSVM format and construct a sparse matrix.
    This was obtained with github copilot and then tested + adapted. 
    '''

    n_features = 784  # MNIST features
    # Initialize storage for sparse matrix components
    data = []
    row_indices = []
    col_indices = []
    
    # Keep track of the current row index
    current_row = 0
    
    # Read the dataset line by line
    with bz2.BZ2File(file_path, "r") as source_file:
        for line in source_file:
            # Decode and split the line
            elements = line.decode("utf-8").strip().split()
            
            # Process feature-value pairs (skip the label)
            for fv_pair in elements[1:]:
                col_idx, value = map(int, fv_pair.split(":"))
                row_indices.append(current_row)
                col_indices.append(col_idx)
                data.append(value)
            
            # Move to the next row
            current_row += 1

    # Construct the sparse matrix in one step
    n_samples = current_row  # The total number of rows processed
    return sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_features))

# Parameters
train_file = '../data/mnist.bz2'
# Parse and construct the sparse matrix
sparse_matrix = parse_MNIST(train_file)
print(f"Constructed sparse matrix with shape {sparse_matrix.shape} and {sparse_matrix.nnz} non-zero elements")
sp.save_npz('../data/mnist_train.npz', sparse_matrix)
test_file = '../data/mnist.t.bz2'
# Parse and construct the sparse matrix
sparse_matrix = parse_MNIST(test_file)
print(f"Constructed sparse matrix with shape {sparse_matrix.shape} and {sparse_matrix.nnz} non-zero elements")
sp.save_npz('../data/mnist_test.npz', sparse_matrix)

# load the sparse matrix from disk
#sparse_matrix = sp.load_npz('../data/mnist.npz')

'''

B,C,D = multiply( A_ij, n, l, simple_vec, comm )
if rank == 0:
    np.random.seed(seed_factor)
    v = np.random.randn(n,l)
    B_ = v.T @ A @ v
    C_ = A @ v
    D_ = v.T @ A 
    print(B.shape, B_.shape)
    print(C.shape, C_.shape)
    print(D.shape, D_.shape)
    #print(D[:,:3])
    #print(D_[:,:3])
    print('B equal: ', np.allclose(B, B_))
    print('C equal: ', np.allclose(C, C_))
    print('D equal: ', np.allclose(D, D_))
    


A_ij = split_matrix(A, comm)
B,C,D = multiply( A_ij, n, l, gaussian_sketching, comm )
B_,C_,D_ = sketch( A_ij, n, l, seed_factor, comm )

if rank == 0:
    print('B equal: ', np.allclose(B, B_))
    print('C equal: ', np.allclose(C, C_))
    print('D equal: ', np.allclose(D, D_)) 

'''
#omega_i_T, omega_j = gaussian_sketching( n, l, seed_factor, comm )
#print('rank: ', rank, 'Omega_j: \n', omega_j)
#print('rank: ', rank, 'Omega_i_T: \n', omega_i_T)


#A_ij = split_matrix(A, comm)
#print('rank: ', rank, 'A_ij: \n', A_ij)



