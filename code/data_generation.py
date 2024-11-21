import numpy as np
import scipy.sparse as sp
import bz2

def synthetic_matrix( n, r, decay_rate, decay_type ) : 
    ones = np.ones(r)
    others = np.arange(1,n-r+1)

    decay = np.empty(others.shape)
    if decay_type == 'exponential' :
        if decay_rate == 'fast' :
            decay_rate = 1.0
        elif decay_rate == 'slow' :
            decay_rate = 0.1
        elif decay_rate == 'medium' :
            decay_rate = 0.25
        decay = 10.0 ** ( -decay_rate*others )
    elif decay_type == 'polynomial' :
        if decay_rate == 'fast' :
            decay_rate = 2.0
        elif decay_rate == 'slow' :
            decay_rate = 0.5
        elif decay_rate == 'medium' :
            decay_rate = 1.0
        decay = (others+1)**(-decay_rate)

    A = np.diag( np.concatenate((ones, decay)) )
    return A

def parse_MNIST_file(file_path):
    '''
    Parse the MNIST dataset in LIBSVM format and construct a sparse matrix.
    This was obtained with github copilot and then tested + adapted. 
    '''

    n_features = 784  # MNIST features
    # Initialize storage for sparse matrix components
    data = []
    row_indices = []
    col_indices = []
    labels = []
    
    # Keep track of the current row index
    current_row = 0
    
    # Read the dataset line by line
    with bz2.BZ2File(file_path, "r") as source_file:
        for line in source_file:
            # Decode and split the line
            elements = line.decode("utf-8").strip().split()
            
            # store the label
            labels.append(int(elements[0]))
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
    return sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_features), dtype=float), np.array(labels)

def save_MNIST(normalize = False):
    '''
    Save the MNIST dataset as a sparse matrix and labels as numpy array.
    '''
    train_file = '../data/mnist.bz2'
    test_file = '../data/mnist.t.bz2'

    sparse_matrix_tr, labels_tr = parse_MNIST_file(train_file)
    sparse_matrix_te, labels_te = parse_MNIST_file(test_file)

    print(f"Constructed sparse matrix with shape {sparse_matrix_tr.shape} and {sparse_matrix_tr.nnz} non-zero elements")
    print(f"Constructed sparse matrix with shape {sparse_matrix_te.shape} and {sparse_matrix_te.nnz} non-zero elements")

    if normalize:
        sparse_matrix_tr /= 255.0
        sparse_matrix_te /= 255.0  

    sp.save_npz('../data/mnist_train.npz', sparse_matrix_tr)
    np.save('../data/mnist_train_labels.npy', labels_tr)
    sp.save_npz('../data/mnist_test.npz', sparse_matrix_te)
    np.save('../data/mnist_test_labels.npy', labels_te)
    
    return


def rbf(data: np.ndarray, c: int, savepath: str = None):
    """
    Generate a radial basis function (RBF) matrix using input data.
    This matrix is generated using the formula exp(-||xi - xj||^2 / c^2).
    """
    data_norm = np.sum(data ** 2, axis=-1)
    # using the fact that ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    A = np.exp( -(1 / c ** 2) * (data_norm[:, None] + data_norm[None, :] - 2 * np.dot(data, data.T)) )
    if savepath is not None:
        assert A.shape[0] < 30000, "Matrix is too large to save"
        np.save(savepath, A)
    return A


def MNIST_matrix( n:int, c:int=10 ):
    mat = sp.load_npz('../data/mnist_train.npz')
    #lab = np.load('../data/mnist_train_labels.npy')
    #idx = np.where(lab == 0)[0]
    #print(A[idx[0],idx[1]], A[idx[0],idx[1]+1])
    return rbf( mat[:n,:].toarray(), c )
    
if __name__ == '__main__':
    # memory (in GB) \approx 7.5 * n**2 / 1e9
    # avoid n > 30000 (equivalent to \approx 7GB)
    n = 1000


'''
def read_yearPredictionMSD(filename: str, size: int = 784, savepath: str = None):
    
    dataR = pd.read_csv(filename, sep=',', header=None)
    n = len(dataR)
    data = np.zeros((n, size))
    labels = np.zeros((n, 1))

    for i in range(n):
        l = dataR.iloc[i, 0]
        labels[i] = int(l[0])
        l = l[6:]
        indices_values = [tuple(map(float, pair.split(':'))) for pair in l.split()]
        indices, values = zip(*indices_values)
        indices = [int(i) for i in indices]
        data[i, indices] = values

    if savepath is not None:
        data.tofile('./denseData.csv', sep=',', format='%10.f')
        labels.tofile('./labels.csv', sep=',', format='%10.f')
'''