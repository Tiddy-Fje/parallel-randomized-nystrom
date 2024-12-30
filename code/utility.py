import numpy as np
import csv
import datetime
#from icecream import install, ic # IceCream is a library that makes print debugging easy. It's a single function that logs a variable and its value to the console. It's a great alternative to print debugging. You can find a quick start and a full tutorial on https://github.com/gruns/icecream


def fwht_mat(A, copy=False): # adapted from wikipedia page with help of GPT
    '''Apply hadamard matrix using in-place Fast Walsh–Hadamard Transform.'''
    h = 1
    m1 = A.shape[0]
    if copy:
        A = np.copy(A)
    while h < m1:
        for i in range(0, m1, h * 2):
            A[i:i+h, :], A[i+h:i+2*h, :] = A[i:i+h, :] + A[i+h:i+2*h, :], A[i:i+h, :] - A[i+h:i+2*h, :]
            # this is done to skip the loop (2-3x faster)
        h *= 2
    if copy:
        return A

def get_counter(name="default"):
    return np.load("../data/utilities/counter_" + name + ".npy")

def add_counter(n, name="default"):
    np.save("../data/utilities/counter_" + name + ".npy", np.array(get_counter(name) + n, dtype=int))

def get_settings_from_csv(line_id, file_name=None):
    if file_name is None:
        file_name = "default.csv"  # Replace this with your actual test name retrieval logic
    with open("../testing/" + file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == line_id:
                break
    if len(row[0]) > 0:
        raise Exception("test already executed")
    n = 2 ** int(row[4])
    matrix_type = int(row[5])
    R = int(row[6]) if len(row[6]) > 0 else 0
    p = float(row[7]) if len(row[7]) > 0 else 0
    sigma = int(row[8]) if len(row[8]) > 0 else 0
    l = int(row[9])
    k = int(row[10])
    sketch_matrix = int(row[11])
    t = int(row[12]) if len(row[12]) > 0 else 0
    return n, matrix_type, R, p, sigma, l, k, sketch_matrix, t

def print_settings(n, matrix_type, R, p, sigma, l, k, sketch_matrix, t, n_processors):
    output = f"Settings: n = {n}\nMatrix_type: "
    match matrix_type:
        case 0:
            output += f"PolyDecay, R = {R}, p = {p}\n"
        case 1:
            output += f"ExpDecay, R = {R}, p = {p}\n"
        case 2:
            output += f"A_MNIST, sigma = {sigma}\n"
        case 3:
            output += f"YearPredictionMSD, sigma = {sigma}\n"
    sketch_dict = {0: "SRHT", 1: "short-axis", 2: "gaussian", 3: "block_SRHT"}
    output += f"sketch {sketch_dict[sketch_matrix]}, l = {l}" + (f", t = {t}" if sketch_matrix == 1 else "") + f"\nk = {k}, in {'parallel' if n_processors > 1 else 'sequential'}"
    print(output)

def save_results_to_csv(line_id, n_processors, cholesky_success, random_seed, error_nuc, wt, file_name=None):
    if file_name is None:
        file_name = "default.csv"  # Replace with actual test name logic
    with open("../testing/" + file_name, newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    data[line_id][0] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data[line_id][2] = str(n_processors)
    data[line_id][3] = "par" if n_processors > 1 else "seq"
    data[line_id][13] = 1 if cholesky_success else 0
    data[line_id][14] = str(random_seed)
    data[line_id][15] = str(error_nuc)
    data[line_id][16] = str(wt)
    with open("../testing/" + file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def nuc_norm_A(matrix_type, n, R, p, sigma):
    match matrix_type:
        case 0:
            return np.linalg.norm(A_PolyDecay(n, R, p), ord='nuc')  # Assuming A_PolyDecay is available
        case 1:
            return np.linalg.norm(A_ExpDecay(n, R, p), ord='nuc')
        case 2:
            return np.linalg.norm(A_MNIST(n, sigma), ord='nuc')
        case 3:
            return np.linalg.norm(A_YearPredictionMSD(n, sigma), ord='nuc')

def print_results(error_nuc, wt, cholesky_success, random_seed):
    print(f"error_nuc = {error_nuc}, runtime = {wt}")
    print(f"cholesky {'succeeded' if cholesky_success else 'failed'}, random_seed = {random_seed}")


if __name__ == '__main__':
    s = 16
    k = s - 6
    a = 6 * np.ones((2**s, 2**k))
    print(a.shape)
    
