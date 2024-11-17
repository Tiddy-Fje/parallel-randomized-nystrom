import os
import bz2
import shutil
import urllib.request

# URLs for downloading the dataset
mnist_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2'
input_file = 'mnist.scale.bz2'
output_file = 'mnist.scale'
output_file_2048 = 'mnist_2048'

# Step 1: Download the file if it does not exist
if not os.path.exists(input_file):
    print(f"Downloading {input_file}...")
    urllib.request.urlretrieve(mnist_url, input_file)
    print(f"Downloaded {input_file}")

# Step 2: Decompress the file
print(f"Decompressing {input_file}...")
with bz2.BZ2File(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print(f"Decompressed to {output_file}")

# Step 3: Extract the first 2048 lines
print(f"Extracting the first 2048 lines from {output_file}...")
with open(output_file, 'r') as f_in:
    with open(output_file_2048, 'w') as f_out:
        for i in range(2048):
            line = f_in.readline()
            if not line:
                break
            f_out.write(line)
print(f"Extracted lines saved to {output_file_2048}")