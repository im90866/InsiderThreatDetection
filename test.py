import pandas as pd
import numpy as np

file_path = "D:/Datasets/r6.2/http.csv"
chunksize = 10 ** 6  # Adjust this value based on your system's available memory

# Read the file in chunks
data_iterator = pd.read_csv(file_path, chunksize=chunksize)

# Process each chunk and store the results
results = []
for chunk in data_iterator:
    # Perform your data processing on the chunk
    chunk['insider'] = np.where(chunk['url'].str.contains('wikileaks.org'), 1, 0)
    # results.append(result)
    result = chunk[chunk['url'].str.contains('wikileaks.org')]
    print(result)

# Combine the results
# final_result = pd.concat(results)