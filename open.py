# This program simply opens a chosen file

import numpy as np
from collections import Counter

# current set file
file_path = 'X_train.npy'

# loads the file and stores it in data
data = np.load(file_path)

print(data) # print the data
print(f"Data shape: {data.shape}") # prints the shape of the data

# Data shape is usually (number of elements, width, height, channels)
# 3 channels means rgb support, 1 channel means only grayscale
# if the loaded file is labes and not images (e.g y_train), then the shape is just (number of elements)
