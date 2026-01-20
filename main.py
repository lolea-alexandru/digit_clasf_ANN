# ===================== IMPORTS ===================== #
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ===================== CONFIG ===================== #

data = pd.read_csv('mnist_data/train.csv')

# Convert panda dataframe to numpy array
data = np.array(data)

m, n = data.shape # Retrieve the number of rows (training examples) and columns (features + 1 -> label column)

np.random.shuffle(data)

# Initialize validation data
data_dev = data[0:1000].T # We initially take only the first 1000 elements (and directly transpose them)
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

# Initialize training data
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
