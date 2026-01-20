# ===================== IMPORTS ===================== #
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ===================== CONFIG ===================== #
def init_params():
    # Initialize the weights ("opinions") for the first layer of artificial neurons. The formula for the first layer of neurons will be Z = W * A + B
    # In this scenario, W must be 10 x 784 (since 10 was chosen as the number of neurons in the hidden layer) cause A = 784 x 1000 
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    # Initialize the weights and bias for the seceond layer of artificial neurons
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Create an m x 10 matrix. 
    one_hot_Y[np.arange(Y.size), Y] = 1 # For every row i, mark the j'th column with a 1 if Y[i] = j
    one_hot_Y = one_hot_Y.T # Flip the matrix
    return one_hot_Y

def backward_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    # ============ OUTPUT LAYER ============ #
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # Compares the prediction with the actual answer
    
    # ============ ROLLBACK FROM OUTPUT TO HIDDEN LAYER ============ #
    dW2 = (1 / m) * dZ2.dot(A1.T) # Reverses the chain rule in order to obtain the "influence" of each weight from the second layer
    
    #? Why is there a 2 in the sum? 
    db2 = (1 / m) * np.sum(dZ2, 2) # Calculate the effect of each bias as equal, being the average of each error from the second layer
    
    # ============ HIDDEN LAYER ============ #
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1) # Goes back to layer 1, then multiplies with the ReLU derivative in order to take into account how strong was a neuron's activation fro mthe 1st layer

    # ============ ROLLBACK FROM HIDDEN TO INPUT LAYER ============ #
    dW1 = (1 / m) * dZ1.dot(X.T) # Reverses the chain rule in order to obtain the "influence" of each weight from the second layer
    db1 = (1 / m) * np.sum(dZ1, 2) # Calculate the effect of each bias as equal, being the average of each error from the second layer


    return dW1, db1, dW2, db2
      
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

init_params()