# ===================== IMPORTS ===================== #
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ===================== CONFIG ===================== #
def init_params():
    # Initialize the weights ("opinions") for the first layer of artificial neurons. The formula for the first layer of neurons will be Z = W * A + B
    # In this scenario, W must be 10 x 784 (since 10 was chosen as the number of neurons in the hidden layer) cause A = 784 x 1000 
    W1 = (np.random.rand(10, 784) - 0.5) * 0.01 # Attempt to ressuscitate the Dying ReLU
    b1 = np.random.rand(10, 1) - 0.5

    # print("The shape of W1 is: ", np.shape(W1))
    # print("The shape of b is: ", np.shape(b1))

    # Initialize the weights and bias for the seceond layer of artificial neurons
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    safe_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))

    return safe_Z / np.sum(safe_Z, axis=0, keepdims=True)

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
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True) # Calculate the effect of each bias as equal, being the average of each error from the second layer
    
    # ============ HIDDEN LAYER ============ #
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1) # Goes back to layer 1, then multiplies with the ReLU derivative in order to take into account how strong was a neuron's activation fro mthe 1st layer

    # ============ ROLLBACK FROM HIDDEN TO INPUT LAYER ============ #
    dW1 = (1 / m) * dZ1.dot(X.T) # Reverses the chain rule in order to obtain the "influence" of each weight from the second layer
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True) # Calculate the effect of each bias as equal, being the average of each error from the second layer


    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    # print("The shape of b before update is: ", np.shape(b1))
    # print("The shape of db1 before update is: ", np.shape(db1))
    # print(b1)
    # print(db1)

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # print("The shape of b after update is: ", np.shape(b1))
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, learning_rate):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        # forward prop
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        
        # backwards prop
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W2, X, Y)

        # update
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # print accuracy from time to time
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
            print(f"Avg dW1 = {np.mean(np.abs(dW1))}")
            print(f"Avg dW2 = {np.mean(np.abs(dW2))}")

    if np.isnan(dW1).any():
        print("Warning: NaNs detected in W1!")

    if np.isnan(dW2).any():
        print("Warning: NaNs detected in W2!")

    return W1, b1, W2, b2
      
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

# Start gradient descent

print("The shape of X_training is: ", np.shape(X_train))
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.01)