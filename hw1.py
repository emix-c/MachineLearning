# Libraries Needed
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import pandas as pd

# Pre process data before feeding to Neural Network
def get_clean_data():
  # fetch dataset
  tic_tac_toe_endgame = fetch_ucirepo(id=101)

  # data (as pandas dataframes)
  X = tic_tac_toe_endgame.data.features
  y = tic_tac_toe_endgame.data.targets

  # Replace x, o, b with numerical values
  di = {'x': 0, 'o': 1, 'b': 2}
  X = X.replace(di)

  # Replace class label to numerical values
  di2 = {'positive': 1, 'negative': 0}
  y = y.replace(di2)

  # Split into training and testing data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

  return X_train, X_test, y_train, y_test


class NN():
    # Initialize network: input-hidden-output layer size, activation function, learning rate, epochs, and momentum values
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid', learning_rate=0.01, epochs=50, beta=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta

        # Momentum terms
        self.vdw_hidden = np.zeros((input_size, hidden_size))
        self.vdb_hidden = np.zeros(hidden_size)
        self.vdw_output = np.zeros((hidden_size, output_size))
        self.vdb_output = np.zeros(output_size)

    # Calculate activation functions
    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation function.")

    # Calculate derivative of activation functions
    def deriv(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - x**2
        elif self.activation == 'relu':
            return np.where(x <= 0, 0, 1)
        else:
            raise ValueError("Unsupported activation function.")

    # Initialize model weights randomly and set biases to 0
    def initialize_weights(self):
        self.hidden_weights = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.hidden_biases = np.zeros(self.hidden_size)
        self.output_weights = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.output_biases = np.zeros(self.output_size)

    # Run through network from input -> hidden -> output while applying activations on hidden and output layer outputs
    def forward_pass(self, X):
        self.z1 = np.dot(X, self.hidden_weights) + self.hidden_biases
        self.a1 = self.activate(self.z1)
        self.z2 = np.dot(self.a1, self.output_weights) + self.output_biases
        self.a2 = self.activate(self.z2)
        return self.a2

    # Perform SGD to get weight update deltas (change)
    def backward_pass(self, X, y):
        y = np.array(y, dtype=float)

        output_error = y - self.a2
        output_delta = output_error * self.deriv(self.a2)

        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self.deriv(self.a1)

        return hidden_delta, output_delta

    # Apply momementum optimizer and update network weights
    def update_weights(self, X, hidden_delta, output_delta):
        # Apply SGD with Momentum to updated weights
        self.vdw_hidden = self.beta * self.vdw_hidden + (1 - self.beta) * np.dot(X.T, hidden_delta)
        self.vdb_hidden = self.beta * self.vdb_hidden + (1 - self.beta) * np.sum(hidden_delta, axis=0)
        self.vdw_output = self.beta * self.vdw_output + (1 - self.beta) * np.dot(self.a1.T, output_delta)
        self.vdb_output = self.beta * self.vdb_output + (1 - self.beta) * np.sum(output_delta, axis=0)

        self.hidden_weights += self.learning_rate * self.vdw_hidden
        self.hidden_biases += self.learning_rate * self.vdb_hidden
        self.output_weights += self.learning_rate * self.vdw_output
        self.output_biases += self.learning_rate * self.vdb_output

    # Training loop for n epochs: on each data point, perform front pass, back pass, and update weights
    def fit(self, X_train, y_train):
        # Ensure X_train is a numpy array for easy manipulation
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float).reshape(-1, 1)  # Ensure y_train is correctly shaped

        self.initialize_weights()
        for epoch in range(self.epochs):
            for i in range(len(X_train)):  # Iterate over the index to access each sample
                X = X_train[i].reshape(1, -1)  # Reshape X to have shape (1, input_size)
                y = y_train[i].reshape(1, -1)  # Reshape y to ensure it's a 2D array, even if it's a single value
                self.forward_pass(X)
                hidden_delta, output_delta = self.backward_pass(X, y)
                self.update_weights(X, hidden_delta, output_delta)
            # Optionally print epoch and loss here

    # Given input X, predict Y using current network weights and biases. Convert output to 0 or 1.
    def predict(self, X):
        if X.ndim == 1:  # If a single sample, reshape it
            X = X.reshape(1, -1)
        output = self.forward_pass(X)
        # For tanh
        if self.activation == 'tanh':
          return (output > 0).astype(int)
        # For Sigmoid and ReLu
        else:
          return (output > 0.5).astype(int)

    # Given predictions and true values, run model and calculate accuracy
    def evaluate_accuracy(self, X_test, y_test):
        X_test = np.array(X_test, dtype=float)  # Ensure X_test is a numpy array
        y_test = np.array(y_test, dtype=float).reshape(-1, 1)  # Ensure y_test is correctly shaped
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy

# Runner block
X_train, X_test, y_train, y_test = get_clean_data()

# Specify hyper-parameters that need to be tested
param_grid = [
    {'learning_rate': 0.1, 'activation': 'sigmoid'},
    {'learning_rate': 0.01, 'activation': 'sigmoid'},
    {'learning_rate': 0.001, 'activation': 'sigmoid'},
    {'learning_rate': 0.1, 'activation': 'tanh'},
    {'learning_rate': 0.01, 'activation': 'tanh'},
    {'learning_rate': 0.001, 'activation': 'tanh'},
    {'learning_rate': 0.1, 'activation': 'relu'},
    {'learning_rate': 0.01, 'activation': 'relu'},
    {'learning_rate': 0.001, 'activation': 'relu'}
]

results = []

# Run Neural Network on all combination of hyper-parameters and save train/test accuracy
for params in param_grid:
    model = NN(input_size=X_train.shape[1],
               hidden_size=6,
               output_size=1,
               activation=params['activation'],
               learning_rate=params['learning_rate'],
               epochs=200,
               beta=0.9)

    model.fit(X_train, y_train)

    train_accuracy = model.evaluate_accuracy(X_train, y_train)
    test_accuracy = model.evaluate_accuracy(X_test, y_test)

    results.append({
        'Learning Rate': params['learning_rate'],
        'Activation Function': params['activation'],
        'Training Accuracy': f"{train_accuracy:.2%}",
        'Test Accuracy': f"{test_accuracy:.2%}"
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)
