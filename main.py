import numpy as np
from math import exp
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActivationFunction:
    """
    Class to encapsulate activation functions and their derivatives.
    """
    @staticmethod
    def sigmoid(x, derivative=False):
        """
        Compute the sigmoid function or its derivative.

        Parameters:
        x (numpy.ndarray): Input array.
        derivative (bool): If True, compute the derivative.

        Returns:
        numpy.ndarray: Result of the sigmoid function or its derivative.
        """
        return np.array([
            (1 / (1 + exp(-xi))) * (1 - (1 / (1 + exp(-xi)))) if derivative else 1 / (1 + exp(-xi))
            for xi in x
        ]).reshape(-1, 1)

    @staticmethod
    def tanh(x, derivative=False):
        """
        Compute the tanh function or its derivative.

        Parameters:
        x (numpy.ndarray): Input array.
        derivative (bool): If True, compute the derivative.

        Returns:
        numpy.ndarray: Result of the tanh function or its derivative.
        """
        return np.array([
            1 - ((exp(xi) - exp(-xi)) / (exp(xi) + exp(-xi))) ** 2 if derivative else (exp(xi) - exp(-xi)) / (exp(xi) + exp(-xi))
            for xi in x
        ]).reshape(-1, 1)

class NeuralNetwork:
    """
    Class to encapsulate the neural network functionality.
    """
    def __init__(self, input_dim, hidden_neurons, learning_rate=0.1, iterations=1000):
        """
        Initialize the neural network parameters.

        Parameters:
        input_dim (int): Number of input features.
        hidden_neurons (int): Number of hidden layer neurons.
        learning_rate (float): Learning rate for weight updates.
        iterations (int): Number of training iterations.
        """
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.iterations = iterations

        # Initialize weights
        self.w1 = np.random.uniform(-1, 1, (hidden_neurons, input_dim))
        self.w2 = np.random.uniform(-1, 1, (hidden_neurons, 1))

        logger.info("Neural network initialized with input_dim=%d, hidden_neurons=%d, learning_rate=%.3f, iterations=%d",
                    input_dim, hidden_neurons, learning_rate, iterations)

    def forward_pass(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        tuple: Hidden layer output, final output.
        """
        h_in = np.dot(x, self.w1.T)
        h_out = ActivationFunction.tanh(h_in)
        o_in = np.dot(h_out.T, self.w2)
        o_out = ActivationFunction.sigmoid(o_in)
        return h_in, h_out, o_in, o_out

    def backward_pass(self, x, y, h_out, o_in, o_out):
        """
        Perform a backward pass to compute weight updates.

        Parameters:
        x (numpy.ndarray): Input data.
        y (float): Target output.
        h_out (numpy.ndarray): Hidden layer output.
        o_in (numpy.ndarray): Final layer input.
        o_out (numpy.ndarray): Final layer output.

        Returns:
        tuple: Gradients for w1 and w2.
        """
        dE_dypred = o_out - y
        dypred_doutout = 1
        doutout_doutin = ActivationFunction.sigmoid(o_in, derivative=True)
        doutin_dw2 = h_out

        doutin_dhout = self.w2
        dhout_dhin = ActivationFunction.tanh(np.dot(x, self.w1.T), derivative=True)
        dhin_dw1 = x

        dE_dw2 = dE_dypred * dypred_doutout * doutout_doutin * doutin_dw2
        dE_dw1 = dE_dypred * dypred_doutout * doutout_doutin * doutin_dhout * dhout_dhin * dhin_dw1

        return dE_dw1, dE_dw2

    def train(self, X, y):
        """
        Train the neural network.

        Parameters:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): Target outputs.
        """
        errors = []

        for iteration in range(self.iterations):
            idx = np.random.randint(len(X))
            x = X[idx]
            target = y[idx]

            h_in, h_out, o_in, o_out = self.forward_pass(x)
            dE_dw1, dE_dw2 = self.backward_pass(x, target, h_out, o_in, o_out)

            self.w1 -= self.learning_rate * dE_dw1
            self.w2 -= self.learning_rate * dE_dw2

            error = 0.5 * (target - (1 if o_out > 0.5 else 0)) ** 2
            errors.append(error)

            if iteration % 100 == 0 or iteration == self.iterations - 1:
                logger.info("Iteration %d: Error=%.6f", iteration, error)

        return errors

    def predict(self, X):
        """
        Make predictions using the trained network.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Predicted labels.
        """
        predictions = []

        for x in X:
            _, _, _, o_out = self.forward_pass(x)
            predictions.append(1 if o_out > 0.5 else 0)

        return np.array(predictions)

# Initialize data
X = np.array([
    [1.1, 2.2], [1.8, 3.7], [2.7, 4.9], [3.1, 4.0],
    [4.0, 3.0], [2.5, 2.5], [3.1, 0.9], [4.3, 0.9],
    [5.2, 2.1], [5.9, 2.9]
])
bias = np.ones((X.shape[0], 1))
X = np.hstack((X, bias))
y = np.zeros((10, 1))
y[5:] = 1

# Train and evaluate the neural network
nn = NeuralNetwork(input_dim=X.shape[1], hidden_neurons=5)
errors = nn.train(X, y)
predictions = nn.predict(X)

# Plot training error
plt.plot(errors)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Training Error Over Iterations")
plt.show()

# Display predictions
logger.info("Predictions: %s", predictions)
