import numpy as np

class My_network:
    def __init__(self,data, y, num_of_layers, layers_nodes, activations, iterations, alpha):
        """
        Initializes the neural network object.

        Args:
        data: numpy array, training data 
        y: numpy array, target values
        num_of_layers: int, number of hidden layers + the output layer
        layers_nodes: list, number of nodes in each hidden layer 
                            inedx 0 is for the input layer, index -1 is for the output layer
        activations: list, activation functions of each layer
        iterations: int, number of iterations to train the network
        alpha: float, learning rate for gradient descent
        """
        self.data = data
        self.y = y
        self.num_of_layers = num_of_layers 
        self.layers_nodes = layers_nodes
        self.activations = activations
        self.iterations=  iterations
        self.alpha = alpha

    # 1. Define needed functions
    # We can include all possible activation functions and their derivatives.
    def sigmoid(self, z):
        """
        Computes sigmoid function on input z.

        Args:
        z: numpy array, input to the sigmoid function

        Returns:
        numpy array, output of sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def der_sigmoid(self, a):
        """
        Computes the derivative of the sigmoid function.

        Args:
        a: numpy array, input to the derivative of sigmoid function

        Returns:
        numpy array, output of the derivative of sigmoid function
        """
        return a * (1 - a)

    def tanh(self, z):
        """
        Computes hyperbolic tangent function on input z.

        Args:
        z: numpy array, input to the hyperbolic tangent function

        Returns:
        numpy array, output of the hyperbolic tangent function
        """
        return np.tanh(z)

    def der_tanh(self, a):
        """
        Computes the derivative of the hyperbolic tangent function.

        Args:
        a: numpy array, input to the derivative of hyperbolic tangent function

        Returns:
        numpy array, output of the derivative of hyperbolic tangent function
        """
        return 1 - a * a

    def relu(self, z):
        """
        Computes rectified linear unit function on input z.

        Args:
        z: numpy array, input to the rectified linear unit function

        Returns:
        numpy array, output of the rectified linear unit function
        """
        return np.maximum(0, z)

    def der_relu(self, a):
        """
        Computes the derivative of the rectified linear unit function.

        Args:
        a: numpy array, input to the derivative of rectified linear unit function

        Returns:
        numpy array, output of the derivative of rectified linear unit function
        """
        return 1 if a > 0 else 0

    def activation_function(self, func, z):
        """
        Returns the activation function given its name.

        Args:
        func: str, name of the activation function
        z: numpy array, input to the activation function

        Returns:
        numpy array, output of the activation function
        """
        if func == "sigmoid":
            return self.sigmoid(z)
        elif func == "tanh":
            return self.tanh(z)
        elif func == "relu":
            return self.relu(z)

    def derivative_function(self, der, a):
        """
        Returns the derivative of the activation function given its name.

        Args:
        der: str, name of the derivative of the activation function
        a: numpy array, input to the derivative of the activation function

        Returns:
        numpy array, output of the  derivative of the activation function
        """
        if der == "sigmoid":
            return self.der_sigmoid(a)
        elif der == "tanh":
            return self.der_tanh(a)
        elif der == "relu":
            return self.der_relu(a)
        
        
    def cost(self, a, y):
        """
        Compute the cost (loss) between a predicted value and a true value.

        Args:
        a: float, the predicted value.
        y: float, the true value.

        Returns:
        The cost (loss) between the predicted value and the true value.
        """
        return np.square(a - y)

    def der_cost(self, a, y):
        """
        Compute the derivative of the cost (loss) function with respect to the predicted value.

        Args:
            a: float, the predicted value.
            y: float, the true value.

        Returns:
        The derivative of the cost (loss) function with respect to the predicted value.
        """
        return 2 * (a-y)

    def predict(self, a):
        """
        Make a binary prediction based on a given predicted value.

        Args:
        a: float, the predicted value.

        Returns:
         - 1 if the predicted value is greater than 0.5, 0 otherwise.
        """
        return 1 if a > 0.5 else 0  

    def initial_values(self):
        """
        Initializes the weights and biases for each layer in the network.

        Returns:
        W: np.ndarray, a list of weight matrices, one for each layer in the network
        B: np.ndarray, a list of bias vectors, one for each layer in the network
        """        
        W = [0] * (self.num_of_layers) 
        B = [0] * (self.num_of_layers)
        for i in range(len(W)): 
            W[i] = np.random.rand(self.layers_nodes[i+1], self.layers_nodes[i])
            B[i] = np.random.rand(self.layers_nodes[i+1], 1)
            
        return W, B


    # 2. Start Implementing
    def forward_pass(self, x, w, b, activation):
        """
        Compute the output of a layer given its inputs, weights, biases, and activation function.

        Args:
        x: numpy.ndarray, the inputs to the layer, as a column vector.
        w: numpy.ndarray, the weights of the layer, as a matrix.
        b: numpy.ndarray, the biases of the layer, as a column vector.
        activation: str, the name of the activation function to use.

        Returns:
        The output of the layer, as a column vector.
        """

        z = np.dot(w, x) + b
        a = self.activation_function(activation, z)
        return a
    
    def back_pass(self, layer_output, w, b, n, activation, output_gradient):
        """
        Perform backpropagation to update the weights and biases of a layer given the output of the previous layer.

        Args:
        layer_output: list, the output vectors of each layer in the network.
        w: np.ndarray, the weights of the layer being updated, as a matrix.
        b: np.ndarray, the biases of the layer being updated, as a column vector.
        n:int, the index of the current layer.
        activation: str, the activation function used in the current layer.
        output_gradient: np.ndarray, the gradient of the cost (loss) function with 
                                        respect to the output of the layer being updated.

        Returns:
        output_gradient: np.ndarray, the gradient of the cost (loss) function with 
                                        respect to the output of the previous layer.
        w: np.ndarray, the updated weights for the current layer, as a matrix.
        b: np.ndarray, the updated biases for the current layer, as a column vector.
        """
        dc_dz = self.derivative_function(activation, layer_output[n]) *  output_gradient
        
        dc_dw = np.dot(dc_dz, layer_output[n-1].T) 
        dc_db = dc_dz
        
        output_gradient = np.dot(w.T, dc_dz)
        
        w -= self.alpha * dc_dw
        b -= self.alpha * dc_db
        
        return output_gradient, w, b

    # 3. Put them together
    def network(self, x, y_, W, B):
        """
        Computes the output of the network given the input, weights and biases.

        Args:
        x: np.ndarray, the input to the network
        y_: np.ndarray, the target output of the network
        W: np.ndarray, a list of weight matrices, one for each layer in the network
        B: np.ndarray, a list of bias vectors, one for each layer in the network

        Returns:
        layer_output[-1]: folat, the final output of the network as a scaler
        """
        layer_output= [0] * (self.num_of_layers+1)
        layer_output[0] = x
        for i in range(self.iterations):
            # forward pass
            for j in range(self.num_of_layers):
                x = self.forward_pass(layer_output[j], W[j], B[j], self.activations[j]) # the output of jth layer is the input for the j+1th layer
                layer_output[j+1] = x
            # the derivative of the cost function
            output_gradient = self.der_cost(layer_output[-1], y_)
            # Back pass
            for j in range(self.num_of_layers-1, -1, -1):
                output_gradient , W[j], B[j] = self.back_pass(layer_output, W[j], B[j], j+1, self.activations[j], output_gradient) 
        
        return layer_output[-1] 

    # 4. Fit the model to a data
    def fit(self):
        """
        Trains the neural network on the given dataset.

        Returns:
        ypred: the predicted outputs for the input data
        error: the mean squared error between the predicted outputs and the true outputs
        """
        error = 0
        ypred = np.zeros_like(self.y)
        
        W, B = self.initial_values()
        for i in range(len(self.data)):
            yhat = self.network(self.data[i], self.y[i], W, B)
            error += self.cost(yhat, self.y[i])
            ypred[i] = self.predict(yhat)
            
        error /= len(self.data)
        
        return ypred, error





