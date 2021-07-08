import numpy as np


class Network(object):

    def __init__(self):
        """
        Initialize weight and bias parameters
        """
        self.params = {'W3': np.random.randn(128, 250),
                       'W2': np.random.randn(3, 16, 128),
                       'W1': np.random.randn(250, 16),
                       'b2': np.zeros((1, 250)),
                       'b1': np.zeros((3, 1, 128))}

    @staticmethod
    def get_one_hot(x):
        """
        Covert data to One-Hot representation
        """
        one_hot_master = np.zeros((x.shape[0], x.shape[1], 250))
        for i in range(0, len(x)):
            one_hot = np.zeros((x.shape[1], 250))
            one_hot[np.arange(x[i].size), x[i]] = 1
            one_hot_master[i] = one_hot
        return one_hot_master

    @staticmethod
    def sigmoid(t):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-t))

    @staticmethod
    def derivative_sigmoid(t):
        """
        Derivative of sigmoid activation function
        """
        return (1 / (1 + np.exp(-t))) * (1 - (1 / (1 + np.exp(-t))))

    def forward_propagation(self, x):
        """
        Forward propagation through layers
        Returns each layer output and softmax classifier probabilities
        """

        N, _, __ = x.shape

        # Word Embedding Layer
        h0 = np.dot(x, self.params['W1'])

        # Hidden Layer
        h1_1 = np.zeros((N, 3, 128))
        for i in range(0, 3):
            h1_1[:, i, :] = np.dot(h0[:, i, :],
                                   self.params['W2'][i]) + self.params['b1'][i]

        h1_2 = np.zeros((N, 128))
        for i in range(0, N):
            h1_2[i] = np.sum(h1_1[i, :, :], axis=0)

        # Sigmoid Activation Function
        h1 = self.sigmoid(h1_2)

        # Output Layer
        h2 = np.dot(h1,
                    self.params['W3']) + self.params['b2']

        # Softmax Classifier
        sum_f = np.sum(np.exp(h2), axis=1, keepdims=True)
        p = np.exp(h2) / sum_f

        return N, h0, h1_1, h1, p

    def backward_propagation(self, x, y, reg):
        """
        Backward propagation to update model parameters to decrease cross-entropy loss
        """
        N, h0, h1_1, h1, p = self.forward_propagation(x)

        # Cross-Entropy Loss
        loss = np.sum(-np.log(p[np.arange(N), y]))

        # Compute Gradients
        grads = {'W3': {},
                 'W2': {},
                 'W1': np.zeros((250, 16)),
                 'b2': {},
                 'b1': {}}

        y_one_hot = np.zeros((y.shape[0], 250))
        for i in range(0, len(y)):
            one_hot = np.zeros(250)
            one_hot[y[i]] = 1
            y_one_hot[i] = one_hot

        # dL/dW3 = Hidden Layer * (p_i - y_i) (derivative of loss w.r.t y) + regularization
        grads['W3'] = h1.T.dot(p - y_one_hot) + reg * self.params['W3']

        # dL/db2 = sum(probabilities of each class)
        grads['b2'] = np.sum((p - y_one_hot), axis=0)

        # dL/dW2 = (p_i - y_i) * W3 * derivative_sigmoid(h0[i]*W2[i]+b[i]) * Word Embedding + regularization
        d_tmp = (p - y_one_hot).dot(self.params['W3'].T)

        # sigmoid backpropagation
        temp = self.derivative_sigmoid(h1_1)

        temp2 = np.zeros((N, 3, 128))
        for i in range(0, 3):
            temp2[:, i, :] = np.multiply(d_tmp, temp[:, i, :])

        for i in range(0, 3):
            grads['W2'][i] = np.dot(temp2[:, i, :].T,
                                    h0[:, i, :]).T + reg * self.params['W2'][i]

        # dL/db1 = sum(output of sigmoid)
        for i in range(0, 3):
            grads['b1'][i] = np.sum(temp[:, i, :], axis=0)

        for i in range(0, 3):
            grads['W1'] += np.dot(temp2[:, i, :],
                                  self.params['W2'][i, :, :].T).T.dot(x[:, i, :]).T\
                           + reg * self.params['W1']

        return loss, grads

    def predict(self, x):
        """Predict forth word after forward propagation"""
        _, __, ___, ____, p = self.forward_propagation(x)
        prediction = np.argmax(p, axis=1)

        return prediction
