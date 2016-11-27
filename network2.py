"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithms for a feedforward neural network.
Improvments include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party Libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output "a" and desired output
        "y",

        """
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with output "a" and desired output
        "y", Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both "a" and "y" have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1 - y)*np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter "z" is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a - y)


#### Main Network class
class Network(object):

    def __init__(self, size, cost=CrossEntropyCost):
        """The list "sizes" contains the number of neurons in the respective
        layers of the network.  FOr example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        "self.default_weight_initializer" (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and Standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and Standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using Gaussian distribution with mean 0
        and Standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and Standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias Initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight Initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        def feedforward(self, a):
            """Return the output of the network if "a" is input."""
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a) + b)
            return a

        def SGD(self, training_data, epochs, mini_batch_size, eta,
                lmbda=0.0,
                evaluation_data=None,
                monitor_evaluation_cost=False,
                monitor_evaluation_accuracy=False,
                monitor_training_cost=False,
                monitor_training_accuracy=False):
            """Train the neural network using mini-batch stochastic gradient
            descent.  The "training_data" is a list of tuples "(x, y)"
            representing the training inputs and the desired outputs.  The
            other non-optimal parameters are self-explanatory, as is the
            regularization parameter "lmbda".  The method also accepts
            "evaluation_data", usually either the validation or test
            data.  We can monitor the cost and accuracy on either the
            evaluation data or the training data, by setting the
            approriate flags.  The method returns a tuple containing four
            lists: the (pre-epoch) costs on the evaluation data, the
            accuracies on the evaluation data, the costs on the training
            data, and the accuracies on the training data.  All values are
            evaluated at the end of each training epoch.  So, for example,
            if we train for 30 epochs, then the first element of the tuple
            will be a 30-element list containing the cost on the
            evaluation data at the end of each epoch. Note that the lists
            are empty if the corresponding flag is not set.

            """
            if evaluation_data: n_data = len(evaluation_data)
            n = len(training_data)
            evaluation_cost, evaluation_accuracy = [], []
            training_cost, training_accuracy = [], []
            for j in xrange(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k: k + mini_batch_size]
                    for k in xrange(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data))
                print "Epoch %s training complete" % j
                if monitor_training_cost:
                    cost = self.total_cost(training_data, lmbda)
                    training_cost.append(cost)
                    print "Cost on training data: {}".format(cost)
                if monitor_training_accuracy:
                    accuracy = self.accuracy(training_data, convert=True)
                    training_accuracy.append(accuracy)
                    print "Accuracy on training data {} / {}".format(
                        accuracy, n)
                if monitor_evaluation_cost:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                    evaluation_cost.append(cost)
                    print "Cost on evaluation data: {}".format(cost)
                if monitor_evaluation_accuracy:
                    accuracy = self.accuracy(evaluation_data)
                    evaluation_accuracy.append(accuracy)
                    print "Accuracy on evaluation data: {} / {}".format(
                        self.accuracy(evaluation_data), n_data)
                print
            return evaluation_cost, evaluation_accuracy, \
                training_cost, training_accuracy

        def update_mini_batch(self, mini_batch, eta, lmbda, n):
            """Update the network's weights and biases by applying gradient
            descent using backpropagation to a single mini batch.  The
            "mini_batch" is a list of tuples "(x, y)", "eta" is the
            learning rate "lmbda" is the regularization parameter, and
            "n" is the total size of the training data set.

            """
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta/len(mini_batch))*nb
                            for b, nb in zip(self.biases, nabla_b)]
