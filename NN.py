# Neural Network

# Neural network implementation. Currently only supports having one
# hidden layer, and only BFGS as the training method.

import numpy as np
import random
import pandas as pd
from scipy import optimize

def GenerateNeuralNet(X, y, alpha):
    """Function returns a neural network that fits to the data"""
    return NN(X.shape[1], X.shape[1] * 3 / 5 + 1, y.shape[1], alpha)

class NN(object):
    """Class for the general neural network with the specified layer
    sizes"""
    def __init__(self, InputLayerSize, HiddenLayerSize,
                 OutputLayerSize, alpha = .01):
        
        self.InputLayerSize = InputLayerSize # Input layer size
        self.HiddenLayerSize = HiddenLayerSize # Hidden Layer Size
        self.OutputLayerSize = OutputLayerSize # Output layer size

        self.alpha = alpha      # Learning rate

        # Weights between input and hidden layer
        self.W1 = np.random.normal(0,(self.InputLayerSize)**(-1./2),
                                   (self.InputLayerSize,
                                    self.HiddenLayerSize))
        # Weights between hidden and output layer
        self.W2 = np.random.normal(0,(self.HiddenLayerSize)**(-1./2),
                                   (self.HiddenLayerSize,
                                    self.OutputLayerSize))

    def SetHiddenLayerSize(self, HiddenLayerSize):
        """Sets the hidden layer size and redefines the weights"""

        self.HiddenLayerSize = HiddenLayerSize
              
        # Weights between input and hidden layer
        self.W1 = np.random.normal(0,(self.InputLayerSize)**(-1./2),
                                   (self.InputLayerSize,
                                    self.HiddenLayerSize))
        # Weights between hidden and output layer
        self.W2 = np.random.normal(0,(self.HiddenLayerSize)**(-1./2),
                                   (self.HiddenLayerSize,
                                    self.OutputLayerSize))
    def FeedForward(self, X):
        """Given an initial input X, this propagates the signal
        through the network and outputs the hypothesis yHat"""
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W2)

        yHat = self.tanh(self.z3)

        return yHat

    def tanh(self, z):
        """Very similar to sigmoid function but outputs a value
        between -1 and 1 given an input"""
        return 1.7159 * np.tanh((float(2)/3) * z)

    def tanhPrime(self, z):
        """Derivative of the function above"""
        return (1.7159 * (1 - np.tanh(float(2)/3 * z)**2))

    def CostFunction(self, X , y):
        """Calculation of the cost given an input and correct 
        value"""
        self.yHat = self.FeedForward(X)
        self.J = 0.5 * sum((y-self.yHat)**2)

        return self.J

    def CostFunctionPrime(self, X, y):
        """Computes the derivative with respect to W1 and W2 given 
        an X and Y"""
        self.yHat = self.FeedForward(X)

        delta3 = np.multiply(-(y-self.yHat),self.tanhPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3) # W2 Update values

        delta2 = np.dot(delta3, self.W2.T)*self.tanhPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) # W1 update values

        return dJdW1, dJdW2
    
    # Helper functions for interacting with other methods
    
    def GetWeights(self):
        """Returns the weights rolled into a long vector"""
        Weights = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return Weights
    
    def SetWeights(self, Weights):
        """Set W1 and W2 using a rolled up vector of weights"""
        W1_start = 0
        W1_end = self.HiddenLayerSize*self.InputLayerSize
        self.W1 = np.reshape(Weights[W1_start:W1_end], \
                             (self.InputLayerSize,
                              self.HiddenLayerSize))
        W2_end = W1_end + self.HiddenLayerSize*self.OutputLayerSize
        self.W2 = np.reshape(Weights[W1_end:W2_end], \
                             (self.HiddenLayerSize,
                              self.OutputLayerSize))

    def ComputeGradients(self, X, y):
        """Returns the gradients for W1 and W2 rolled into a 
        vector"""
        dJdW1, dJdW2 = self.CostFunctionPrime(X, y)

        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def CostFunctionWrapper(self, Weights, X, y):
        """Wrapper function to create correct input format for 
        BFGS"""
        self.SetWeights(Weights)
        cost = self.CostFunction(X, y)
        grad = self.ComputeGradients(X,y)
        
        return cost, grad
        
    def BFGS(self, X, y, iterations = 1000):
        """Backpropagation using BFGS in the scipy library"""
        Weights = self.GetWeights()

        # options = {'maxiter': iterations, 'disp' : False}
        _res = optimize.minimize(self.CostFunctionWrapper,
                                       Weights,
                                       jac=True,
                                       method='BFGS',
                                       args=(X, y))

        self.SetWeights(_res.x)
        self.optimizationResults = _res
        
    def UpdateWeights(self, dJdW1, dJdW2):
        """Updates the weights using the inputted matrices"""
        self.NN.W1 = self.NN.W1 - self.NN.alpha * dJdW1
        self.NN.W2 = self.NN.W2 - self.NN.alpha * dJdW2
        
    def StochasticLearning(self,X, y, iterations):
        """Backpropagation using stochastic learning"""
        self.CostF = []
        self.CostFPrime = []

        for j in range(iterations):
            for i in range(X.shape[0]):
                self.CostF.append(self.CostFunction(X[i:(i+1),:],
                                                    y[i]))

                #These Functions are just for tracking purposes
                dJdW1, dJdW2 = self.CostFunctionPrime(X[i:(i+1),:],
                                                      y[i])

                self.CostFPrime.append(np.sum(abs(dJdW1)) +
                                       np.sum(abs(dJdW2)))

                #The actual weight updating is here
                self.UpdateWeights(dJdW1, dJdW2)

    def BatchLearning(self, iterations):
        """Uses batch learning for weight updates - does not work"""
        dJdW1 = np.array(self.NN.InputLayerSize,
                         self.NN.HiddenLayerSize)
        dJdW2 = np.array(self.NN.HiddenLayerSize,
                         self.NN.OutputLayerSize)
        
        for i in range(iterations):
            for j in range(self.X.shape[0]):
                dJdW1T, dJdW2T = self.NN.CostFunctionPrime(
                    self.X,
                    self.y)
                dJdW1 += dJdW1T
                dJdW2 += dJdW2T
            
            self.CostFPrime.append(np.sum(abs(dJdW1)) +
                                   np.sum(abs(dJdW2)))

            self.CostF.append(sum(self.NN.J)/len(y))

            self.UpdateWeights(dJdW1, dJdW2)

    def Train(self, X, y, method = 'BFGS', iterations = 1000):
        """Trains the neural network using the method inputted. 
        Set to BFGS for now"""
        if  method == 'BFGS':
            self.BFGS(X, y, iterations)

        elif method == 'SL':
            self.StochasticLearning(X,y, iterations)

        elif method == 'BL':
            print "Error: batch learning not correctly implemented."

        else:
            print "Error: inputted learning method not recognized."
    
