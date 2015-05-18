#Neural Network
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import DataFormatting as df
from scipy import optimize

#Class for the general neural network with the specified layer sizes
class Neural_Network(object):

    #Attributes Created with each instance of the NN
    def __init__(self, TrainingX, TrainingY, alpha):
        #Define the hyperparameters (input, hidden and output)
        self.InputLayerSize = TrainingX.shape[1]
        self.HiddenLayerSize = TrainingX.shape[1] * 3 / 5 + 1
        self.OutputLayerSize = TrainingY.shape[1]

        self.alpha = alpha

        self.W1 = np.random.normal(0,(self.InputLayerSize)**(-1./2),
                                   (self.InputLayerSize,
                                    self.HiddenLayerSize))
        self.W2 = np.random.normal(0,(self.HiddenLayerSize)**(-1./2),
                                   (self.HiddenLayerSize,
                                    self.OutputLayerSize))

    #Given an initial input X, this propagates the signal through the
    #network and outputs the hypothesis yHat
    def FeedForward(self, X):
        #This propagates the signal through the network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W2)

        yHat = self.tanh(self.z3)

        return yHat

    #Sigmoid function that outputs a value(-1 to 1) given an input
    def tanh(self, z):
        return 1.7159 * np.tanh((float(2)/3) * z)

    #Derivative of the sigmoid function above
    def tanhPrime(self, z):
        return (1.7159 * (1 - np.tanh(float(2)/3 * z)**2))

    #Calculation of the cost given an input and correct value
    def CostFunction(self, X , y):
        self.yHat = self.FeedForward(X)
        self.J = 0.5 * sum((y-self.yHat)**2)
        return self.J

    #Computes the derivative of the cost fuction wrt to the weights
    def CostFunctionPrime(self, X, y):
        #Computes the derivative with respect to W1 and W2 given an X
        #and Y
        self.yHat = self.FeedForward(X)

        delta3 = np.multiply(-(y-self.yHat),
                             self.tanhPrime(self.z3))
        
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.tanhPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def GetParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def SetParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.HiddenLayerSize*self.InputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.InputLayerSize,
                              self.HiddenLayerSize))
        W2_end = W1_end + self.HiddenLayerSize*self.OutputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.HiddenLayerSize,
                              self.OutputLayerSize))
        
    def ComputeGradients(self, X, y):
        dJdW1, dJdW2 = self.CostFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

        
class Trainer(object):
    #Function that trains the NN for as many iterations as there is
    #data: to train the NN more, input more data
    def __init__(self, NN, X, y):
        self.NN = NN
        self.X = X
        self.y = y
        self.CostF = []
        self.CostFPrime = []

    def UpdateWeights(self, dJdW1, dJdW2):
        self.NN.W1 = self.NN.W1 - self.NN.alpha * dJdW1
        self.NN.W2 = self.NN.W2 - self.NN.alpha * dJdW2
        
    #The function that does the training
    def StochasticLearning(self, iterations):
        for j in range(iterations):
            for i in range(self.X.shape[0]):
                self.CostF.append(self.NN.CostFunction(
                    self.X[i:(i+1), :], self.y[i]))

                #These Functions are just for data purposes
                dJdW1, dJdW2 = self.NN.CostFunctionPrime(
                    self.X[i:(i+1), :], self.y[i])

                self.CostFPrime.append(np.sum(abs(dJdW1)) +
                                       np.sum(abs(dJdW2)))

                #The actual weight updating is here
                self.UpdateWeights(dJdW1, dJdW2)


    def CostFunctionWrapper(self, params, X, y):
        self.NN.SetParams(params)
        cost = self.NN.CostFunction(X, y)
        grad = self.NN.ComputeGradients(X,y)
        
        return cost, grad
        
    def BFGS(self, iterations = 1000):
        params0 = self.NN.GetParams()

        options = {'maxiter': iterations, 'disp' : False}
        _res = optimize.minimize(self.CostFunctionWrapper, params0,
                                 jac=True, method='BFGS',
                                 args=(self.X, self.y),
                                 options=options)

        self.NN.SetParams(_res.x)
        self.optimizationResults = _res

    # This does not work as of now
    def BatchLearning(self, iterations):
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

        
