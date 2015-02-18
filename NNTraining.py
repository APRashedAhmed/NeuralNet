import numpy as np
import NN

class Trainer(object):
    #Function that trains the NN for as many iterations as there is
    #data: to train the NN more, input more data
    def __init__(self, NNet, Data, Y):
        self.NNet = NNet
        self.Data = Data
        self.Y = Y

    def UpdateWeights(self, dJdW1, dJdW2):
        alpha = 1

        self.NNet.W1 = self.NNet.W1 - alpha * dJdW1
        self.NNet.W2 = self.NNet.W2 - alpha * dJdW2
    
    #The function that does the training
    def StochasticLearning(self, iterations):
        for j in range(iterations):
            for i in range(len(self.Data)):
                self.CostF.append(self.NNet.CostFunction(
                    self.Data[i:(i+1), 0:5], self.Y[i]))

                #These Functions are just for data purposes
                dJdW1, dJdW2 = self.NNet.CostFunctionPrime(
                    self.Data[i:(i+1), 0:5], self.Y[i])

                self.CostFPrime.append(np.sum(abs(dJdW1)) +
                                  np.sum(abs(dJdW2)))

                #The actual weight updating is here
                self.UpdateWeights(dJdW1, dJdW2)

    def BatchLearning(self, iterations):
        dJdW1 = np.array(self.NNet.InputLayerSize,
                         self.NNet.HiddenLayerSize)
        dJdW2 = np.array(self.NNet.HiddenLayerSize,
                         self.NNet.OutputLayerSize)
        
        for i in range(iterations):
            for j in range(len(self.Data)):
                dJdW1T, dJdW2T = self.NNet.CostFunctionPrime(
                    self.Data,
                    self.Y)
                dJdW1 += dJdW1T
                dJdW2 += dJdW2T

            
            self.CostFPrime.append(np.sum(abs(dJdW1)) +
                                   np.sum(abs(dJdW2)))

            self.CostF.append(sum(self.NNet.J)/len(Y))

            self.UpdateWeights(dJdW1, dJdW2)
