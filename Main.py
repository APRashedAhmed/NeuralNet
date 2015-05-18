#NN Main

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import DataFormatting as df
import NN


def CreateTrainingAndTestingData(Data,Y,TrainingLength,TestingLength):
    TrainingData = np.empty((TrainingLength, Data.shape[1])) 
    TestingData = np.empty((TestingLength, Data.shape[1]))
    TrainingY = np.empty((TrainingLength, Y.shape[1]))
    TestingY = np.empty((TestingLength, Y.shape[1]))
    TestingIndex = []
    
    TrainingIndex = random.sample(range(1, Data.shape[0]),
                                  TrainingLength)
    for i in range(TestingLength):
        Index = np.random.randint(0, Data.shape[0])
        while Index in TrainingIndex:
            Index = np.random.randint(0, Data.shape[0])

        TestingIndex.append(Index)
        
    for i in range(TrainingLength):
        TrainingData[i] = Data[TrainingIndex[i]]
        TrainingY[i] = Y[TrainingIndex[i]]
        
    for i in range(TestingLength):
        TestingData[i] = Data[TestingIndex[i]]
        TestingY[i] = Y[TestingIndex[i]]
        
    return TrainingData, TrainingY, TestingData, TestingY


def AverageError(NN, Data, Y):
    TotalError = 0

    for i in range(len(Data)):
        TotalError += np.absolute(NN.FeedForward(
            Data[i:(i+1),:]) - Y[i])
    
    for i in range(TotalError.shape[1]):
        Range = (y[:,i].max() - y[:,i].min())
        TotalError[:,i] = TotalError[:,i] / Range

    return (TotalError / len(Data))

        
# AllData_df = df.GenerateMasterListWithHeader(
#     "DataFiles\Data2_normalized.txt")
# AllData = df.TurnDFintoMatrix(AllData_df)

# All_y_df = df.GenerateMasterListWithHeader(
#     "DataFiles\Data2_y.txt")
# y = df.TurnDFintoMatrix(All_y_df)

AllData_df = df.GenerateMasterListWithHeader(
    "Housing\HousingData_N.txt")
AllData = df.TurnDFintoMatrix(AllData_df)

All_y_df = df.GenerateMasterListWithHeader(
    "Housing\HousingY_N.txt")
y = df.TurnDFintoMatrix(All_y_df)



TrainingX, TrainingY, TestingX, TestingY = CreateTrainingAndTestingData(AllData, y, 450, 10)

MyNeuralNet = NN.Neural_Network(TrainingX, TrainingY, .0003)

# print MyNeuralNet.W1.shape
# print MyNeuralNet.W2.shape

# print TrainingX.shape
MyTrainer = NN.Trainer(MyNeuralNet, TrainingX,
                               TrainingY)

Error0 = AverageError(MyNeuralNet, TestingX, TestingY)
print "Error before training:"
print Error0

MyTrainer.BFGS()

Errorf = AverageError(MyNeuralNet, TestingX, TestingY)
print "Error after training:"
print Errorf

# plt.plot(MyTrainer.CostF)

