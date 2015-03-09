This neural net repository contains all the components necessary to
run the net and the data used.

The goal of this neural network is to be able to take in an arbitrary
set of data, format it to be inputted into the NN, perform some pre-
processing to maximize convergence, create a neural network of the
appropriate size (correct input and output layer sizes) and then run
the regression. Some optimizations were implemented such as performing
the pre-processing, using a different sigmoid function (tanh) and
initializing the weights within a specified range. However, advanced
optimization techniques (such as BFGS) were not implemented.

The neural network was tested on two sets of data, the first being the
classic housing prices as a function of the house features, and the
second being solar flares. Pre-processing became necessary for the
solar flare data because there were letters present in the data in
addition to numbers, so they had to be formatted (split into binary
columns for each letter).

The actual neural net script is in the NN.py file and also contains
the class that trains the NN.
    
DataFormatting.py is a Pandas script meant to be used for pre-
processing of the data to maximize the NN solution convergence.

Main contains the code that will initializes the NN and the data that
will be used by it.

