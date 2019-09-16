## Trained on ImageNet validation set

## Model: Convolutional auto encoder decoder
Consists of 4 layers of 2D convolutions with kernel size = 5 and stride = 2.
Each convolution is followed by a ReLu activation except the bottleneck layer (we want all elements to be real values so that it encodes information more effectively). We also select the channel sizes such that the bottleneck layer make full use of the bytes limit. 

We also use a "Leaky ReLu-1" activation at the last layer of the decoder to help the network output values that are between 0 and 1 but without getting stuck.

## Loss: L1 loss to match with the evaluation metric

## Optimizer: Adam with lr=1e-4, other parameters set to Pytorch default