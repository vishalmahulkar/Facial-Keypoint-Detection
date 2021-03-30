## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, 5)
        # output size (224-5)/1 + 1 = 220
        # (8, 220, 220)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        # output size = 220/2 = 110
        # (8, 110, 110)
        
        self.drop1 = nn.Dropout(p=0.4)
        # (8, 110, 110)
              
        
        self.conv2 = nn.Conv2d(8, 16, 3)
        # output size (110-3)/1 + 1 = 108
        # (16, 108, 108)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        # output size 
        # (16, 54, 54)
        
        self.drop2 = nn.Dropout(p=0.4)
        # (16, 54, 54)
        
        
        self.conv3 = nn.Conv2d(16, 32, 3)
        # output size (54-3)/1 + 1 = 52
        # (32, 52, 52)
        
        self.pool3 = nn.MaxPool2d(2, 2)
        # output size 
        # (32, 26, 26)
        
        self.drop3 = nn.Dropout(p=0.4)
        # (32, 26, 26)
        
        
        # flatten
        self.fc1 = nn.Linear(32*26*26, 6400)
        self.drop4 = nn.Dropout(p=0.4)
        
        # Linear
        self.fc2 = nn.Linear(6400, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        
        # a softmax layer to convert the 136 outputs into a distribution of class scores
        # x = F.log_softmax(x, dim=1)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
