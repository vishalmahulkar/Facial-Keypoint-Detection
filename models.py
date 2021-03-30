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
        # (32, 220, 220)
        
        self.pool = nn.MaxPool2d(2, 2)
        # output size = 220/2 = 110
        # (32, 110, 110)
        
        self.drop1 = nn.Dropout(p=0.4)
        # (32, 110, 110)
              
        
        self.conv2 = nn.Conv2d(8, 16, 5)
        # output size (110-5)/1 + 1 = 106
        # (64, 106, 106)
        
        #self.pool2 = nn.MaxPool2d(2, 2)
        # output size 
        # (64, 53, 53)
        
        self.drop2 = nn.Dropout(p=0.4)
        # (64, 53, 53)
        
        
        #self.conv3 = nn.Conv2d(64, 128, 3)
        # output size (53-3)/1 + 1 = 51
        # (128, 51, 51)
        
        # self.pool = nn.MaxPool2d(2, 2)
        # output size 
        # (128, 25, 25)
        
        #self.drop3 = nn.Dropout(p=0.4)
        # (128, 25, 25)
        
        
        # flatten
        self.fc1 = nn.Linear(16*53*53, 6400)
        self.drop4 = nn.Dropout(p=0.4)
        
        # Linear
        self.fc2 = nn.Linear(6400, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        #x = self.pool(F.relu(self.conv3(x)))
        #x = self.drop3(x)
        
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
