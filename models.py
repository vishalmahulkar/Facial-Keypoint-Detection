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
        self.conv1 = nn.Conv2d(1, 32, 5)
        # output size (144-5)/1 + 1 = 140
        # (32, 140, 140)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        # output size = 140/2 = 70
        # (32, 70, 70)
        
        self.drop1 = nn.Dropout(p=0.1)
        # (32, 70, 70)
              
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        # output size (70-3)/1 + 1 = 68
        # (64, 68, 68)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        # output size 
        # (64, 34, 34)
        
        self.drop2 = nn.Dropout(p=0.2)
        # (64, 34, 34)
        
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        # output size (34-3)/1 + 1 = 32
        # (128, 32, 32)
        
        self.pool3 = nn.MaxPool2d(2, 2)
        # output size 
        # (128, 16, 16)
        
        self.drop3 = nn.Dropout(p=0.3)
        # (128, 16, 16)
        
        
        self.conv4 = nn.Conv2d(128, 256, 1)
        # output size (16-1)/1 + 1 = 16
        # (256, 16, 16)
        
        self.pool4 = nn.MaxPool2d(2, 2)
        # output size 
        # (256, 8, 8)
        
        self.drop4 = nn.Dropout(p=0.4)
        # (256, 8, 8)
        
                
        # flatten
        self.fc1 = nn.Linear(256*8*8, 6400)
        self.fc1_drop = nn.Dropout(p=0.5)
        
        # Linear
        self.fc2 = nn.Linear(6400, 1000)
        self.fc2_drop = nn.Dropout(p=0.6)
        
        # Linear
        self.fc3 = nn.Linear(1000, 1000)
        self.fc3_drop = nn.Dropout(p=0.6)
        
         # Linear
        self.fc4 = nn.Linear(1000, 136)
        
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
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        
        x = self.fc4(x)
        
        # a softmax layer to convert the 136 outputs into a distribution of class scores
        # x = F.log_softmax(x, dim=1)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
