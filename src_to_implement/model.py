import torch
import torch.nn as nn
import torch.nn.functional as F
### Define the ResNet


### in_channel: Input Feature Map
### out_channel: output feature map
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=self.stride,padding=(1,1))
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=self.out_channel,out_channels=self.out_channel,kernel_size=3,padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.identity_conv = nn.Conv2d(in_channels=in_channel,out_channels=self.out_channel,kernel_size=1,stride=self.stride)
        self.identity_bn = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()
        self.downsample = None
        
        

        # if self.stride != 1 or in_channel != out_channel:
        #     print("Test")
        #     self.downsample = nn.Sequential(
        #         nn.Conv2d(
        #             in_channel,
        #             out_channel,
        #             kernel_size=1,
        #             stride=self.stride,
        #             padding=2
        #         ),
        #         nn.BatchNorm2d(out_channel)
        #     )
        # else:
        #     self.downsample = None
        
        
        
        
    def forward(self,X):
        identity = X
        
        #####
        identity = self.identity_conv(identity)
        identity = self.identity_bn(identity)
    #    print("identity shape: ", identity.shape)
   #     print("stride: ",self.stride)
        ### Main Path
  #      print("X: ", X.shape)
        X = self.conv1(X)
 #       print("nach conv1 ",X.shape)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.conv2(X)
#print("nach conv2 ",X.shape)
        X = self.bn2(X)
        
        ### Skip Connection Path
        ### Since Conv1 downsamples, Idenity Path must be fit
        
        # if self.downsample is not None:
        #     identity = self.downsample(identity)
        #     print("Checkpoint: downsample")
        # print("input identity ",identity.shape)
        X += identity
        
        X = self.relu(X)
        return X
        

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Conv2d(3,64,7,2)
        
        self.bn = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(3,2)
        
        self.resblock1 = ResBlock(64,64,1)
        self.resblock2 = ResBlock(64,128,2)
        self.resblock3 = ResBlock(128,256,2)
        self.resblock4 = ResBlock(256,512,2)
        
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d((1,1))

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(512,2)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self,X):
        X = self.hidden(X)
        X = self.bn(X)
        X = self.relu(X)
        X = self.resblock1(X)
        X = self.resblock2(X)
        X = self.resblock3(X)
        X = self.resblock4(X)
        X = self.GlobalAvgPool(X)
        X = self.flatten(X)
        #X = self.dropout(X)
        X = self.fc(X)
        X = self.sigmoid(X)
        return X

        