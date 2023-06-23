from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, normalization='Batch', dropout_value = 0.05):
        super(Net, self).__init__()

        self.convblock1 = self.conv2d(1, 8, 3, normalization, dropout_value, 2)
        self.convblock2 = self.conv2d(8, 16, 3, normalization, dropout_value, 4) 
        
        #Transition Block
        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias= False )
        )
        
        self.convblock3 = self.conv2d(8, 16, 3, normalization, dropout_value, 4) 
        self.convblock4 = self.conv2d(16, 24, 3, normalization, dropout_value,  4)
        self.gap = nn.AvgPool2d(kernel_size = 8)
        self.convblock5 = self.conv2d(24, 32, 1, normalization, dropout_value,  4)
        self.convblock6 =  self.conv2d(32, 16, 1, normalization, dropout_value, 4)
        self.convblock7 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False) 

    
    def conv2d(self, in_channels, out_channels, kernel_size, normalization, dropout, num_of_groups):
        if normalization == "Batch":
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=0, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout)
            )
        elif normalization == "Layer":
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
                nn.ReLU(),
                ## When number of groups is 1, its layernorm
                nn.GroupNorm(1, out_channels),
                nn.Dropout(dropout)
            )
        elif normalization == "Group":
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
                nn.ReLU(),
                nn.GroupNorm(num_of_groups, out_channels),
                nn.Dropout(dropout)
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
      
        return conv

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transblock1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)        
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
