import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_value):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1A
        self.convblock1A = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        # CONVOLUTION BLOCK 2A
        self.convblock2A = nn.Sequential(
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # CONVOLUTION BLOCK 3A
        self.convblock3A = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14
        
        # CONVOLUTION BLOCK 4A
        self.convblock4A = nn.Sequential(
            nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14

        # CONVOLUTION BLOCK 1B
        self.convblock1B = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        # CONVOLUTION BLOCK 2B
        self.convblock2B = nn.Sequential(
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # CONVOLUTION BLOCK 3B
        self.convblock3B = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14
        
        # CONVOLUTION BLOCK 4B
        self.convblock4B = nn.Sequential(
            nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14


        ## CONVOLUTION BLOCK 4 - Depthwise Convolution
        self.depthwise_separable_block = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (3, 3), padding = 0, groups = 96, bias = False),
            nn.Conv2d(in_channels = 96, out_channels = 60, kernel_size = (1, 1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.Dropout(dropout_value)
        ) ## output_size = 12

        ## CONVOLUTION BLOCK 5 - Reduction
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels = 60, out_channels = 30, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 30, out_channels = 10, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6) ## Global Average Pooling
        ) # output_size = 1


    def forward(self, x):
        x1 = self.convblock1A(x)
        x1 = self.convblock2A(x1)
        x1 = self.convblock3A(x1)
        x1 = self.convblock4A(x1)

        x2 = self.convblock1B(x)
        x2 = self.convblock2B(x2)
        x2 = self.convblock3B(x2)
        x2 = self.convblock4B(x2)

        y = torch.cat((x1,x2), 1)

        y = self.depthwise_separable_block(y)
        y = self.convblock5(y)
        y = self.gap(y)

        y = y.view(-1, 10)
        return F.log_softmax(y, dim=-1)