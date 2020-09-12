import torch.nn as nn
import torch.nn.functional as F



class CustomDilatedNet(nn.Module):
    def __init__(self, dropout_value=0):
        super(CustomDilatedNet, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=1, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12
        # Block with 1 Dilation
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        # Block with Depthwise separable Convolution
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,groups=128),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0,groups=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )
        self.gap = nn.AvgPool2d(kernel_size=8)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.output(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)