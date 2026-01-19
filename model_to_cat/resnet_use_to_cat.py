import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGResNet(nn.Module):        
    def __init__(self,):
        super(ECGResNet, self).__init__()

        input_channels = 12
        self.resblock1 = self.ResBlock(input_channels, input_channels * 1)
              
        self.resblock2 = self.ResBlock(input_channels * 1, input_channels * 1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.resblock3 = self.ResBlock(input_channels * 1, input_channels * 1)

              
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(192 * 3, 96)
              
        self.fc2 = nn.Sequential(nn.Linear(96, 32), nn.ReLU())

    def forward(self, x_in):
        x_in = x_in.transpose(1, 2)
        x = self.resblock1(x_in)
              
        x = self.resblock2(x)
        x = self.dropout1(x)
        x = self.resblock3(x)

        x = F.avg_pool1d(x, kernel_size=2)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)

        return x

    class ResBlock(nn.Module):
        def __init__(self, n_filters_in, n_filters_out):
            super(ECGResNet.ResBlock, self).__init__()
            kernel_size1 = 11
            self.padding1 = (kernel_size1 - 1) // 2

            self.conv1 = nn.Sequential(
                nn.Conv1d(n_filters_in, n_filters_in, kernel_size=kernel_size1, padding=self.padding1),
                nn.BatchNorm1d(num_features=n_filters_in, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU()
            )
            kernel_size2 = 7
            self.padding2 = (kernel_size2 - 1) // 2
            self.conv2 = nn.Sequential(
                nn.Conv1d(n_filters_in, n_filters_in, kernel_size=kernel_size2, padding=self.padding2),
                nn.BatchNorm1d(num_features=n_filters_in, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU()
            )
            kernel_size3 = 3
            self.padding3 = (kernel_size3 - 1) // 2
            self.conv3 = nn.Sequential(
                nn.Conv1d(n_filters_in, n_filters_out, kernel_size=kernel_size3, padding=self.padding3),
                nn.BatchNorm1d(num_features=n_filters_out, eps=1e-05, momentum=0.1, affine=True),
            )
            kernel_size_sk = 1
            self.sk_conv1 = nn.Sequential(
                nn.Conv1d(n_filters_in, n_filters_out, kernel_size=kernel_size_sk),
                nn.BatchNorm1d(num_features=n_filters_out, eps=1e-05, momentum=0.1, affine=True),
            )
            self.relu4 = nn.ReLU()
            self.pool5 = nn.MaxPool1d(kernel_size=2)

        def forward(self, inputs):
            y = inputs
            x = inputs
            y = self.sk_conv1(y)        
            x = self.conv1(x)        
            x = self.conv2(x)        
            x = self.conv3(x)        
            x = x + y
            x = self.relu4(x)
            x = self.pool5(x)
            return x

