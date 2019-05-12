

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class given_net_1(nn.Module):
    def __init__(self):
        super(given_net_1, self).__init__()
        #layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0, stride=1)
        self.bn_1 = nn.BatchNorm2d(num_features=8)
        self.relu_1 = nn.ReLU()

        # layer 2
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer 3
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0, stride=1)
        self.bn_2 = nn.BatchNorm2d(num_features=16)
        self.relu_2 = nn.ReLU()

        # layer 4
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer 5
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0, stride=1)
        self.bn_3 = nn.BatchNorm2d(num_features=32)
        self.relu_3 = nn.ReLU()

        # layer 6
        self.fc1 = nn.Linear(in_features=288, out_features=10)
        self.SM1 = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu_1(self.bn_1(x))
        x = self.pool_1(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.relu_2(self.bn_2(x))
        x = self.pool_2(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.relu_3(self.bn_3(x))

        # print(x.shape)

        x = x.view(-1, 288)
        x = self.fc1(x)

        # print(x.shape)
        return self.SM1(x)

class found_net_1(nn.Module):
    def __init__(self):
        super(found_net_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)