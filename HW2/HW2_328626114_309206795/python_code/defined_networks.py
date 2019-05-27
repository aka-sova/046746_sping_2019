

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Train_History():
    def __init__(self, n_epochs, train_dataset_length):

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.train_validation_losses = []
        self.accuracy_hist_test = []
        self.accuracy_hist_train = []
        self.test_counter = [i*train_dataset_length for i in range(n_epochs + 1)]
        self.test_counter_epoch = [i for i in range(n_epochs + 1)]
        self.train_counter_epoch = []



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


class given_net_2(nn.Module):
    def __init__(self):
        super(given_net_2, self).__init__()
        #layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=0, stride=1)
        self.bn_1 = nn.BatchNorm2d(num_features=2)
        self.relu_1 = nn.ReLU()

        # layer 2
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer 3
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=0, stride=1)
        self.bn_2 = nn.BatchNorm2d(num_features=4)
        self.relu_2 = nn.ReLU()

        # layer 4
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer 5
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=0, stride=1)
        self.bn_3 = nn.BatchNorm2d(num_features=8)
        self.relu_3 = nn.ReLU()

        # layer 6
        self.fc1 = nn.Linear(in_features=72, out_features=10)
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

        x = x.view(-1, 72)
        x = self.fc1(x)

        # print(x.shape)
        return self.SM1(x)
