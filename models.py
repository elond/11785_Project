import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(self, input_rows, input_len, first_hu, second_hu, third_hu, CAT_encode):
        super(LSTMNet, self).__init__()
        self.CAT_encode = CAT_encode
        input_rows = input_len
        if CAT_encode:
            input_col = 3
            self.lstm1 = nn.LSTM(input_col, first_hu, batch_first=True)
            self.lstm2 = nn.LSTM(first_hu, second_hu, batch_first=True)
            self.fc1 = nn.Linear(input_len* 21* second_hu, 9)
        else:
            input_col = 4
            self.lstm1 = nn.LSTM(input_col, first_hu, batch_first=True)
            self.lstm2 = nn.LSTM(first_hu, second_hu, batch_first=True)
            self.lstm3 = nn.LSTM(second_hu, third_hu, batch_first=True)
            self.fc1 = nn.Linear(input_len * third_hu, 9)

    def forward(self, x):
        if self.CAT_encode:
            x = torch.flatten(x, start_dim=2)
        x = x.permute(0,2,1)
        x = x.float()
        x = F.relu(self.lstm1(x)[0])
        x = F.relu(self.lstm2(x)[0])
        if not self.CAT_encode:
            x = F.relu(self.lstm3(x)[0])
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


class CNNLSTMNet(nn.Module):
    def __init__(self, input_rows, input_len, first_hu, second_hu, third_hu, CAT_encode):
        super(CNNLSTMNet, self).__init__()
        input_rows = input_len
        stride = 1
        padding = 0
        self.CAT_encode = CAT_encode
        if CAT_encode:
            window = 1
            input_col = 3
            self.conv = nn.Conv2d(input_col,first_hu,(21,window),stride,padding)
            self.out_dim = int((input_rows + 2*padding - window)/stride + 1)
            self.lstm1 = nn.LSTM(first_hu, second_hu, batch_first=True)
            self.fc1 = nn.Linear(self.out_dim * second_hu, 9)
        else:
            window = 3
            input_col = 4
            self.conv = nn.Conv1d(input_col,first_hu,window,stride,padding)
            self.out_dim = int((input_rows + 2*padding - window)/stride + 1)
            self.lstm1 = nn.LSTM(first_hu, second_hu, batch_first=True)
            self.lstm2 = nn.LSTM(second_hu, third_hu, batch_first=True)
            self.fc1 = nn.Linear(self.out_dim * third_hu, 9)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv(x))
        if self.CAT_encode:
            x = torch.flatten(x, start_dim=2)
        x = x.permute(0,2,1)
        x = F.relu(self.lstm1(x)[0])
        if not self.CAT_encode:
            x = F.relu(self.lstm2(x)[0])
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


class CNNNet(nn.Module):
    def __init__(self, input_rows, input_len, conv1_hu, conv2_hu, conv3_hu, CAT_encode):
        super(CNNNet, self).__init__()
        conv_input_rows = input_len
        stride = 1
        padding = 0
        self.CAT_encode = CAT_encode
        if self.CAT_encode:
            window = 1
            conv_input_col = 3
            self.conv1 = nn.Conv2d(conv_input_col,conv1_hu,(21,window),stride,padding)
            self.out_dim = int((conv_input_rows + 2*padding - window)/stride + 1)
            self.conv2 = nn.Conv2d(conv1_hu, conv2_hu,window,stride)
            self.out_dim = int((self.out_dim - window)/stride + 1)
            self.fc1 = nn.Linear(self.out_dim*conv2_hu, 9)

        else:
            window = 3
            conv_input_col = 4
            self.conv1 = nn.Conv1d(conv_input_col,conv1_hu,window,stride,padding)
            self.out_dim = int((conv_input_rows + 2*padding - window)/stride + 1)
            self.conv2 = nn.Conv1d(conv1_hu, conv2_hu,window,stride)
            self.out_dim = int((self.out_dim - window)/stride + 1)
            self.conv3 = nn.Conv1d(conv2_hu, conv3_hu, window, stride)
            self.out_dim = int((self.out_dim - window)/stride + 1)
            self.fc1 = nn.Linear(self.out_dim*conv3_hu, 9)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if not self.CAT_encode:
            x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x
