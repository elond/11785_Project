class LSTMNet(nn.Module):
    def __init__(self, input_rows, input_len, first_hu, second_hu, third_hu):
        super(LSTMNet, self).__init__()
        input_rows = input_len # 250 for One_hot 83 for CAT
        input_col = 4 # 4 for One_hot and 3 if CAT
        self.lstm1 = nn.LSTM(input_col, first_hu, batch_first=True)
        self.lstm2 = nn.LSTM(first_hu, second_hu, batch_first=True)
        self.lstm3 = nn.LSTM(second_hu, third_hu, batch_first=True)
        self.fc1 = nn.Linear(input_len * third_hu, 8)
        
    def forward(self, x):
        # input size is (batch, 4 or 3, 250 or 83)
        x = x.float()
        x = x.permute(0, 2, 1)
        x = F.relu(self.lstm1(x)[0])
        x = F.relu(self.lstm2(x)[0])
        x = F.relu(self.lstm3(x)[0])
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


class CNNLSTMNet(nn.Module):
    def __init__(self, input_rows, input_len, first_hu, second_hu, third_hu, window):
        super(CNNLSTMNet, self).__init__()
        input_rows = input_len # 250 for One_hot 83 for CAT
        input_col = 4 # 4 for One_hot and 3 if CAT
        stride = 1
        padding = 0
        self.conv = nn.Conv1d(input_col,21,window,stride,padding) 
        self.out_dim = int((input_rows + 2*padding - window)/stride + 1)
        self.lstm1 = nn.LSTM(21, first_hu, batch_first=True)
        self.lstm2 = nn.LSTM(first_hu, second_hu, batch_first=True)
        self.lstm3 = nn.LSTM(second_hu, third_hu, batch_first=True)
        self.fc1 = nn.Linear(self.out_dim * third_hu, 8)
        
    def forward(self, x):
        # input size is (batch, 4 or 3, 250 or 83)
        x = x.float()
        x = F.relu(self.conv(x))
        x = x.permute(0,2,1)
        x = F.relu(self.lstm1(x)[0])
        x = F.relu(self.lstm2(x)[0])
        x = F.relu(self.lstm3(x)[0])
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x