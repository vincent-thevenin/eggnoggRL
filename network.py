import torch
import torch.nn as nn
from torchvision.models import resnet18

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        #b, 8, 222
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(
            input_size=222,
            hidden_size=512, 
            batch_first=True, 
            bidirectional=False)

        self.lin1 = nn.Linear(512, 1024)
        #b, 1024
        self.lin2 = nn.Linear(1024, 512)
        #b, 512
        self.lin3 = nn.Linear(512, 256)
        #b, 256
        self.lin4 = nn.Linear(256,128)
        #b, 128
        self.lin5 = nn.Linear(128, 64)
        #b, 64
        self.lin6 = nn.Linear(64,8)
        #b,8

    def forward(self, state):
        #b,8,2
        _, (out, _) = self.lstm(state)
        #b,1,512

        out = out.squeeze(1)
        #b,512

        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
        out = self.relu(out)
        out = self.lin4(out)
        out = self.relu(out)
        out = self.lin5(out)
        out = self.relu(out)
        out = self.lin6(out)
        #b, 8

        out = out.split([3,3,1,1], dim=1)
        #left, right, noop
        out1 = self.softmax(out[0]) #b,3
        #up, down, noop
        out2 = self.softmax(out[1]) #b,3

        #jump press, release
        out3 = self.sigmoid(out[2]) #b,1
        #stab press, release
        out4 = self.sigmoid(out[3]) #b,1

        return (out1, out2, out3, out4)

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        #b, 8, 222
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(
            input_size=222,
            hidden_size=512, 
            batch_first=True, 
            bidirectional=False)

        self.lin1 = nn.Linear(512, 1024)
        #b, 1024
        self.lin2 = nn.Linear(1024, 512)
        #b, 512
        self.lin3 = nn.Linear(512, 256)
        #b, 256
        self.lin4 = nn.Linear(256,128)
        #b, 128
        self.lin5 = nn.Linear(128, 64)
        #b, 64
        self.lin6 = nn.Linear(64,1)
        #b,1

    def forward(self, state):
        #b,8,2
        _, (out, cn) = self.lstm(state)
        #b,1,512

        out = out.squeeze(1)
        #b,512

        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
        out = self.relu(out)
        out = self.lin4(out)
        out = self.relu(out)
        out = self.lin5(out)
        out = self.relu(out)
        out = self.lin6(out)
        #b, 1
        #out = self.tanh(out)
        #b,1

        out = out.squeeze()
        #b

        return out