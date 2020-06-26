import torch
import torch.nn as nn
from torchvision.models import resnet18
    
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        #b, 8, 106
        self.relu = nn.LeakyReLU()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.eps = 1e-6

        #b,8,12,33
        self.conv1 = nn.Conv2d(8, 4, (3,3), padding=1)
        self.pool1 = nn.AvgPool2d((2,2))
        #b,4,6,16
        self.conv2 = nn.Conv2d(4, 2, (3,3), padding=1)
        self.pool2 = nn.AvgPool2d((2,2))
        #b,2,3,8
        self.conv3 = nn.Conv2d(2, 1, (3,3), padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1,8))
        #b,1,1,8

        self.lstm = nn.LSTM(
            input_size=106,
            hidden_size=504, 
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
        self.lin6 = nn.Linear(64,10)
        #b,10

    def forward(self, state, map):
        #b, 8, 12, 33
        out_map = self.conv1(map)
        out_map = self.pool1(out_map)
        out_map = self.relu(out_map)
        out_map = self.conv2(out_map)
        out_map = self.pool2(out_map)
        out_map = self.relu(out_map)
        out_map = self.conv3(out_map)
        out_map = self.pool3(out_map)
        out_map = self.relu(out_map)
        out_map = out_map.reshape(out_map.shape[0], 8)
        #b,8

        #b,8,2
        _, (out, _) = self.lstm(state)
        #b,1,506

        out = out.squeeze(1)
        #b,506
        out = torch.cat((out, out_map), dim=1)
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
        out = self.relu(out)
        #b, 10

        out = out.split([3,3,1,3], dim=1)
        #left, right, noop
        out1 = self.logsoftmax(out[0]) #b,3
        #up, down, noop
        out2 = self.logsoftmax(out[1]) #b,3

        #jump press & release, noop
        out3 = self.logsigmoid(out[2]) #b,1
        #stab press, stab throw, noop
        out4 = self.logsoftmax(out[3]) #b,3

        return (out1, out2, out3, out4) #4, b, 3

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        #b, 8, 222
        self.relu = nn.LeakyReLU()

        #b,8,12,33
        self.conv1 = nn.Conv2d(8, 4, (3,3), padding=1)
        self.pool1 = nn.AvgPool2d((2,2))
        #b,4,6,16
        self.conv2 = nn.Conv2d(4, 2, (3,3), padding=1)
        self.pool2 = nn.AvgPool2d((2,2))
        #b,2,3,8
        self.conv3 = nn.Conv2d(2, 1, (3,3), padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1,8))
        #b,1,1,8

        self.lstm = nn.LSTM(
            input_size=106,
            hidden_size=494, 
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

    def forward(self, state, map, actions):
        #b, 8, 12, 33
        out_map = self.conv1(map)
        out_map = self.pool1(out_map)
        out_map = self.relu(out_map)
        out_map = self.conv2(out_map)
        out_map = self.pool2(out_map)
        out_map = self.relu(out_map)
        out_map = self.conv3(out_map)
        out_map = self.pool3(out_map)
        out_map = self.relu(out_map)
        out_map = out_map.reshape(out_map.shape[0], 8)
        #b,8

        #b,8,2
        _, (out, _) = self.lstm(state)
        #b,1,494

        out = out.squeeze(1)
        #b,494
        out = torch.cat((out, out_map), dim=1)
        #b,502
        for action in actions:
            out = torch.cat((out, action), dim=1)
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