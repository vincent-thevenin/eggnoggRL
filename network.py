import torch
import torch.nn as nn
from torchvision.models import resnet18
    
class Policy(nn.Module):
    def __init__(self, epsilon=0.05):
        super(Policy, self).__init__()
        self.epsilon = epsilon

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

        self.lin1 = nn.Linear(512, 512)
        #b, 1024
        self.lin2 = nn.Linear(512, 512)
        #b, 512
        self.lin3 = nn.Linear(512, 512)
        #b, 256
        self.lin4 = nn.Linear(512, 512)
        #b, 128
        self.lin5 = nn.Linear(512, 512)
        #b, 64
        self.lin6 = nn.Linear(512,10)
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

        out1 = self.lin1(out)
        out = self.relu(out1) + out
        out2 = self.lin2(out)
        out = self.relu(out2) + out
        out3 = self.lin3(out)
        out = self.relu(out3) + out
        out4 = self.lin4(out)
        out = self.relu(out4)
        out5 = self.lin5(out)
        out = self.relu(out5)
        out = self.lin6(out)
        #out = self.relu(out)
        #b, 10

        out = out.split([3,3,1,3], dim=1)
        #left, right, noop
        out1 = self.logsoftmax(out[0]) #b,3
        if torch.exp(out1.min()) <= self.epsilon:
            out1 = torch.exp(out1)
            add = out1 <= self.epsilon
            out1 = out1 + add.type(out1.type())*self.epsilon - (~add).type(out1.type())*self.epsilon*add.sum()/(~add).sum()
            out1 = torch.log(out1)

        #up, down, noop
        out2 = self.logsoftmax(out[1]) #b,3
        if torch.exp(out2.min()) <= self.epsilon:
            out2 = torch.exp(out2)
            add = out2 <= self.epsilon
            out2 = out2 + add.type(out2.type())*self.epsilon - (~add).type(out2.type())*self.epsilon*add.sum()/(~add).sum()
            out2 = torch.log(out2)

        #jump press & release, noop
        out3 = self.logsigmoid(out[2]) #b,1
        if torch.exp(out3) <= self.epsilon or torch.exp(out3) >= 1-self.epsilon:
            out3 = torch.exp(out3)
            add = out3 <= self.epsilon
            out3 = out3 + add.type(out3.type())*self.epsilon - (~add).type(out3.type())*self.epsilon
            out3 = torch.log(out3)

        #stab press, stab throw, noop
        out4 = self.logsoftmax(out[3]) #b,3
        if torch.exp(out4.min()) <= self.epsilon:
            out4 = torch.exp(out4)
            add = out4 <= self.epsilon
            out4 = out4 + add.type(out4.type())*self.epsilon - (~add).type(out4.type())*self.epsilon*add.sum()/(~add).sum()
            out4 = torch.log(out4)

        return (out1, out2, out3, out4) #4, b, 3

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        #b, 8, 222
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        #b,8,12,33
        self.conv1 = nn.Conv2d(8, 4, (3,3), padding=1)
        self.pool1 = nn.AvgPool2d((2,2))
        #b,4,6,16
        self.conv2 = nn.Conv2d(4, 2, (3,3), padding=1)
        self.pool2 = nn.AvgPool2d((2,2))
        self.instance_norm = nn.InstanceNorm2d(2)
        #b,2,3,8
        self.conv3 = nn.Conv2d(2, 1, (3,3), padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1,8))
        #b,1,1,8

        self.lstm = nn.LSTM(
            input_size=106,
            hidden_size= 504, 
            batch_first=True, 
            bidirectional=False)

        self.lin1 = nn.Linear(512, 512)
        #b, 1024
        self.lin2 = nn.Linear(512, 512)
        #b, 512
        self.lin3 = nn.Linear(512, 512)
        #b, 256
        self.lin4 = nn.Linear(512, 512)
        #b, 128
        self.lin5 = nn.Linear(512, 512)
        #b, 64
        self.lin6 = nn.Linear(512,1)
        #b,1

        self.linact = nn.Linear(11,512)
    def forward(self, state, map, actions):
        #b, 8, 12, 33
        out_map = self.conv1(map)
        out_map = self.pool1(out_map)
        out_map = self.relu(out_map)
        out_map = self.conv2(out_map)
        out_map = self.pool2(out_map)
        out_map = self.instance_norm(out_map)
        out_map = self.relu(out_map)
        out_map = self.conv3(out_map)
        out_map = self.pool3(out_map)
        out_map = self.relu(out_map)
        out_map = out_map.reshape(out_map.shape[0], 8)
        #b,8

        #b,8,2
        _, (out, _) = self.lstm(state)
        #b,1,504

        out = out.squeeze(1)
        #b,504
        out = torch.cat((out, out_map), dim=1)
        #b,512
        
        out_actions = self.linact(actions)
        out_actions = self.relu(out_actions)
        #b,512
        
        out1 = self.lin1(out+out_actions)
        out = self.relu(out1)
        out2 = self.lin2(out)
        out = self.relu(out2) + out
        out3 = self.lin3(out)
        out = self.relu(out3) + out
        out4 = self.lin4(out)
        out = self.relu(out4) + out
        out5 = self.lin5(out)
        out = self.relu(out5)
        out = self.lin6(out)
        #b, 1

        out = out.squeeze()
        #b

        return out