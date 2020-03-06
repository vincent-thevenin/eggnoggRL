import torch
import torch.nn as nn
from torchvision.models import resnet18

class Observation(nn.Module):
    def __init__(self, need_pretrained=False):
        super(Observation, self).__init__()
        self.relu = nn.LeakyReLU()

        self.conv3d1 = nn.Conv3d(3, 3, (4, 3,3), padding=(0,1,1))
        self.pool1 = nn.AvgPool2d(2)

        self.resnet = resnet18(pretrained=need_pretrained)
    
    def forward(self, states):
        #b,3,4,80,120
        out = self.conv3d1(states)
        #b,3,1,80,120
        out = out.squeeze()
        #b, 3, 80,120
        out = self.pool1(out)
        out = self.relu(out)
        #b, 3, 40, 60

        out = self.resnet(out)
        out = self.relu(out)
        #b, 1000

        return out

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        #b, 3, 320,480
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        """self.conv1 = nn.utils.spectral_norm(nn.Conv2d(3, 32, 3, padding=1)) 
        #b,32,480,320
        self.pool1 = nn.AvgPool2d(2) 
        #b,32,240,160

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, padding=1)) 
        #b,64,240,160
        self.pool2 = nn.AvgPool2d(2) 
        #b,64,120,80

        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, padding=1)) 
        #b, 128, 120, 80
        self.pool3 = nn.AvgPool2d(2) 
        #b, 128, 60, 40

        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, padding=1)) 
        #b, 128, 60, 40
        self.pool4 = nn.AvgPool2d(2) 
        #b, 128, 30, 20

        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, padding=1)) 
        #b, 128, 30, 20
        self.pool5 = nn.AdaptiveAvgPool2d(output_size=(1, 1)) 
        #b, 128, 1, 1

        #view
        #b, 128
        #unsqueeze
        #b, 1, 128*15*10"""

        self.lin1 = nn.Linear(1000, 200)
        #b, 200
        self.lin2 = nn.Linear(200, 40)
        #b, 40
        self.lin3 = nn.Linear(40, 8)
        #b, 8

    def forward(self, observation):
        """
        out = self.conv1(state)
        out = self.pool1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.pool2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.pool3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.pool4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.pool5(out)
        out = self.relu(out)

        out = out.view(-1, 128*15*10)
        out = out.unsqueeze(1)"""
        
        out = self.lin1(observation)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
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
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.lin1 = nn.Linear(1000, 100)
        #b, 1000
        self.lin2 = nn.Linear(100, 10)
        #b, 100
        self.lin3 = nn.Linear(10, 1)
        #b, 1

    def forward(self, observation_normal):
        out = self.lin1(observation_normal)
        out = self.relu(out)

        out = self.lin2(out)
        out = self.relu(out)

        out = self.lin3(out)
        #out = self.tanh(out)
        #b,1

        out = out.squeeze(-1)
        #b

        return out