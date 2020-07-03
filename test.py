from gym import EggnoggGym
from network import Policy, Value
import torch.optim as optim
import os
import torch
from tqdm import tqdm
from datetime import datetime
from time import sleep
from loss import PerfPolicy, PerfValue
from math import sqrt, exp, log
import sys
import pygame
from torch.distributions.categorical import Categorical

V1 = Value().to(torch.device('cuda:0'))

optimizerV = optim.SGD(params=list(V1.parameters()),
                        lr=1e-2)

while True:
    v = V1(0,0,0)
    ((10-v)**2).backward()
    optimizerV.step()
    optimizerV.zero_grad()
    print(v)