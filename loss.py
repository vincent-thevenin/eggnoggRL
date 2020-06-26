import torch.nn as nn
import torch

class PerfPolicy(nn.Module):
    def __init__(self):
        super(PerfPolicy, self).__init__()
        self.eps = 1e-6
    def forward(self, V, states, gym_map, U, g, actions_prob, N=1):
        """returns gradient in the sense of pathwise MC gradient
        Args:
            V: Q-function of the policy being optimized
            states: 1st arg for V and P
            gym_map: 2nd arg for V and P
            U: Uniform distribution 
            g: path fucntion from pathwise gradient theorem
            actions_prob: parameters of distribution from policy,
            N: number of sampling from Uniform distribution for estimation"""
        mu = 0
        for _ in range(N):
            eps = U.sample()
            actions = [
                g(U.sample(), torch.exp(actions_prob[0]).squeeze()),
                g(U.sample(), torch.exp(actions_prob[1]).squeeze()),
                g(U.sample(), [1-torch.exp(actions_prob[2]).squeeze(), torch.exp(actions_prob[2].squeeze())]),
                g(U.sample(), torch.exp(actions_prob[3]).squeeze())
            ]

            mu += V(states, gym_map, actions)
        return -mu/N #gradient ascent
        

class PerfValue(nn.Module):
    def __init__(self):
        super(PerfValue, self).__init__()
    
    def forward(self, delta, v_old):
        return delta * v_old