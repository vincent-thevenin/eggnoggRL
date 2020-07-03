import torch.nn as nn
import torch

class PerfPolicy(nn.Module):
    def __init__(self, device):
        super(PerfPolicy, self).__init__()
        self.device = device
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
            encoding = torch.zeros(11).to(self.device)
            #x move
            g_eps = g(U.sample().to(self.device).requires_grad_(), torch.exp(actions_prob[0]).squeeze())
            encoding[int(g_eps)] = g_eps +1- int(g_eps)
            #y move
            g_eps = g(U.sample().to(self.device).requires_grad_(), torch.exp(actions_prob[1]).squeeze())
            encoding[int(g_eps)+3] = g_eps +1- int(g_eps)
            #jump
            g_eps = g(U.sample().to(self.device).requires_grad_(), [1-torch.exp(actions_prob[2]).squeeze(), torch.exp(actions_prob[2].squeeze())])
            encoding[int(g_eps)+3+3] = g_eps +1- int(g_eps)
            #stab
            g_eps = g(U.sample().to(self.device).requires_grad_(), torch.exp(actions_prob[3]).squeeze())
            encoding[int(g_eps)+3+3+2] = g_eps +1- int(g_eps)
            print(encoding[0:3])
            encoding = encoding.unsqueeze(0)

            mu += V(states, gym_map, encoding)
        return -mu/N #gradient ascent
        

class PerfValue(nn.Module):
    def __init__(self):
        super(PerfValue, self).__init__()
    
    def forward(self, delta, v_old):
        return delta * v_old