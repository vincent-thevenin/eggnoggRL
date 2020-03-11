import torch.nn as nn
import torch

class PerfPolicy(nn.Module):
    def __init__(self):
        super(PerfPolicy, self).__init__()
        self.eps = 1e-6
    def forward(self, I, actions_prob, actions_choice, G_idx):
        """returns performance
        Args:
            delta: no_grad float
            I: gamma**t no_grad float
            actions_prob: tuple of actions probabilities from policy
            actions_choice: list of tensors from gym.prev_actions"""
        
        perf_player = 1

        for i in range(2): #x, y actions
            perf_player = perf_player + actions_prob[i][0, actions_choice[i][G_idx]]

        #True -> p(choice)
        #False -> 1-p(choice) because sigmoid gives probs of jump and stab
        perf_player = perf_player + (~actions_choice[2][G_idx] + (2*actions_choice[2][G_idx]-1) * actions_prob[2][0, 0])

        perf_player = perf_player + actions_prob[3][0, actions_choice[3][G_idx]]
        
        return I * perf_player

class PerfValue(nn.Module):
    def __init__(self):
        super(PerfValue, self).__init__()
    
    def forward(self, delta, v_old):
        return delta * v_old