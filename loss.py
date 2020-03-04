import torch.nn as nn
import torch

class perf_policy(nn.Module):
    def __init__(self):
        super(perf_policy, self).__init__()
        self.eps = 1e-6
    def forward(self, delta, I, actions_prob, actions_choice):
        """returns performance
        Args:
            delta: no_grad float
            I: gamma**t no_grad float
            actions_prob: tuple of actions probabilities from policy
            actions_choice: list of tensors from gym.prev_actions"""
        
        perf_player = 1
        perf_advers = 1

        for i in range(2): #x, y actions
            perf_player = perf_player * actions_prob[i][0, actions_choice[i][0]]
            perf_advers = perf_advers * actions_prob[i][1, actions_choice[i][1]]

        #True -> p(choice)
        #False -> 1-p(choice) because sigmoid gives probs of jump and stab
        for i in range(2,4):
            perf_player = perf_player * (~actions_choice[i][0] + (2*actions_choice[i][0]-1) * actions_prob[i][0])
            perf_advers = perf_advers * (~actions_choice[i][1] + (2*actions_choice[i][1]-1) * actions_prob[i][1])

        perf_player = perf_player + self.eps
        perf_advers = perf_advers + self.eps

        perf = torch.log(perf_player / perf_advers) #pylint: disable=no-member
        return I * delta * perf

class perf_value(nn.Module):
    def __init__(self):
        super(perf_value, self).__init__()
    
    def forward(self, delta, v_old):
        return delta * v_old