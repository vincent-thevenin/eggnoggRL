import torch.nn as nn
import torch

class perf_policy(nn.Module):
    def __init__(self):
        super(perf_policy, self).__init__()
    def forward(self, delta, I, actions_prob, actions_choice):
        """returns performance
        Args:
            delta: no_grad float
            I: gamma**t no_grad float
            actions_prob: tuple of actions probabilities from policy
            actions_choice: list of tensors from gym.prev_actions"""
        
        perf = torch.ones(1, requires_grad=True)

        for i in range(2): #x, y actions
            perf = perf * actions_prob[i][0, actions_choice[i][0]]
            perf = perf / actions_prob[i][1, actions_choice[i][1]]

        #True -> p(choice)
        #False -> 1-p(choice) because sigmoid gives probs of jump and stab
        for i in range(2,4):
            perf = perf * (~actions_choice[i][0] + (2*actions_choice[i][0]-1) * actions_prob[i][0])
            perf = perf / (~actions_choice[i][1] + (2*actions_choice[i][1]-1) * actions_prob[i][1])

        perf = torch.log(perf)
        return I * delta * perf

class perf_value(nn.Module):
    def __init__(self):
        super(perf_value, self).__init__()
    
    def forward(self, delta, v_old):
        return delta * v_old