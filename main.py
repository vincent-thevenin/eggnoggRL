from gym import EggnoggGym
from network import Policy, Value
import torch.optim as optim
import os
import torch
from tqdm import tqdm
from datetime import datetime
from time import sleep
from loss import perf_policy, perf_value

#params
gamma = 0.9
path_to_chkpt = 'weights.tar'
cpu = torch.device('cpu') #pylint: disable=no-member
gpu = torch.device('cuda:0') #pylint: disable=no-member

#networks
P = Policy()
V = Value()
need_pretrained = not os.path.isfile(path_to_chkpt)
gym = EggnoggGym(need_pretrained, gpu) #network in gym.observation

#performance measures
Perf_p = perf_policy()
Perf_v = perf_value()

#optimizer
optimizerP = optim.Adam(params= list(P.parameters()) + list(gym.observation.parameters()),
                        lr=1e-4)
optimizerV = optim.Adam(params=list(V.parameters()) + list(gym.observation.parameters()),
                        lr=4e-4)

#info
episode = 0
episode_len = []

#init save upon new start
if need_pretrained:
    print('Initializing weights...')
    torch.save({
            'episode': episode,
            'episode_len': episode_len,
            'P_state_dict': P.state_dict(),
            'V_state_dict': V.state_dict(),
            'O_state_dict': gym.observation.state_dict(),
            'optimizerP': optimizerP.state_dict(),
            'optimizerV': optimizerV.state_dict()
            }, path_to_chkpt)
    print('...Done')

#load weights
else:
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    episode = checkpoint['episode']
    episode_len = checkpoint['episode_len']
    P.load_state_dict(checkpoint['P_state_dict'])
    V.load_state_dict(checkpoint['V_state_dict'])
    gym.observation.load_state_dict(checkpoint['O_state_dict'])
    optimizerP.load_state_dict(checkpoint['optimizerP'])
    optimizerV.load_state_dict(checkpoint['optimizerV'])

P.to(gpu)
V.to(gpu)


#############################################################################
#one-step actor critic
##############################################################################
"""
actions = p(gym.observation(gym.states))
obs, reward, is_terminal = gym.step(actions)
actions = p(obs)

#init noop action
actions = (torch.tensor([[.0,.0,1.0], #pylint: disable=not-callable
                        [.0,.0,1.0]]),
        torch.tensor([[.0,.0,1.0], #pylint: disable=not-callable
                        [.0,.0,1.0]]),
        torch.tensor([[.0], #pylint: disable=not-callable
                        [.0]]),
        torch.tensor([[.0], #pylint: disable=not-callable
                        [.0]])
        )"""

while True:
    #INITS
    is_terminal = False
    steps = 0
    
    I = 1
    obs = gym.observation(gym.states)

    with torch.enable_grad():
        while not is_terminal:
            start = datetime.now()

            steps += 1
            actions = P(obs)
            obs_new, reward, is_terminal = gym.step(actions)
            v_old = V(obs[:1, :])
            with torch.autograd.no_grad():
                v_new = V(obs_new[:1, :])
                #calculate delta_t: R_t+1 + gamma*V(S_t+1) - V(S_t)
                delta = reward[0] + gamma*v_new - v_old.detach()

            perf_v = Perf_v(delta, v_old)
            (-perf_v).backward(retain_graph=True) #gradient ascent
            optimizerV.step()

            perf_p = Perf_p(delta, I, actions, gym.prev_action)
            (-perf_p).backward()
            optimizerP.step()

            optimizerV.zero_grad()
            optimizerP.zero_grad()

            I *= gamma
            obs = obs_new

            stop = datetime.now()
            """print(stop-start)"""
    episode += 1
    print(episode)
    episode_len.append(steps)
    print('reset')
    gym.reset()
    print('Saving weights...')
    torch.save({
            'episode': episode,
            'episode_len': episode_len,
            'P_state_dict': P.state_dict(),
            'V_state_dict': V.state_dict(),
            'O_state_dict': gym.observation.state_dict(),
            'optimizerP': optimizerP.state_dict(),
            'optimizerV': optimizerV.state_dict()
            }, path_to_chkpt)
    print('...Done')
