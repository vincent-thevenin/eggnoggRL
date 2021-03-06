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

#params
min_I = 1e-5
max_steps = 1000
gamma = exp(log(min_I)/max_steps)
print(gamma)
path_to_chkpt = 'weights.tar'
cpu = torch.device('cpu') #pylint: disable=no-member
gpu = torch.device('cuda:0') #pylint: disable=no-member

#networks
P = Policy()
V = Value()
need_pretrained = not os.path.isfile(path_to_chkpt)
gym = EggnoggGym(need_pretrained, gpu) #network in gym.observation

#performance measures
Perf_p = PerfPolicy()
Perf_v = PerfValue()

#info
episode = 1
episode_len = []

#init save upon new start
if need_pretrained:
    """print('Initializing weights...')
    torch.save({
            'episode': episode,
            'episode_len': episode_len,
            'P_state_dict': P.state_dict(),
            'V_state_dict': V.state_dict(),
            'O_state_dict': gym.observation.state_dict()#,
            #'optimizerP': optimizerP.state_dict(),
            #'optimizerV': optimizerV.state_dict()
            }, path_to_chkpt)
    print('...Done')"""

#load weights
else:
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    episode = checkpoint['episode']
    episode_len = checkpoint['episode_len']
    P.load_state_dict(checkpoint['P_state_dict'])
    V.load_state_dict(checkpoint['V_state_dict'])
    gym.observation.load_state_dict(checkpoint['O_state_dict'])
    

P.to(gpu)
V.to(gpu)
gym.observation.to(gpu)

#optimizer
optimizerP = optim.SGD(params= list(P.parameters()),
                        lr=1e-5)
optimizerV = optim.SGD(params=list(V.parameters())+list(gym.observation.parameters()),
                        lr=4e-5)


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

G_idx = 0
try:
    while True:
        #INITS
        is_terminal = False
        
        steps = green_reward_sum = 0
        I = 1
        obs = gym.observation(gym.states)
        G_idx = (G_idx+1)%2

        with torch.enable_grad():
            while not is_terminal:
                start = datetime.now()

                steps += 1
                actions = P(obs)
                obs_new, reward, is_terminal = gym.step(actions)
                if steps%max_steps == 0:
                    is_terminal = True

                v_old = V(obs)
                with torch.autograd.no_grad():
                    #calculate delta_t: R_t+1 + gamma*V(S#I = max(1e-5, gamma*I)b,ds_t+1) - V(S_t)
                    if not is_terminal:
                        v_new = V(obs_new)
                        delta = reward[G_idx] + gamma*v_new[G_idx] - v_old[G_idx].detach()
                    else:
                        delta = reward[G_idx] - v_old[G_idx]

                perf_v = Perf_v(delta, v_old, G_idx)
                #(-perf_v).backward() #gradient ascent
                
                perf_p = Perf_p(delta, I, actions, gym.prev_action, G_idx)
                #(-perf_p).backward() #gradient ascent
                
                perf = -(perf_v + perf_p) #gradient ascent
                perf.backward()
                optimizerV.step()
                optimizerP.step()


                optimizerV.zero_grad()
                optimizerP.zero_grad()

                I *= gamma
                obs = obs_new

                stop = datetime.now()
                """for a in actions:
                    print(a)"""
                print('delta:',delta.item(), '\n',
                    'v_old:',v_old, '\n',
                    'v_new:',v_new, '\n',
                    'steps:',steps)
                """if reward[0] > reward[1]:
                    print('G')
                elif reward[0] == reward[1]:
                    print('E')
                else:
                    print('R')"""
                green_reward_sum += I*reward[0]
                print('green_reward_sum:', green_reward_sum)
                print('undiscounted_reward:', reward[0])
                print('time/step:',stop-start)
                print()
        episode += 1
        print(episode, steps)
        episode_len.append(steps)
        print('Finished')
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
except KeyboardInterrupt:
    gym.reset()
    sys.exit()
