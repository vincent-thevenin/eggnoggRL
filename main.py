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
min_I = 1e-3
max_steps = 2000
gamma = exp(log(min_I)/max_steps)
print(gamma)
path_to_chkpt = 'weights.tar'
cpu = torch.device('cpu') #pylint: disable=no-member
gpu = torch.device('cuda:0') #pylint: disable=no-member
lib_path = "libeggnogg/bin/libeggnogg.so"
executable_path = "eggnoggplus-linux/eggnoggplus"

#networks
P1 = Policy()
P2 = Policy()
V1 = Value()
V2 = Value()
need_pretrained = not os.path.isfile(path_to_chkpt)

#performance measures
Perf_p = PerfPolicy()
Perf_v = PerfValue()

#info
episode = 1
episode_len = []

#load weights if backup exists
if not need_pretrained:
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    episode = checkpoint['episode']
    episode_len = checkpoint['episode_len']
    P1.load_state_dict(checkpoint['P1_state_dict'])
    P2.load_state_dict(checkpoint['P2_state_dict'])
    V1.load_state_dict(checkpoint['V1_state_dict'])
    V2.load_state_dict(checkpoint['V2_state_dict'])

P1.to(gpu)
P2.to(gpu)
V1.to(gpu)
V2.to(gpu)

#optimizer
optimizerP = optim.SGD(params= list(P1.parameters())+list(P2.parameters()),
                        lr=1e0)
optimizerV = optim.SGD(params=list(V1.parameters())+list(V2.parameters()),
                        lr=1e-2)


#############################################################################
#one-step actor critic
##############################################################################

#init gym
gym = EggnoggGym(need_pretrained, gpu, lib_path, executable_path, speed=600, seq_len=16)
try:
    while True:
        #INITS
        is_terminal = False
        
        steps = green_reward_sum = 0
        I = 1
        state = gym.states
        state = state.to(gpu)

        with torch.enable_grad():
            while not is_terminal:
                start = datetime.now()

                steps += 1
                actions1 = P1(state.detach())
                actions2 = P2(state.detach())
                state_new, reward, is_terminal = gym.step(actions1, actions2)
                state_new = state_new.to(gpu)
                if steps%max_steps == 0:
                    is_terminal = True

                v1_old = V1(state.detach())
                v2_old = V2(state.detach())
                with torch.autograd.no_grad():
                    #calculate delta_t: R_t+1 + gamma*V(S#I = max(1e-5, gamma*I)b,ds_t+1) - V(S_t)
                    if not is_terminal:
                        v1_new = V1(state_new)
                        delta1 = reward[0] + gamma*v1_new - v1_old.detach()
                        v2_new = V2(state_new)
                        delta2 = reward[1] + gamma*v2_new - v2_old.detach()
                    else:
                        delta1 = reward[0] - v1_old
                        delta2 = reward[1] - v2_old

                perf_v = Perf_v(delta1, v1_old) + Perf_v(delta2, v2_old)
                #(-perf_v).backward() #gradient ascent
                
                perf_p = Perf_p(delta1, I, actions1, gym.prev_action, 0) + Perf_p(delta2, I, actions2, gym.prev_action, 1)
                #(-perf_p).backward() #gradient ascent
                
                perf = -(perf_v + perf_p) #gradient ascent
                perf.backward()
                optimizerV.step()
                optimizerP.step()


                optimizerV.zero_grad()
                optimizerP.zero_grad()

                I *= gamma
                state = state_new

                stop = datetime.now()
                """for a in actions1:
                    print(a)
                for a in actions2:
                    print(a)
                print('delta2:',delta2.item(), '\n',
                    'v2_old:',v2_old.item(), '\n',
                    'v2_new:',v2_new.item(), '\n',
                    'steps:',steps)"""
                """if reward[0] > reward[1]:
                    print('G')
                elif reward[0] == reward[1]:
                    print('E')
                else:
                    print('R')"""
                """green_reward_sum += I*reward[1]
                print('red_reward_sum:', green_reward_sum)
                print('undiscounted_reward:', reward[1])"""
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
                'P1_state_dict': P1.state_dict(),
                'P2_state_dict': P2.state_dict(),
                'V1_state_dict': V1.state_dict(),
                'V2_state_dict': V2.state_dict(),
                'optimizerP': optimizerP.state_dict(),
                'optimizerV': optimizerV.state_dict()
                }, path_to_chkpt)
        print('...Done')
except KeyboardInterrupt:
    gym.reset()
    sys.exit()
