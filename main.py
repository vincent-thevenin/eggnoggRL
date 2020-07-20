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
from torch.nn.utils import clip_grad_value_

pygame.init()

SIZE = WIDTH, HEIGHT = (400, 750)
screen = pygame.display.set_mode(SIZE, pygame.RESIZABLE)

def blit_text(surface, text, pos, font, color=pygame.Color('white')):
	words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
	space = font.size(' ')[0]  # The width of a space.
	max_width, max_height = surface.get_size()
	x, y = pos
	for line in words:
		for word in line:
			word_surface = font.render(word, 0, color)
			word_width, word_height = word_surface.get_size()
			if x + word_width >= max_width:
				x = pos[0]  # Reset the x.
				y += word_height  # Start on new row.
			surface.blit(word_surface, (x, y))
			x += word_width + space
		x = pos[0]  # Reset the x.
		y += word_height  # Start on new row.

font = pygame.font.SysFont('Arial', 20)

def display_text(text):
    screen.fill(pygame.Color('black'))
    blit_text(screen, text, (20, 20), font)
    pygame.display.update()





#params
min_I = 1e-3
max_steps = 500
lambda_policy = 0.5
lambda_value = 0.0
gamma = 0.1 #exp(log(min_I)/max_steps)
print(gamma)
N = 1
path_to_chkpt = 'weightsPathwise.tar'
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
Perf_p = PerfPolicy(gpu)
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
                        lr=1e-2)
optimizerV = optim.SGD(params=list(V1.parameters())+list(V2.parameters()),
                        lr=1e-3)


#############################################################################
#one-step actor critic
##############################################################################

#init gym
gym = EggnoggGym(need_pretrained, gpu, lib_path, executable_path, speed=120, seq_len=32)
try:
    while True:
        #INITS
        start = datetime.now()
        is_terminal = False
        
        steps = 0
        I = 1
        state = gym.states
        stateP1, stateP2, stateV1, stateV2 = (
            state.to(gpu),
            state.to(gpu),
            state.to(gpu),
            state.to(gpu)
        )
        stateP1.requires_grad_()
        stateP2.requires_grad_()
        stateV1.requires_grad_()
        stateV2.requires_grad_()
        z_policy1 = [0]*len(list(P1.parameters())) #eligibility trace for policy
        z_policy2 = [0]*len(list(P2.parameters()))
        z_value1 = [0]*len(list(V1.parameters()))
        z_value2 = [0]*len(list(V2.parameters())) #eligibility trace for value function

        with torch.enable_grad():
            while not is_terminal:
                steps += 1
                with torch.autograd.no_grad():
                    #calculate actions prob
                    actions1 = P1(stateP1, gym.map)
                    actions2 = P2(stateP2, gym.map)

                    #act on environement
                    map_old = gym.map.detach_().requires_grad_()
                    state_new, reward, is_terminal = gym.step(actions1, actions2)
                    state_new = state_new.to(gpu)
                    if steps%max_steps == 0:
                        is_terminal = True
                
                with torch.autograd.no_grad():
                    encode1 = torch.zeros(11).to(gpu)
                    encode1[int(gym.prev_action[0][0])] = gym.prev_action[0][0]#torch.exp(actions1[0][0,int(gym.prev_action[0][0])])
                    encode1[int(gym.prev_action[1][0])+3] = gym.prev_action[1][0]#torch.exp(actions1[1][0,int(gym.prev_action[1][0])])
                    encode1[int(gym.prev_action[2][0])+3+3] = gym.prev_action[2][0]#[1-torch.exp(actions1[2][0]), torch.exp(actions1[2][0])][int(gym.prev_action[2][0])]
                    encode1[int(gym.prev_action[3][0])+3+3+2] = gym.prev_action[3][0]#torch.exp(actions1[3][0,int(gym.prev_action[3][0])])
                    for i in range(encode1.shape[0]):
                        encode1[i] = int(encode1[i] != 0)*(100+encode1[i])/100
                    encode1 = encode1.unsqueeze(0).requires_grad_()
                    
                    encode2 = torch.zeros(11).to(gpu)
                    encode2[int(gym.prev_action[0][1])] = gym.prev_action[0][1]#torch.exp(actions1[0][0,int(gym.prev_action[0][0])])
                    encode2[int(gym.prev_action[1][1])+3] = gym.prev_action[1][1]#torch.exp(actions1[1][0,int(gym.prev_action[1][0])])
                    encode2[int(gym.prev_action[2][1])+3+3] = gym.prev_action[2][1]#[1-torch.exp(actions1[2][0]), torch.exp(actions1[2][0])][int(gym.prev_action[2][0])]
                    encode2[int(gym.prev_action[3][1])+3+3+2] = gym.prev_action[3][1]#torch.exp(actions1[3][0,int(gym.prev_action[3][0])])
                    for i in range(encode2.shape[0]):
                        encode2[i] = int(encode2[i] != 0)*(100+encode2[i])/100
                    encode2 = encode2.unsqueeze(0).requires_grad_()


                optimizerV.zero_grad()
                v1_old = V1(stateV1, map_old, encode1)#.detach())
                v2_old = V2(stateV2, map_old, encode2)#.detach())
                with torch.autograd.no_grad():
                    #calculate delta_t: R_t+1 + gamma*V(S#I = max(1e-5, gamma*I)b,ds_t+1) - V(S_t)
                    if not is_terminal:
                        actions1_new = P1(state_new, gym.map)
                        actions2_new = P2(state_new, gym.map)
                        
                        encode1_new = torch.zeros(11).to(gpu)
                        y = gym.g(gym.U.sample().to(gpu), torch.exp(actions1_new[0]).squeeze())
                        encode1_new[int(y)] = (y + 100)/100
                        y = gym.g(gym.U.sample().to(gpu), torch.exp(actions1_new[1]).squeeze())
                        encode1_new[int(y)+3] = (y + 100)/100
                        y = gym.g(gym.U.sample().to(gpu), [1-torch.exp(actions1_new[2]).squeeze(), torch.exp(actions1_new[2].squeeze())])
                        encode1_new[int(y)+3+3] = (y + 100)/100
                        y = gym.g(gym.U.sample().to(gpu), torch.exp(actions1_new[3]).squeeze())
                        encode1_new[int(y)+3+3+2] = (y + 100)/100
                        encode1_new = encode1_new.unsqueeze(0).requires_grad_()

                        encode2_new = torch.zeros(11).to(gpu)
                        y = gym.g(gym.U.sample().to(gpu), torch.exp(actions2_new[0]).squeeze())
                        encode2_new[int(y)] = (y + 100)/100
                        y = gym.g(gym.U.sample().to(gpu), torch.exp(actions2_new[1]).squeeze())
                        encode2_new[int(y)+3] = (y + 100)/100
                        y = gym.g(gym.U.sample().to(gpu), [1-torch.exp(actions2_new[2]).squeeze(), torch.exp(actions2_new[2].squeeze())])
                        encode2_new[int(y)+3+3] = (y + 100)/100
                        y = gym.g(gym.U.sample().to(gpu), torch.exp(actions2_new[3]).squeeze())
                        encode2_new[int(y)+3+3+2] = (y + 100)/100
                        encode2_new = encode2_new.unsqueeze(0).requires_grad_()

                        v1_new = V1(state_new, gym.map, encode1_new)
                        delta1 = reward[0] + gamma*v1_new - v1_old#.detach()
                        
                        v2_new = V2(state_new, gym.map, encode2_new)
                        delta2 = reward[1] + gamma*v2_new - v2_old#.detach()
                    else:
                        delta1 = reward[0] - v1_old
                        delta2 = reward[1] - v2_old
                    delta1 = max(min(delta1, 10.0), -10.0)
                    delta2 = max(min(delta2, 10.0), -10.0)

                (-delta1*v1_old).backward()
                """for i,p in enumerate(V1.parameters()):
                    z_value1[i] = gamma*lambda_value*z_value1[i] + p.grad
                    p.grad = -delta1*z_value1[i] #gradient ascent"""

                (-delta2*v2_old).backward()
                """for i,p in enumerate(V2.parameters()):
                    z_value2[i] = gamma*lambda_value*z_value2[i] + p.grad
                    p.grad = -delta2*z_value2[i] #gradient ascent"""

                optimizerV.step()
                optimizerV.zero_grad()

                optimizerP.zero_grad()

                """perf_p1 = Perf_p(I, actions1, gym.prev_action, 0)
                perf_p1.backward()
                for i,p in enumerate(P1.parameters()):
                    z_policy1[i] = gamma*lambda_policy*z_policy1[i] + p.grad
                    p.grad = -delta1*z_policy1[i]

                perf_p2 = Perf_p(I, actions2, gym.prev_action, 1)
                perf_p2.backward()
                for i,p in enumerate(P2.parameters()):
                    z_policy2[i] = gamma*lambda_policy*z_policy2[i] + p.grad
                    p.grad = -delta2*z_policy2[i]"""
                #calculate actions prob
                actions1 = P1(stateP1.detach(), map_old.detach())
                actions2 = P2(stateP2.detach(), map_old.detach())
                perf_p1 = Perf_p(V1, stateP1.detach().requires_grad_(), map_old.detach().requires_grad_(), gym.U, gym.g, actions1, N=N)
                perf_p2 = Perf_p(V2, stateP2.detach().requires_grad_(), map_old.detach().requires_grad_(), gym.U, gym.g, actions2, N=N)
                perf_p1.backward()
                perf_p2.backward()
                
                #clip_grad_value_(P1.parameters(), 1e-4)
                optimizerP.step()
                optimizerP.zero_grad()

                I *= gamma
                (stateV1,stateV2,stateP1,stateP2) = torch.split(state_new.repeat(4,1,1),1,dim=0)
                stateV1.detach_().requires_grad_()
                stateV2.detach_().requires_grad_()
                stateP1.detach_().requires_grad_()
                stateP2.detach_().requires_grad_()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

                
                
                if not steps%1:
                    out = ""
                    p1_reward = reward[0]
                    action_str1 = []
                    for a in [actions1[0], actions1[1], actions1[3]]:
                        a = torch.exp(a)
                        for i in range(3):
                            action_str1.append(a[0,i].item())
                    action_str1.append(torch.exp(actions1[2]).item())
                    out += (f"P1 left: {action_str1[0]:.6f}\n"
                        f"P1 right: {action_str1[1]:.6f}\n"
                        f"P1 x noop: {action_str1[2]:.6f}\n"
                        f"P1 up: {action_str1[3]:.6f}\n"
                        f"P1 down: {action_str1[4]:.6f}\n"
                        f"P1 y noop: {action_str1[5]:.6f}\n"
                        f"P1 jump: {action_str1[-1]:.6f}\n"
                        f"P1 hit: {action_str1[6]:.6f}\n"
                        f"P1 throw: {action_str1[7]:.6f}\n"
                        f"P1 atk noop: {action_str1[8]:.6f}\n"
                        f"delta1: {delta1:.6f}\n"
                        f"v1_old: {v1_old.item()=:.6f}\n"
                        f"v1_new: {v1_new.item()=:.6f}\n"
                        f"{p1_reward=:.6f}\n\n"
                    )

                    p2_reward = reward[1]
                    action_str2 = []
                    for a in [actions2[0], actions2[1], actions2[3]]:
                        a = torch.exp(a)
                        for i in range(3):
                            action_str2.append(a[0,i].item())
                    action_str2.append(torch.exp(actions2[2]).item())
                    out += (f"P2 left: {action_str2[0]:.6f}\n"
                        f"P2 right: {action_str2[1]:.6f}\n"
                        f"P2 x noop: {action_str2[2]:.6f}\n"
                        f"P2 up: {action_str2[3]:.6f}\n"
                        f"P2 down: {action_str2[4]:.6f}\n"
                        f"P2 y noop: {action_str2[5]:.6f}\n"
                        f"P2 jump: {action_str2[-1]:.6f}\n"
                        f"P2 hit: {action_str2[6]:.6f}\n"
                        f"P2 throw: {action_str2[7]:.6f}\n"
                        f"P2 atk noop: {action_str2[8]:.6f}\n"
                        f"delta2: {delta2:.6f}\n"
                        f"v2_old: {v2_old.item()=:.6f}\n"
                        f"v2_new: {v2_new.item()=:.6f}\n"
                        f"{p2_reward=:.6f}\n\n"

                        f"{steps=}\n"
                    )
                    display_text(out)
                    #for a in actions1:
                        #print(a)
                    #for a in actions2:
                        #print(a)
                    """print(
                        f"{delta1.item()=:.6f}\n"
                        f"{delta2.item()=:.6f}\n"
                        f"{v1_old.item()=:.6f}\n"
                        f"{v2_old.item()=:.6f}\n"
                        f"{v1_new.item()=:.6f}\n"
                        f"{v2_new.item()=:.6f}\n"
                        f"{steps=}"
                    )
                    print('delta1', delta1.item(), '\n',
                        'delta2:',delta2.item(), '\n',
                        'v1_old:',v1_old.item(), '\n',
                        'v2_old:',v2_old.item(), '\n',
                        'v1_new:',v1_new.item(), '\n',
                        'v2_new:',v2_new.item(), '\n',
                        'steps:',steps)
                    if reward[0] > reward[1]:
                        print('G')
                    elif reward[0] == reward[1]:
                        print('E')
                    else:
                        print('R')
                    p1_reward_sum += I*reward[0]
                    p2_reward_sum += I*reward[1]
                    print('p1_reward_sum:', reward)
                    #print('undiscounted_reward:', reward[1])
                    #print()"""
        stop = datetime.now()
        print('avg time/step:',(stop-start)/steps)
        episode += 1
        print(episode, steps)
        episode_len.append(steps)
        print('Finished')
        gym.reset()
        if not episode%5:
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
