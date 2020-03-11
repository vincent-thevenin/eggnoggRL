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



pygame.init()


SIZE = WIDTH, HEIGHT = (570, 600)
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
max_steps = 1000
lambda_policy = 0.8
lambda_value = 0.8
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
                        lr=2**-7)
optimizerV = optim.SGD(params=list(V1.parameters())+list(V2.parameters()),
                        lr=2**-5)


#############################################################################
#one-step actor critic
##############################################################################

#init gym
gym = EggnoggGym(need_pretrained, gpu, lib_path, executable_path, speed=300, seq_len=32)
try:
    while True:
        #INITS
        start = datetime.now()
        is_terminal = False
        
        steps = p1_reward_sum = p2_reward_sum = 0
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
                actions1 = P1(stateP1, gym.map)
                actions2 = P2(stateP2, gym.map)
                state_new, reward, is_terminal = gym.step(actions1, actions2)
                state_new = state_new.to(gpu)
                if steps%max_steps == 0:
                    is_terminal = True

                v1_old = V1(stateV1, gym.map)#.detach())
                v2_old = V2(stateV2, gym.map)#.detach())
                with torch.autograd.no_grad():
                    #calculate delta_t: R_t+1 + gamma*V(S#I = max(1e-5, gamma*I)b,ds_t+1) - V(S_t)
                    if not is_terminal:
                        v1_new = V1(state_new, gym.map)
                        delta1 = reward[0] + gamma*v1_new - v1_old.detach()
                        v2_new = V2(state_new, gym.map)
                        delta2 = reward[1] + gamma*v2_new - v2_old.detach()
                    else:
                        delta1 = reward[0] - v1_old
                        delta2 = reward[1] - v2_old

                v1_old.backward()
                for i,p in enumerate(V1.parameters()):
                    z_value1[i] = gamma*lambda_value*z_value1[i] + p.grad
                    p.grad = -delta1*z_value1[i] #gradient ascent

                v2_old.backward()
                for i,p in enumerate(V2.parameters()):
                    z_value2[i] = gamma*lambda_value*z_value2[i] + p.grad
                    p.grad = -delta2*z_value2[i] #gradient ascent

                optimizerV.step()

                perf_p1 = Perf_p(I, actions1, gym.prev_action, 0)
                perf_p1.backward()
                for i,p in enumerate(P1.parameters()):
                    z_policy1[i] = gamma*lambda_policy*z_policy1[i] + p.grad
                    p.grad = -delta1*z_policy1[i]

                perf_p2 = Perf_p(I, actions2, gym.prev_action, 0)
                perf_p2.backward()
                for i,p in enumerate(P2.parameters()):
                    z_policy2[i] = gamma*lambda_policy*z_policy2[i] + p.grad
                    p.grad = -delta2*z_policy2[i]
                
                optimizerP.step()

                optimizerV.zero_grad()
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

                p1_reward_sum += I*reward[0]
                p2_reward_sum += I*reward[1]
                if not steps%10:
                    out = ""
                    for a in actions1:
                        out += (repr(a) + "\n")
                        print(a)
                    for a in actions2:
                        out += (repr(a) + "\n")
                        print(a)
                    out += (f"{delta1.item()=:.6f}\n"
                        f"{delta2.item()=:.6f}\n"
                        f"{v1_old.item()=:.6f}\n"
                        f"{v2_old.item()=:.6f}\n"
                        f"{v1_new.item()=:.6f}\n"
                        f"{v2_new.item()=:.6f}\n"
                        f"{steps=}\n"
                        f"{p1_reward_sum=:.6f}\n"
                        f"{p2_reward_sum=:.6f}\n"
                    )
                    display_text(out)
                   #print(
                        #f"{delta1.item()=:.6f}\n"
                        #f"{delta2.item()=:.6f}\n"
                        #f"{v1_old.item()=:.6f}\n"
                        #f"{v2_old.item()=:.6f}\n"
                        #f"{v1_new.item()=:.6f}\n"
                        #f"{v2_new.item()=:.6f}\n"
                        #f"{steps=}"
                    #)
                    """print('delta1', delta1.item(), '\n',
                        'delta2:',delta2.item(), '\n',
                        'v1_old:',v1_old.item(), '\n',
                        'v2_old:',v2_old.item(), '\n',
                        'v1_new:',v1_new.item(), '\n',
                        'v2_new:',v2_new.item(), '\n',
                        'steps:',steps)"""
                    """if reward[0] > reward[1]:
                        print('G')
                    elif reward[0] == reward[1]:
                        print('E')
                    else:
                        print('R')""",
                    """p1_reward_sum += I*reward[0]
                    p2_reward_sum += I*rewar[1]
                    print('p1_reward_sum:', p1_reward_sum)"""
                    #print('undiscounted_reward:', reward[1])
                    #print()
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
