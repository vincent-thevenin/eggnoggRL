from mss import mss
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from xdo import Xdo
from time import sleep
import torch
from torch.distributions.categorical import Categorical
import pyautogui
import pyeggnogg as EggNogg

class EggnoggGym():
    """
    Class for the environement
    
    Args:
        None
        
    Attributes:
        monitor (dict): the coodinates of the screen :top, left, width, height
        sct (func): <function mss.factory.mss(**kwargs)>
    """    
    def __init__(self, need_pretrained, device, lib_path, executable_path, speed=60, seq_len=8):
        self.keys_tensor = torch.tensor([2**i for i in range(6)])
        self.action_tensor = torch.tensor([i for i in range(13)])
        self.device = device
        self.seq_len = seq_len
        self.gym_keys = ['z','q','s','d','c','v','o','k','l','m','n',',']

        #launch game
        EggNogg.init(lib_path, executable_path)
        sleep(5)
        EggNogg.setSpeed(speed)
        for key in self.gym_keys:
            pyautogui.keyUp(key)

        #init noop prev_action and room
        self.prev_action = [[2,2], #x_action
                            [2,2], #y_action
                            [False, False], #jump_action
                            [False, False]] #stab_action
        self.current_room = 0.5

        #grab first 8 frames
        self.states = self.get_single_state()[0]
        for _ in range(self.seq_len-1):
            self.states = torch.cat((self.states, # pylint: disable=no-member
                                        self.get_single_state()[0]),
                                    dim=1)

    def act(self, action_tensors1, action_tensors2):
        #Transforms action_tensor to string for xdo
        #coord: 0 -> left, right, noop (right,left,noop for player2)
        #       1 -> up, down, noop
        #       2 -> jump press
        #       3 -> stab press
        x_action = [Categorical(action_tensors1[0]).sample(),
                    Categorical(action_tensors2[0]).sample()]
        y_action = [Categorical(action_tensors1[1]).sample(),
                    Categorical(action_tensors2[1]).sample()]
        
        jump_action = [action_tensors1[2] > torch.rand((1,1), device=self.device), # pylint: disable=no-member
                        action_tensors2[2] > torch.rand((1,1), device=self.device)]# pylint: disable=no-member
        stab_action = [action_tensors1[3] > torch.rand((1,1), device=self.device), # pylint: disable=no-member
                        action_tensors2[3] > torch.rand((1,1), device=self.device)]# pylint: disable=no-member
        string_press = []
        string_lift = []

        #x action
        if x_action[0] == 0:
            string_press.append('q')
        elif x_action[0] == 1:
            string_press.append('d')
        elif x_action[0] == 2 or x_action[0] != self.prev_action[0][0]:
            string_lift.extend(['q','d'])

        if x_action[1] == 0:
            string_press.append('k')
        elif x_action[1] == 1:
            string_press.append('m')
        elif x_action[1] == 2 or x_action[1] != self.prev_action[0][1]:
            string_lift.extend(['k','m'])

        #y action
        if y_action[0] == 0:
            string_press.append('z')
        elif y_action[0] == 1:
            string_press.append('s')
        elif y_action[0] == 2 or y_action[0] != self.prev_action[1][0]:
            string_lift.extend(['z','s'])

        if y_action[1] == 0:
            string_press.append('o')
        elif y_action[1] == 1:
            string_press.append('l')
        elif y_action[1] == 2 or y_action[1] != self.prev_action[1][1]:
            string_lift.extend(['o','l'])
        
        #jump action
        if jump_action[0]:
            string_press.append('c')
        else:
            string_lift.append('c')

        if jump_action[1]:
            string_press.append('n')
        else:
            string_lift.append('n')
        
        #stab action
        if stab_action[0]:
            string_press.append('v')
        else:
            string_lift.append('v')
        
        if stab_action[1]:
            string_press.append(',')
        else:
            string_lift.append(',')
        
        #update previous actions
        self.prev_action = [x_action, y_action, jump_action, stab_action]

        #send inputs to eggnogg
        for lift in string_lift:
            pyautogui.keyUp(lift, _pause=False)
        for press in string_press:
            pyautogui.keyDown(press, _pause=False)



    def get_single_state(self):
        state_dict = EggNogg.getGameState()
        p1_life = (state_dict['player1']['life']-50)/50 #[0,100]
        p2_life = (state_dict['player2']['life']-50)/50

        p1_x = (state_dict['player1']['pos_x']-2904)/2904 #[0, 5808]
        p2_x = (state_dict['player2']['pos_x']-2904)/2904
            
        p1_y = (state_dict['player1']['pos_y']-89)/89 #[0,178]
        p2_y = (state_dict['player2']['pos_y']-89)/89

        p1_has_sword = (state_dict['player1']['hasSword']-0.5)/0.5 #bool
        p2_has_sword = (state_dict['player2']['hasSword']-0.5)/0.5

        p1_sword_x = (state_dict['player1']['sword_pos_x']-2904)/2904 #[0, 5808]
        p2_sword_x = (state_dict['player2']['sword_pos_x']-2904)/2904
        
        p1_sword_y = (state_dict['player1']['sword_pos_y']-89)/89 #[0,178]
        p2_sword_y = (state_dict['player2']['sword_pos_y']-89)/89

        p1_direction = state_dict['player1']['direction']/1 #[-1,1]
        p2_direction = state_dict['player2']['direction']/1

        p1_bounce_ctr = (state_dict['player1']['bounce_ctr']-2)/2 #[0,4]
        p2_bounce_ctr = (state_dict['player2']['bounce_ctr']-2)/2

        p1_situation = (state_dict['player1']['situation']-4)/4 #{0,1,8}
        p2_situation = (state_dict['player2']['situation']-4)/4

        p1_action = (self.action_tensor == state_dict['player1']['action']).float().reshape(1,1,13) #1,1,13
        p2_action = (self.action_tensor == state_dict['player2']['action']).float().reshape(1,1,13)

        p1_keys_pressed = ((self.keys_tensor & state_dict['player1']['keys_pressed']) != 0).float().reshape(1,1,6) #1,1,6
        p2_keys_pressed = ((self.keys_tensor & state_dict['player2']['keys_pressed']) != 0).float().reshape(1,1,6)

        leader = (state_dict['leader']-1)/1
        room_number = (state_dict['room_number']-5)/5
        #nb_swords = (state_dict['nb_swords'])
        swords = torch.zeros((1,1,16*3))
        for i, swordkey in enumerate(list(state_dict['swords'].keys())):
            swords[0,0,i*3+0] = (state_dict['swords'][swordkey]['pos_x']-2904)/2904
            swords[0,0,i*3+1] = (state_dict['swords'][swordkey]['pos_y']-89)/89
            swords[0,0,i*3+2] = 1.0
        
        state = torch.tensor([
            p1_life,
            p2_life,
            p1_x,
            p2_x,
            p1_y,
            p2_y,
            p1_has_sword,
            p2_has_sword,
            p1_sword_x,
            p2_sword_x,
            p1_sword_y,
            p2_sword_y,
            p1_direction,
            p2_direction,
            p1_bounce_ctr,
            p2_bounce_ctr,
            p1_situation,
            p2_situation,
            leader,
            room_number
        ])
        state = state.reshape(1,1,state.shape[0])
        
        state = torch.cat(
            (
                state,
                p1_action,
                p2_action,
                p1_keys_pressed,
                p2_keys_pressed,
                swords
            ),
            dim=2
        )
        #1,1,106

        #inversed
        #TODO keys_pressed how to reverse?
        #state2 = state[:,[1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,18,19]]
        #state2[:,[2,3,8,9,12,13,18,19]] *= -1

        #calculate gradual reward
        if room_number == self.current_room:
            r1 = r2 = 0
        elif room_number > self.current_room:
            r1 = 1
            r2 = -1
        else:
            r1 = -1
            r2 = 1
        self.current_room = room_number
        """#calculate one reward
        if room_number == 1.0:
            r1 = 1
            r2 = -1
        elif room_number == -1.0:
            r1 = -1
            r2 = 1
        else:
            r1 = 0
            r2 = 0"""

        #check terminal
        is_terminal = (room_number == 1.0) or (room_number == -1.0)
            

        return state, (r1, r2), is_terminal


    def reset(self):
        for key in self.gym_keys:
            pyautogui.keyUp(key)

        pyautogui.keyDown('f5')
        pyautogui.keyUp('f5')

        #init noop prev_action and room
        self.prev_action = [[2,2], #x_action
                            [2,2], #y_action
                            [False, False], #jump_action
                            [False, False]] #stab_action
        self.current_room = 0.5

        #grab first seq_len frames
        self.states = self.get_single_state()[0]
        for _ in range(self.seq_len-1):
            self.states = torch.cat((self.states, # pylint: disable=no-member
                                        self.get_single_state()[0]),
                                    dim=1)

    def step(self, actions_tensor1, actions_tensor2):
        with torch.autograd.no_grad():
            #remove oldest state
            self.states = self.states.split([1,self.seq_len-1], dim=1)[1]
            #b,7,x

            #act
            self.act(actions_tensor1, actions_tensor2)

            #get state
            state, reward, is_terminal = self.get_single_state()

            self.states = torch.cat((self.states, state), dim=1)# pylint: disable=no-member
            #b,8,x
        return self.states, reward, is_terminal
