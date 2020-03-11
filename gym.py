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
        pyautogui.keyDown('n')
        pyautogui.keyUp('n')

        #init noop prev_action, room, map, throwing
        self.prev_action = [[2,2], #x_action
                            [2,2], #y_action
                            [False, False], #jump_action
                            [2,2]] #stab_action
        self.current_room = 0.5
        self.map = self.getMap()
        self.is_throwing1 = False
        self.is_throwing2 = False

        #grab first seq_frame frames
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
        x_action = [Categorical(torch.exp(action_tensors1[0])).sample(),
                    Categorical(torch.exp(action_tensors2[0])).sample()]

        y_action = [Categorical(torch.exp(action_tensors1[1])).sample(),
                    Categorical(torch.exp(action_tensors2[1])).sample()]

        jump_action = [torch.exp(action_tensors1[2]) > torch.rand((1,1), device=self.device), # pylint: disable=no-member
                        torch.exp(action_tensors2[2]) > torch.rand((1,1), device=self.device)]# pylint: disable=no-member
        
        stab_action = [Categorical(torch.exp(action_tensors1[3])).sample(), # pylint: disable=no-member
                       Categorical(torch.exp(action_tensors2[3])).sample()]# pylint: disable=no-member
        
        string_press = []
        string_lift = ['c','n']

        #x action
        if x_action[0] == 0:
            string_press.append('q')
        elif x_action[0] == 1:
            string_press.append('d')
        if x_action[0] == 2 or x_action[0] != self.prev_action[0][0]:
            string_lift.extend(['q','d'])

        if x_action[1] == 0:
            string_press.append('k')
        elif x_action[1] == 1:
            string_press.append('m')
        if x_action[1] == 2 or x_action[1] != self.prev_action[0][1]:
            string_lift.extend(['k','m'])

        #y action
        if y_action[0] == 0:
            string_press.append('z')
        elif y_action[0] == 1:
            string_press.append('s')
        if y_action[0] == 2 or y_action[0] != self.prev_action[1][0]:
            string_lift.extend(['z','s'])

        if y_action[1] == 0:
            string_press.append('o')
        elif y_action[1] == 1:
            string_press.append('l')
        if y_action[1] == 2 or y_action[1] != self.prev_action[1][1]:
            string_lift.extend(['o','l'])
        
        #jump action
        if jump_action[0]:
            string_press.append('c')

        if jump_action[1]:
            string_press.append('n')
        
        #stab action
        self.is_throwing1 = self.states[0,-1,6]==1 and (stab_action[0]==1 or self.is_throwing1)
        if self.is_throwing1:
            string_press.append('v')
        elif stab_action[0]==0 or stab_action[0]==1:
            string_lift.append('v')
            string_press.append('v')
        if stab_action[0]==2:
            string_lift.append('v')

        self.is_throwing2 = self.states[0,-1,7]==1 and (stab_action[1]==1 or self.is_throwing2)
        if self.is_throwing2:
            string_press.append(',')
        elif stab_action[1]==0 or stab_action[1]==1:
            string_lift.append(',')
            string_press.append(',')
        if stab_action[1]==2:
            string_lift.append(',')
        
        #update previous actions
        self.prev_action = [x_action, y_action, jump_action, stab_action]

        #send inputs to eggnogg
        for lift in string_lift:
            pyautogui.keyUp(lift, _pause=False)
        for press in string_press:
            pyautogui.keyDown(press, _pause=False)

    def getMap(self):
        """{"v":"down_pikes",
        "w":"water",
        "@":"wall",
        "*":"sword",
        "X":"up_pikes",
        "+":"mine",
        "^":"win_surface",
        "E":"lava",
        "~":"back_fall",
        "O":"back_sun1",
        " ":"back_air",
        "F":"back_column1",
        "H":"back_column2",
        "I":"back_column3",
        "_":"back_chandelier_support",
        "C":"back_chandelier1",
        "G":"back_sun2",
        "#":"back_crack",
        "S":"back_eyes",
        "P":"back_hiero",
        "A":"back_mill",
        "=":"back_grilling1",
        "|":"back_column4",
        "f":"back_column5",
        "L":"back_perso1",
        "N":"back_perso2",
        "Y":"back_perso3",
        ":":"back_column6",
        "-":"back_line",
        "Q":"back_skull1",
        "t":"back_tentacle1",
        "T":"back_tentacle2",
        "s":"back_tentacle3",
        "(":"back_left_sewers",
        ")":"back_right_sewers",
        "c":"back_chandelier2",
        "Z":"back_bluesquare",
        "x":"back_grilling2",
        "q":"back_skull2",
        "u":"back_sewers2",
        "K":"moving_pikes",
        ".":"back_black",
        "Z":"back_blue",
        "i":"back_people",
        "`":"back_grilling3"}"""
        map_str = EggNogg.getRoomDef()
        map = torch.zeros((1,8,12,33), device=self.device)
        for i,char in enumerate(map_str):
            if char == 'v':
                map[0,0,i//33,i%33] = 1
            if char == 'w':
                map[0,1,i//33,i%33] = 1
            if char == '@':
                map[0,2,i//33,i%33] = 1                
            if char == '*':
                map[0,3,i//33,i%33] = 1
            if char == 'X':
                map[0,4,i//33,i%33] = 1
            if char == '+':
                map[0,5,i//33,i%33] = 1
            if char == '^':
                map[0,6,i//33,i%33] = 1
            if char == 'E':
                map[0,7,i//33,i%33] = 1
        return map


    def get_single_state(self):
        state_dict = EggNogg.getGameState()

        p1_life = (state_dict['player1']['life']-50)/50 #[0,100]
        p2_life = (state_dict['player2']['life']-50)/50
            
        #TODO LAST POS IS SAME AS POS??
        p1_x = (state_dict['player1']['last_pos_x']-2904)/2904 #[0, 5808]
        p2_x = (state_dict['player2']['last_pos_x']-2904)/2904
            
        p1_y = (state_dict['player1']['last_pos_y']-89)/89 #[0,178]
        p2_y = (state_dict['player2']['last_pos_y']-89)/89

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

        p1_contact_point = (state_dict['player1']['contact_point']-4)/4 #{0,1,8}
        p2_contact_point = (state_dict['player2']['contact_point']-4)/4

        p1_action = (self.action_tensor == state_dict['player1']['action']).float().reshape(1,1,13) #1,1,13
        p2_action = (self.action_tensor == state_dict['player2']['action']).float().reshape(1,1,13)

        p1_keys_pressed = ((self.keys_tensor & state_dict['player1']['keys_pressed']) != 0).float().reshape(1,1,6) #1,1,6
        p2_keys_pressed = ((self.keys_tensor & state_dict['player2']['keys_pressed']) != 0).float().reshape(1,1,6)

        leader = (state_dict['leader']-1)/1 #-1 neutral, 0 player1, 1 player2
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
            p1_contact_point,
            p2_contact_point,
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
        
        #refresh map
        if room_number != self.current_room:
            self.map = self.getMap()

        r1 = r2 = 0
        #calculate reward for changing room
        if room_number > self.current_room:
            r1 += 2*(room_number+1)
            r2 += -2*(room_number+1)
        elif room_number < self.current_room:
            r1 += 2*(room_number-1)
            r2 += -2*(room_number-1)

        #calculate reward for pushing when leading
        if leader == 0: #player 1 lead
            bonus = (p1_x)/5
            r1 += bonus
            r2 -= bonus
        elif leader == 1: #player 2 lead
            bonus = (p2_x)/5
            r1 -= -bonus
            r2 += -bonus

        #calculate reward for surviving, dying and killing
        if leader == 0: #player 1 lead
            bonus = (p1_life)/10
            r1 += bonus
            r2 -= bonus
        elif leader == 1: #player 2 lead
            bonus = (p2_life)/10
            r1 -= bonus
            r2 += bonus           

            
        #calculate reward for being high
        r1 += -(p1_y)/200
        r2 += -(p2_y)/200

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

        EggNogg.resetGame()

        #init noop prev_action and room
        self.prev_action = [[2,2], #x_action
                            [2,2], #y_action
                            [False, False], #jump_action
                            [2, 2]] #stab_action
        self.current_room = 0.5
        self.is_throwing1 = False
        self.is_throwing2 = False
        self.map = self.getMap()

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
