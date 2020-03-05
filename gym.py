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
from network import Observation
from torch.distributions.categorical import Categorical
import pyautogui

class EggnoggGym():
    """
    Class for the environement
    
    Args:
        None
        
    Attributes:
        monitor (dict): the coodinates of the screen :top, left, width, height
        sct (func): <function mss.factory.mss(**kwargs)>
    """    
    def __init__(self, need_pretrained, device):
        # xwininfo -name eggnoggplus
        self.monitor = {"top": 70, "left": 64, "width": 1440, "height":960}
        self.sct = mss()
        self.resize_factor = self.monitor['width']//240 #width 240, height 160
        self.pil2tensor = transforms.ToTensor()
        self.device = device


        self.delay = int(130e3)
        self.xdo = Xdo()
        self.win_id = max(self.xdo.search_windows(winname=b'eggnoggplus'))

        #swap to window
        self.xdo.activate_window(self.win_id)
        self.xdo.send_keysequence_window_down(self.win_id, b'v')
        self.xdo.send_keysequence_window_up(self.win_id, b'v')

        #init observation network
        self.observation = Observation(need_pretrained=need_pretrained).to(device)

        #init noop prev_action
        self.prev_action = [[2,2], #x_action
                            [2,2], #y_action
                            [False, False], #jump_action
                            [False, False]] #stab_action

        #grab first 4 frames
        self.states = self.get_single_state()[0]
        for _ in range(3):
            self.states = torch.cat((self.states, self.get_single_state()[0]), dim=2) # pylint: disable=no-member



    def act(self, action_tensors):
        #Transforms action_tensor to string for xdo
        #coord: 0 -> left, right, noop (right,left,noop for player2)
        #       1 -> up, down, noop
        #       2 -> jump press
        #       3 -> stab press
        x_action = Categorical(action_tensors[0]).sample()
        y_action = Categorical(action_tensors[1]).sample()
        
        jump_action = action_tensors[2] < torch.rand((2,1), device=self.device)# pylint: disable=no-member
        stab_action = action_tensors[3] < torch.rand((2,1), device=self.device)# pylint: disable=no-member

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
            string_press.append('right') #reversed
        elif x_action[1] == 1:
            string_press.append('left') #reversed
        elif x_action[1] == 2 or x_action[1] != self.prev_action[0][1]:
            string_lift.extend(['left','right'])

        #y action
        if y_action[0] == 0:
            string_press.append('z')
        elif y_action[0] == 1:
            string_press.append('s')
        elif y_action[0] == 2 or y_action[0] != self.prev_action[1][0]:
            string_lift.extend(['z','s'])

        if y_action[1] == 0:
            string_press.append('up')
        elif y_action[1] == 1:
            string_press.append('down')
        elif y_action[1] == 2 or y_action[1] != self.prev_action[1][1]:
            string_lift.extend(['up','down'])
        
        #jump action
        if jump_action[0]:
            string_press.append('v')
        else:
            string_lift.append('v')

        if jump_action[1]:
            string_press.append('n')
        else:
            string_lift.append('n')
        
        #stab action
        if stab_action[0]:
            string_press.append('b')
        else:
            string_lift.append('b')
        
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
        with self.sct:
            sct_img = self.sct.grab(self.monitor)
    
            # Create the Image
            state = Image.frombytes("RGB",
                                  sct_img.size,
                                  sct_img.bgra,
                                  "raw",
                                  "BGRX")
            state = state.resize((state.size[0]//self.resize_factor,
                              state.size[1]//self.resize_factor))
            state = self.pil2tensor(state)

            r1 = r2 = 0
            is_terminal = False
            #p1 wins, red water, bottom right
            if state[0, state.shape[1]-1, state.shape[2]-1] == 1.0:
                is_terminal = True
                r1 = 1.0
                r2 = -1.0
            #p2 wins, green water, bottom left
            elif state[1, state.shape[1]-1, 0] == 1.0:
                is_terminal = True
                r1 = -1.0
                r2 = 1.0
            
            state = state.unsqueeze(0)
            #b,3,320,480
            state = state.unsqueeze(2)
            #b,3,1,320,480

            #flip image and swap red and green channels
            state_inversed = state.flip([-1])[:,[1,0,2],:,:,:]

            #cat state and inversed on batch dimension
            state = torch.cat((state, state_inversed), dim=0)# pylint: disable=no-member
        return state.to(self.device), (r1, r2), is_terminal


    def reset(self):
        pyautogui.write('zqsdvbn,')
        pyautogui.keyUp('up')
        pyautogui.keyUp('left')
        pyautogui.keyUp('down')
        pyautogui.keyUp('right')

        pyautogui.keyDown('f5')
        pyautogui.keyUp('f5')

    def step(self, action_tensor):
        #remove oldest state
        self.states = self.states.split([1,3], dim=2)[1]
        #2,3,3,320,480

        #act
        self.act(action_tensor)

        #get state
        state, reward, is_terminal = self.get_single_state()

        self.states = torch.cat((self.states, state), dim=2)# pylint: disable=no-member
        #2,3,4,320,480
        obs = self.observation(self.states)

        return obs, reward, is_terminal
