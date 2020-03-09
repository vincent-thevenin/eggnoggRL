import pygame as pg
from pynput import keyboard
import sys
import time

global BASEWIDTH

BASEWIDTH = 40# width of the suares , diamater of the circles
# a temp surf for controller a
SCREENA= pg.Surface((BASEWIDTH*5,BASEWIDTH*3)) #pylint: disable=too-many-function-args
# a temp surf for controller b
SCREENB= pg.Surface((BASEWIDTH*5,BASEWIDTH*3)) #pylint: disable=too-many-function-args
LISTA = ['z','q','s','d','c','v']#the controls for a  
LISTB = ['o','k','l','m','n',',']#the controls for b
#! order of controlls must be preserved !!!
#* order is : up,left,down,right,jump,punch
COLORA = (255,0,0) #color of controller a
COLORB = (0,255,0) #color of controller b
THELIST = [
    [LISTA,SCREENA,COLORA],
    [LISTB,SCREENB,COLORB],
] #this list contains all the info relativ to all controllers
"""
    following are all the functions used to display the presses
    args:
        screen: controller screen to be updated
        basecolor: controller color
"""
def actifup(screen,basecolor):
    pg.draw.rect(screen,basecolor,pg.Rect(BASEWIDTH,0,BASEWIDTH,BASEWIDTH))
def actifdown(screen,basecolor):
    pg.draw.rect(screen,basecolor,pg.Rect(BASEWIDTH,BASEWIDTH*2,BASEWIDTH,BASEWIDTH))
def actifleft(screen,basecolor):
    pg.draw.rect(screen,basecolor,pg.Rect(0,BASEWIDTH,BASEWIDTH,BASEWIDTH))
def actifright(screen,basecolor):
    pg.draw.rect(screen,basecolor,pg.Rect(BASEWIDTH*2,BASEWIDTH,BASEWIDTH,BASEWIDTH))
def actifjump(screen,basecolor):
    pg.draw.circle(screen,basecolor,((BASEWIDTH*4)+5-BASEWIDTH//2,int(BASEWIDTH*2.5)-10),BASEWIDTH//2)
def actifpunch(screen,basecolor):
    pg.draw.circle(screen,basecolor,((BASEWIDTH*4)+BASEWIDTH//2,(BASEWIDTH//2)+10),BASEWIDTH//2)

"""
    following are all the functions used to display the releases
    args:
        screen: controller screen to be updated
        basecolor: controller color
        mod: a ratio to quantify the diming of the color
"""

def unactifup(screen,basecolor):
    mod = 0.6
    pg.draw.rect(screen,colo(basecolor,mod),pg.Rect(BASEWIDTH,0,BASEWIDTH,BASEWIDTH))
def unactifdown(screen,basecolor):
    mod = 0.6
    pg.draw.rect(screen,colo(basecolor,mod),pg.Rect(BASEWIDTH,BASEWIDTH*2,BASEWIDTH,BASEWIDTH))
def unactifleft(screen,basecolor):
    mod = 0.6
    pg.draw.rect(screen,colo(basecolor,mod),pg.Rect(0,BASEWIDTH,BASEWIDTH,BASEWIDTH))
def unactifright(screen,basecolor):
    mod = 0.6
    pg.draw.rect(screen,colo(basecolor,mod),pg.Rect(BASEWIDTH*2,BASEWIDTH,BASEWIDTH,BASEWIDTH))
def unactifjump(screen,basecolor):
    mod = 0.6
    pg.draw.circle(screen,colo(basecolor,mod),((BASEWIDTH*4)+5-BASEWIDTH//2,int(BASEWIDTH*2.5)-10),BASEWIDTH//2)
def unactifpunch(screen,basecolor):
    mod = 0.6
    pg.draw.circle(screen,colo(basecolor,mod),((BASEWIDTH*4)+BASEWIDTH//2,(BASEWIDTH//2)+10),BASEWIDTH//2)


def colo(basecolor, mod):
    """
    function that applies a ratio to a color

    returns: the dimed color

    args:
        basecolor: the bright color to be dimed
        mod: the ratio at witch the color must be dimed by, float between 0 and 1
    """
    (a,b,c) = basecolor
    return (int(a*mod),int(b*mod),int(c*mod))

def listpress(key , list , tscreenn,basecolor):
    """
    links the the keys and controller info to the draw function for presses
    args:
        key: the key that is pressed given by pynput.keyboard
        list: the list of controlls
        tscreen: the controllers temp surface
        basecolor:  the controllers bright color
    """
    if key == keyboard.KeyCode.from_char( list[0]) : #up
        actifup(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[1]) : #left
        actifleft(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[2]) : #down 
        actifdown(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[3]) : #right
        actifright(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[4]) : #jump button
        actifjump(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[5]) : #punch button
        actifpunch(tscreenn,basecolor)

def listrelease(key , list , tscreenn,basecolor):
    """
    links the the keys and controller info to the draw function for releases
    args:
        key: the key that is pressed given by pynput.keyboard
        list: the list of controlls
        tscreen: the controllers temp surface
        basecolor:  the controllers bright color
    """
    if key == keyboard.KeyCode.from_char( list[0]) : #up
        unactifup(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[1]) : #left
        unactifleft(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[2]) : #down
        unactifdown(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[3]) : #right
        unactifright(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[4]) : #jump button
        unactifjump(tscreenn,basecolor)
    elif key == keyboard.KeyCode.from_char( list[5]) : #punch button
        unactifpunch(tscreenn,basecolor)

def onpress(key):
    """
        callback for the listener object, links the keys to the different controllers before updating the display
        args:
        key: is passed by the listener when pressed
    """
    global THELIST
    for llist in THELIST: #for each controller
        listpress(key,llist[0],llist[1],llist[2]) # draw according to presses
    screen.blit(SCREENA,(0,0))#draw the left controller's temp surf to the screen
    screen.blit(SCREENB,(BASEWIDTH*5+10,0)) # draw the right controllers temp surf to the screen at the right of the left temp surf
    #pg.display.flip() # update the screen to be shown
    

def onrelease(key):
    """
        callback for the listener object, links the keys to the different controllers before updating the display
        args:
        key: is passed by the listener when released
    """
    global THELIST
    for llist in THELIST:#for each controller
        listrelease(key,llist[0],llist[1],llist[2])#draw according to releases
    screen.blit(SCREENA,(0,0))#draw controller to screen
    screen.blit(SCREENB,(BASEWIDTH*5+10,0)) #draw conroller to d=screen
    #pg.display.flip()#show the screen

def default():
    """
        initializes the keyboard listener by attaching callbacks 
    """
    listener = keyboard.Listener(
    on_press=onpress,
    on_release=onrelease)
    listener.start()#start listening

def runloop():
    """
        filler loop to keep the prog from finishing, includes clean stop
    """
    done  = False
    while not done:
        for event in pg.event.get():
                    if event.type == pg.QUIT:# pylint: disable=no-member
                            done = True
                            sys.exit()
        #time.sleep(1)# sleep one second to keep the cpu consumtion low, means that it takes a second to close
        pg.display.flip()

if __name__ == "__main__":
    """
        executed as script not module
    """
    #initialize pygame
    pg.init()#pylint:disable=no-member
    screen = pg.display.set_mode((BASEWIDTH*10+10,BASEWIDTH*3))#set the display screen
    pg.display.set_caption("the incredible controller display")#set a caption for the window
    default()#default initialiser of the listener
    runloop()#filler loop