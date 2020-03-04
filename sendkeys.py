from xdo import Xdo
import matplotlib.pyplot as plt
import pyautogui


delay = int(130e3)
xdo = Xdo()
win_id = max(xdo.search_windows(winname=b'eggnoggplus'))

xdo.activate_window(win_id)

xdo.send_keysequence_window_down(win_id, b'v',0)
xdo.send_keysequence_window_up(win_id, b'v',0)
"""
plt.pause(2)
xdo.send_keysequence_window_down(win_id, b'a+d+Left+Right+w+s+Up+Down+v+n+comma',0)
print(1)
xdo.send_keysequence_window_up(win_id, b'a+d+Left+Right+w+s+Up+Down+v+n+comma',delay)
print(2)
#xdo.send_keysequence_window_up(win_id, b'a+d+Left+Right+w+s+Up+Down+v+n+comma',0)
#plt.pause(2)
"""
pyautogui.keyDown(',')