#!/usr/bin/python3.8

import os, time
from datetime import datetime
os.sys.path.append("../libeggnogg/bin/pyeggnogg/lib/python3.8/site-packages/")
import pyeggnogg as EggNogg

lib_path = "../libeggnogg/bin/libeggnogg.so"
executable_path = "eggnoggplus-linux/eggnoggplus"

EggNogg.init(lib_path, executable_path)
print("Current speed %d" % EggNogg.getSpeed())
time.sleep(1)
EggNogg.setSpeed(60)
print("Current speed %d" % EggNogg.getSpeed())

while True:
	#time.sleep(1)
	#print(EggNogg.getGameState())
	e = EggNogg.getGameState()
	print(e['player1'].keys())
	break
