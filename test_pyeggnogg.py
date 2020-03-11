#!/usr/bin/python3.8

import os, time
import pyeggnogg as EggNogg
from datetime import datetime

lib_path = "libeggnogg/bin/libeggnogg.so"
executable_path = "eggnoggplus-linux/eggnoggplus"

def show_map(map_str):
	width = 33
	for line in [map_str[i:i+width] for i in range(0, len(map_str), width)]:
		print(line)

EggNogg.init(lib_path, executable_path)
print("Current speed %d" % EggNogg.getSpeed())
#EggNogg.setSpeed(60)
print("Current speed : %d" % EggNogg.getSpeed())
prev_map = -1
print("Playing on : %s" % EggNogg.getMapName())
i=0
key = 5
while True:
	gs = EggNogg.getGameState()
	#print(gs)
	#print("\n")
	"""if gs['room_number'] != prev_map:
		print("Current room definition :")
		show_map(EggNogg.getRoomDef())
		prev_map = gs['room_number']"""
	if gs['player1']['keys_pressed'] == key:
		start = datetime.now()
		key = -1
		i = 0
	if not gs['player1']['hasSword']:
		stop = datetime.now()
		break
	i += 1
print(stop -start)
print(i)