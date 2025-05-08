import sys
# print(sys.path)
import time
from datetime import date
import csv
import json
import asyncio
import re

## module for server
# from kani import Kani
# from kani.engines.huggingface import HuggingEngine

import subprocess
import requests

import os
import json
import glob
import random
import argparse
from os.path import join as pjoin

from textworld_express import TextWorldExpressEnv
import textworld
import textworld.gym

from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType

from openai import OpenAI

env = TextWorldExpressEnv(envStepLimit=100)

NUM_LOCATIONS_lst = [3,5,7,9,11]
seed_lst = range(1,11)

NUM_LOCATIONS = NUM_LOCATIONS_lst[0]
env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
obs, infos = env.reset(seed=seed_lst[0], gameFold="train", generateGoldPath=True)
valid_actions = sorted(infos['validActions'])
valid_actions.remove('look around')
valid_actions.remove('inventory')

print(f"Observations: {obs} \n") 
print(f"Gold path: {env.getGoldActionSequence()} \n")
print(f"Valid Actions: {valid_actions} \n")
print(f"taskDescription: {infos['taskDescription']} \n")

# actions = ['open door to south', 'move south', 'open door to west', 'move west', 'move east', 'move north', 'open door to west', 'move west', 'move east', 'move south', 'move south', 'move north', 'move east', 'open door to north', 'move north', 'take coin']
actions = ['open door to south']

for action in actions:
    print('>', action)
    obs, reward, done, infos = env.step(action)
    print(obs)

print(infos['done'])