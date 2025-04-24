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

# choose a problem to solve
problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)

problems = [p for p in problems if "movable_recep" not in p]
if len(problems) == 0:
    raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
# problem = os.path.dirname(random.choice(problems)) # random select one problem

# for i in range(100, 200):
#     problem = os.path.dirname(problems[i]) # select a specific problem to test
#     problem = problem.replace('/Users/krystalgong/.cache/alfworld/json_2.1.1/valid_seen/', '')
#     print(f"{i}: {problem}")

problem = os.path.dirname(problems[3]) # select a specific problem to test
domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")

GAME_LOGIC = {
        "pddl_domain": open(domain).read(),
        "grammar": open(grammar).read(),
    }

# load state and trajectory files
pddl_file = os.path.join(problem, 'initial_state.pddl')
json_file = os.path.join(problem, 'traj_data.json')
with open(json_file, 'r') as f:
    traj_data = json.load(f)
# print(traj_data) # check gold actions
GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)
gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
json.dump(gamedata, open(gamefile, "w"))

# expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

request_infos = textworld.EnvInfos(
    won=True,
    admissible_commands=True,
    score=True,
    max_score=True,
    intermediate_reward=True,
    extras=["expert_plan"]
)
# reset environment starts here!
env_id = textworld.gym.register_game(
    gamefile,
    request_infos,
    max_episode_steps=1000000,
    wrappers=[AlfredDemangler(), expert]
)
env = textworld.gym.make(env_id)

obs, infos = env.reset()

init_obs = obs.split('\n')[2]
goal = obs.split('\n')[-1]
valid_actions = infos["admissible_commands"]
valid_actions.remove('look')
valid_actions.remove('inventory')
valid_actions.remove('help')
# print(infos)
print(goal)
print(init_obs)
# print(infos)
print(obs)
print(valid_actions)
# brief_obs = "Action: look around\n" + init_obs+'\n'

# cabinets = [f'cabinet {i}' for i in range(1,22)]

# == Problem 0: put a clean ladle in countertop. ==

# == Problem 1: put a cloth in bathtubbasin. == Done
# 'open shelf 1', 'examine shelf 1', 'open shelf 1', 'take cloth 1 from cabinet 4', 'go to bathtubbasin 1', 'examine bathtubbasin 1', 'move cloth 1 to bathtubbasin 1']
# actions = ['go to bathtubbasin 1', 'go to cabinet 1', 'open cabinet 1', 'go to cabinet 2', 'open cabinet 2', 'go to cabinet 3', 'open cabinet 3', 'go to cabinet 4'] 
# == Problem 2: find two spraybottle and put them in toilet. ==

# == Problem 3: put a hot slice of bread in fridge. ==
actions = ['go to countertop 1', 'take knife 1 from countertop 1', 'slice bread 1 with knife 1', 'move knife 1 to countertop 1', 'take bread 1 from countertop 1', \
           'go to microwave 1', 'open microwave 1', 'heat bread 1 with microwave 1', '(move bread 1 to microwave 1)', '(take bread 1 from microwave 1)', 'go to fridge 1','open fridge 1', 'move bread 1 to fridge 1']

# == Problem 4: put a hot potato in fridge. ==

# == Problem 5: examine the alarmclock with the desklamp. ==
# actions = ['go to desk 1', 'take alarmclock 2 from desk 1', 'use desklamp 1']

# == Problem 6: clean some fork and put it in countertop. == Done
# actions = ['go to diningtable 1', 'take fork 1 from diningtable 1', 'go to sinkbasin 1', 'clean fork 1 with sinkbasin 1', 'go to countertop 1', 'move fork 1 to countertop 1']

# == Problem 7: look at laptop under the desklamp. ==
# actions = ['go to desk 1', 'use desklamp 1', 'take laptop 1 from desk 1']

# == Problem 8: put a remotecontrol in armchair. ==

# == Problem 9: cool some winebottle and put it in diningtable. == Done
# actions = ['go to diningtable 1', 'take winebottle 1 from diningtable 1', 'go to fridge 1', '(open fridge 1)', 'cool winebottle 1 with fridge 1', 'go to diningtable 1', 'move winebottle 1 to diningtable 1']
# actions = ['go to diningtable 1', 'take winebottle 1 from diningtable 1', 'go to fridge 1', 'move winebottle 1 to fridge 1 (Nothing happens.)', 'cool winebottle 1 with fridge 1','take winebottle 1 from fridge 1 (Nothing happens.)', 'go to diningtable 1', 'move winebottle 1 to diningtable 1']


problem_type_dic = {0: 'clean', 1: 'basic', 2: 'basic', 3:'slice & heat', 4: 'heat',\
     5:'use', 6:'clean', 7: 'use', 8: 'basic', 9:'cool'}

# actions = ['go to']

for action in actions:
    print('>', action)
    obs, reward, done, infos = env.step(action)
    print(obs)
    print(infos['admissible_commands'])

print(infos['won'])