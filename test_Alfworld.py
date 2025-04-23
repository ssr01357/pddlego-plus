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

# def run_solver(domain_file, problem_file, solver):

#     req_body = {"domain" : domain_file, "problem" : problem_file}

#     # Send job request to solve endpoint
#     solve_request_url=requests.post(f"https://solver.planning.domains:5001/package/{solver}/solve", json=req_body).json()

#     # Query the result in the job
#     celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

#     while celery_result.json().get("status","")== 'PENDING':
#         # Query the result every 0.5 seconds while the job is executing
#         celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

#     result = celery_result.json()['result']
#     return result

# df = """(define (domain exploration)
#     (:requirements :strips :typing)
#     (:types
#         location
#         direction
#     )
#     (:predicates
#         (door-open ?loc1 ?loc2 ?dir)
#         (visited ?loc)
#         (at ?loc)
#     )
#     (:action open-door
#     :parameters (?loc1 - location ?loc2 - location ?dir - direction)
#     :precondition (and (at ?loc1) (not (door-open ?loc1 ?loc2 ?dir)))
#     :effect (door-open ?loc1 ?loc2 ?dir)
#     )

#     (:action move
#     :parameters (?from - location ?to - location ?dir - direction)
#     :precondition (and (at ?from) (door-open ?from ?to ?dir))
#     :effect (and (at ?to) (not (at ?from)) (visited ?to))
# )"""

# pf = """(define (problem exploration-problem)
#     (:domain exploration)
#     (:objects
#         kitchen - location
#         south-room - location
#         west-room - location
#         south - direction
#         west - direction
#     )
#     (:init
#         (at kitchen)
#         (visited kitchen)
#         (not (door-open kitchen south-room south))
#         (not (door-open kitchen west-room west))
#     )
#     (:goal
#         (at west-room)
#     )
# )"""

# print(run_solver(df, pf, "dual-bfws-ffparser"))

# =================================================================================================
# import re

# def map_actions(action):
#     actions = action.lower().replace("(", "").replace(")", "").split('\n')
#     action_lst = []
#     for act in actions:
#         if "gotolocation" in act: # '(GOTOLOCATION AGENT1 NEW_LOCATION TOWELHOLDER1)\n' => ['go to towelholder 1']
#             location = act.split(' ')[-1]
#             # Insert a space between non-digits and digits, e.g., "towelholder1" -> "towelholder 1"
#             formatted_location = re.sub(r"(\D+)(\d+)", r"\1 \2", location)
#             action_lst.append(f"go to {formatted_location}")
#         elif "openobject" in act: # '(OPENOBJECT CABINET4)\n' => ['open cabinet 4']
#             object_ = act.split(' ')[-1]
#             formatted_object = re.sub(r"(\D+)(\d+)", r"\1 \2", object_)
#             action_lst.append(f"open {formatted_object}")
#         elif "pickupobject" in act:  # '(PICKUPOBJECT CLOTH1 CABINET4)' => ['take cloth 1 from cabinet 4']
#             parts = act.split()
#             # Expecting parts: ['pickupobject', 'cloth1', 'cabinet4']
#             if len(parts) >= 3:
#                 obj = parts[1]
#                 container = parts[2]
#                 formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
#                 formatted_container = re.sub(r"(\D+)(\d+)", r"\1 \2", container)
#                 action_lst.append(f"take {formatted_obj} from {formatted_container}")
#         # elif : # '(PUTOBJECT CLOTH1 BATHTUBBASIN1)\n'  
#         elif "putobject" in act:  # e.g., '(PUTOBJECT CLOTH1 BATHTUBBASIN1)' => ['put cloth 1 in/on bathtubbasin 1']
#             parts = act.split()
#             if len(parts) >= 3:
#                 obj = parts[1]
#                 container = parts[2]
#                 formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
#                 formatted_container = re.sub(r"(\D+)(\d+)", r"\1 \2", container)
#                 action_lst.append(f"put {formatted_obj} in/on {formatted_container}")
#     if len(action_lst) == 0:
#         return None
#     return action_lst

# print(map_actions("(PUTOBJECT CLOTH1 BATHTUBBASIN1)\n"))

# =================================================================================================
### ========= Alfworld =========
# choose a problem to solve
problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)

problems = [p for p in problems if "movable_recep" not in p]
if len(problems) == 0:
    raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
# problem = os.path.dirname(random.choice(problems)) # random select one problem


problem = os.path.dirname(problems[1]) # select a specific problem to test
print(f"Playing {problem}")

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
actions = ['go to bathtubbasin 1', 'go to cabinet 1', 'open cabinet 1', 'go to cabinet 2', 'open cabinet 2', 'go to cabinet 3', 'open cabinet 3', 'go to cabinet 4'] 
# == Problem 2: find two spraybottle and put them in toilet. ==

# == Problem 3: put a hot slice of bread in fridge. ==
# actions = ['go to countertop 1', 'take knife 1 from countertop 1', 'slice bread 1 with knife 1', 'move knife 1 to countertop 1', 'take bread 1 from countertop 1', \
#            'go to microwave 1', 'open microwave 1', 'heat bread 1 with microwave 1', '(move bread 1 to microwave 1)', '(take bread 1 from microwave 1)', 'go to fridge 1','open fridge 1', 'move bread 1 to fridge 1']

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
    # print(infos['admissible_commands'])

print(infos['won'])