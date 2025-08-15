import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

import time
from datetime import date
import csv
import json
import asyncio
import re

from dotenv import load_dotenv
load_dotenv()

# module for server
from kani import Kani
from kani.engines.huggingface import HuggingEngine


import requests

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

import torch
import gc



_hf_engine_cache: dict[str, HuggingEngine] = {}
_kani_cache: dict[str, Kani] = {}

PROBLEMS = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
PROBLEMS = [p for p in PROBLEMS if "movable_recep" not in p]
if len(PROBLEMS) == 0:
    raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")

GAME_DICT = {
    "basic&use": [1,8,23,21,30,5,7,27,20,22],
    "cool": [9,13,18,25,42,44,46,50,58,68],
    "heat": [4,60,69,71,93,104,119,151,153,154],
    "clean": [0,6,10,37,38,52,61,79,80,83],
    "slice+": [29,39,56,3,17,34,49,73,19,64]
}

def clear_cuda_memory(model_name):
    """Clears a specific model from cache and frees GPU memory."""
    global _kani_cache, _hf_engine_cache

    if model_name in _kani_cache:
        del _kani_cache[model_name]
        # print(f"Cleared {model_name} from kani_cache.")

    if model_name in _hf_engine_cache:
        del _hf_engine_cache[model_name]
        # print(f"Cleared {model_name} from hf_engine_cache.")

    # Force Python's garbage collector to run
    gc.collect()

    # Tell PyTorch to release unused cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # print("Cleared CUDA cache.")



def run_solver(domain_file, problem_file, solver, max_retries=3):
    # domain_file = open(f'domain.pddl').read()
    # problem_file = open(f'problem.pddl').read()

    req_body = {"domain" : domain_file, "problem" : problem_file}

    retries = 0

    while retries < max_retries:
        try:
            # Send job request to solve endpoint
            solve_request_url = requests.post(
                f"https://solver.planning.domains:5001/package/{solver}/solve", 
                json=req_body
            ).json()

            # Query the result in the job
            celery_result = requests.post(
                'https://solver.planning.domains:5001' + solve_request_url['result']
            )

            while celery_result.json().get("status", "") == 'PENDING':
                time.sleep(0.5)
                celery_result = requests.post(
                    'https://solver.planning.domains:5001' + solve_request_url['result']
                )

            return celery_result.json()['result']

        except Exception as e:
            last_error = e  # Save the last exception
            print(
                f"Error encountered: {e}.\n"
                f"Attempt {retries + 1}/{max_retries}. Retrying in 5 seconds...\n"
                f"--- Failing Domain File ---\n{domain_file}\n"
                f"--- Failing Problem File ---\n{problem_file}\n"
                f"--------------------------"
            )
            retries += 1
            time.sleep(5)

    # If all retries fail, raise a detailed error
    raise RuntimeError(
        f"Max retries exceeded. Failed to get result from solver.\n"
        f"Last error: {last_error}\n"
        f"--- Failing Domain File ---\n{domain_file}\n"
        f"--- Failing Problem File ---\n{problem_file}\n"
        f"--------------------------"
    )

def get_action_from_pddl(df, pf):
    result = run_solver(df, pf, "dual-bfws-ffparser")
    action = result['output']['plan']
    print(f"action from solver: {action}")
    err_2 = result['stderr'] + result['stdout']
    return map_actions(action), err_2


OPENAI_MODELS_LIST = ['gpt-4o','o3-mini',"gpt-4.1","o4-mini"]
def run_llm(prompt, model_name):
    if any(model_name.startswith(base) for base in OPENAI_MODELS_LIST): 
        client = OpenAI()

        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        if re.match(r'^o\d+', model_name):  
            params["reasoning_effort"] = "high"

        response = client.chat.completions.create(**params)
        response_content = response.choices[0].message.content

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                repaired_content = repair_json(response_content)
                result = json.loads(repaired_content)
            except json.JSONDecodeError:  
                raise ValueError(
                    f"Model response is not valid JSON after repair attempts:\n{response_content}"
                )

        df = result.get("df", None)
        pf = result.get("pf", None)
        return df, pf
    
    elif model_name == 'deepseek':
        deepseekAPI = os.getenv("deepseek_API")
        client = OpenAI(api_key=deepseekAPI, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are generating PDDL according to your observations."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        response_content = response.choices[0].message.content

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                repaired_content = repair_json(response_content)
                result = json.loads(repaired_content)
            except json.JSONDecodeError:  
                raise ValueError(
                    f"Model response is not valid JSON after repair attempts:\n{response_content}"
                )
        df = result.get("df", None)
        pf = result.get("pf", None)
        return df, pf
    
    else: # Open source LLMs
        async def _ask_model(model_name, user_prompt):
            if model_name not in _hf_engine_cache:
                engine = HuggingEngine(
                    model_id=model_name,
                    use_auth_token=True,
                    model_load_kwargs={"device_map": "auto", "trust_remote_code": True}
                )
                _hf_engine_cache[model_name] = engine
                _kani_cache[model_name] = Kani(engine, system_prompt="")
            ai = _kani_cache[model_name]
            return await ai.chat_round_str(user_prompt)
     
        response_content = asyncio.run(_ask_model(model_name, prompt))

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                repaired_content = repair_json(response_content)
                result = json.loads(repaired_content)
            except json.JSONDecodeError:  
                raise ValueError(
                    f"Model response is not valid JSON after repair attempts:\n{response_content}"
                )

        df = result.get("df")
        pf = result.get("pf")
        return df, pf



# Set up baseline model: get actions directly from model
def run_llm_for_actions_baseline(prompt, model_name):
    if any(model_name.startswith(base) for base in OPENAI_MODELS_LIST): 
        client = OpenAI()

        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        if re.match(r'^o\d+', model_name):  
            params["reasoning_effort"] = "high"

        response = client.chat.completions.create(**params)
        response_content = response.choices[0].message.content

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                repaired_content = repair_json(response_content)
                result = json.loads(repaired_content)
            except json.JSONDecodeError:  
                raise ValueError(
                    f"Model response is not valid JSON after repair attempts:\n{response_content}"
                )

        actions = result.get("actions", None)

        return actions
    
    elif model_name == 'deepseek':
        deepseekAPI = os.getenv("deepseek_API")
        client = OpenAI(api_key=deepseekAPI, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are generating PDDL according to your observations."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        response_content = response.choices[0].message.content

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                repaired_content = repair_json(response_content)
                result = json.loads(repaired_content)
            except json.JSONDecodeError:  
                raise ValueError(
                    f"Model response is not valid JSON after repair attempts:\n{response_content}"
                )
            
        actions = result.get("actions", None)

        return actions
        
    else: # Open source LLMs
        async def _ask_model(model_name, user_prompt):
            if model_name not in _hf_engine_cache:
                engine = HuggingEngine(
                    model_id=model_name,
                    use_auth_token=True,
                    model_load_kwargs={"device_map": "auto", "trust_remote_code": True}
                )
                _hf_engine_cache[model_name] = engine
                _kani_cache[model_name] = Kani(engine, system_prompt="")
            ai = _kani_cache[model_name]
            return await ai.chat_round_str(user_prompt)
 
        response_content = asyncio.run(_ask_model(model_name, prompt))

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                repaired_content = repair_json(response_content)
                result = json.loads(repaired_content)
            except json.JSONDecodeError:  
                raise ValueError(
                    f"Model response is not valid JSON after repair attempts:\n{response_content}"
                )

        actions = result.get("actions", None)

        return actions

def repair_json(content):
    
    # Clean up the content
    content = _remove_think_tags(content)
    content = _extract_json_from_codeblock(content)
    content = _fix_triple_quoted_strings(content)
    content = _extract_json_from_plain_text(content)
    content = _fix_unescaped_characters(content)
    content = _strip_formatting(content)

    return content

def _remove_think_tags(text):
    think_end = text.find('</think>')
    if think_end != -1:
        return text[think_end + 8:].strip()  # len('</think>') = 8
    return text

def _extract_json_from_codeblock(text):
    # Try to match ```json ... ``` first
    patterns = [
        r"(?s)```json\s*(.*?)\s*```",  # JSON code block
        r"(?s)```\s*(.*?)\s*```"        # Generic code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return text

def _extract_json_from_plain_text(text: str) -> str:
    last_brace_pos = text.rfind('{')
    if last_brace_pos != -1:
        return text[last_brace_pos:]

    return text

def _fix_triple_quoted_strings(text):
    pattern = r'"(\w+)":\s*"""(.*?)"""'
    
    def escape_for_json(match):
        key = match.group(1)
        content = match.group(2)
        
        # Escape special characters for JSON
        replacements = [
            ('\\', '\\\\'),  # Backslashes first
            ('"', '\\"'),    # Quotes
            ('\n', '\\n'),   # Newlines
            ('\r', '\\r'),   # Carriage returns
            ('\t', '\\t'),   # Tabs
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        return f'"{key}": "{content}"'
    
    return re.sub(pattern, escape_for_json, text, flags=re.DOTALL)


def _fix_unescaped_characters(text: str) -> str:
    # Set a limit to prevent infinite loops on unfixable errors

    text = text.replace('\\\n', '\\n') # llama

    max_attempts = 200
    
    for i in range(max_attempts):
        try:
            # Try to parse the text. If it works, we're done.
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            # On error, the parser tells us exactly where the problem is.
            # e.pos is the character index of the error.
            
            # Case 1: An unescaped newline or other control character in a string
            if "Invalid control character" in e.msg:
                # Replace the problematic character with its escaped version
                char_to_escape = text[e.pos]
                if char_to_escape == '\n':
                    escaped_char = '\\n'
                elif char_to_escape == '\r':
                    escaped_char = '\\r'
                elif char_to_escape == '\t':
                    escaped_char = '\\t'
                else:
                    # If it's some other control character, just remove it
                    escaped_char = ''
                
                text = text[:e.pos] + escaped_char + text[e.pos + 1:]

            # Case 2: An unescaped double quote in a string
            elif "Unterminated string" in e.msg:
                # This often means a '"' is inside a string without being escaped.
                # We'll try to find the quote just before the error and escape it.
                text = text[:e.pos - 1] + '\\' + text[e.pos - 1:]

            # If we can't fix it, break the loop and return the broken text
            else:
                break
    return text

def _strip_formatting(text):
    # Remove leading/trailing markdown json markers
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]  # len("```json") = 7
    if text.endswith("```"):
        text = text[:-3]
    
    # Remove surrounding quotes if present
    text = text.strip()
    if text.startswith("'") and text.endswith("'") and len(text) > 1:
        text = text[1:-1]
    
    return text.strip()


# VAL setup
# common_path = "/Users/krystalgong/Documents/GitHub/pddlego-df/"



# Additional functions
def summarize_obs(obs):
    # If obs only has one line, return it
    if len(obs.split('\n')) == 1:
        return obs
    # Only keep where you are and location informtion
    else:
        return obs.split('\n')[0].split(". ")[0] + ". " + obs.split('\n')[1]

def map_actions(action):
    actions = action.lower().replace("(", "").replace(")", "").split('\n')
    action_lst = []
    for act in actions:
        if "gotolocation" in act: # '(GOTOLOCATION AGENT1 NEW_LOCATION TOWELHOLDER1)\n' => ['go to towelholder 1']
            location = act.split(' ')[-1]
            # Insert a space between non-digits and digits, e.g., "towelholder1" -> "towelholder 1"
            formatted_location = re.sub(r"(\D+)(\d+)", r"\1 \2", location)
            action_lst.append(f"go to {formatted_location}")
        elif "openobject" in act: # '(OPENOBJECT CABINET4)\n' => ['open cabinet 4']
            object_ = act.split(' ')[-1]
            formatted_object = re.sub(r"(\D+)(\d+)", r"\1 \2", object_)
            action_lst.append(f"open {formatted_object}")
        elif "pickupobject" in act:  # '(PICKUPOBJECT CLOTH1 CABINET4)' => ['take cloth 1 from cabinet 4']
            parts = act.split()
            if len(parts) >= 3:
                obj = parts[1]
                container = parts[2]
                formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
                formatted_container = re.sub(r"(\D+)(\d+)", r"\1 \2", container)
                action_lst.append(f"take {formatted_obj} from {formatted_container}")
        elif "putobject" in act:  # e.g., '(PUTOBJECT CLOTH1 BATHTUBBASIN1)' => ['move cloth 1 to bathtubbasin 1']
            parts = act.split()
            if len(parts) >= 3:
                obj = parts[1]
                container = parts[2]
                formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
                formatted_container = re.sub(r"(\D+)(\d+)", r"\1 \2", container)
                action_lst.append(f"move {formatted_obj} to {formatted_container}")
        # elif "examinereceptacle" in act: # (EXAMINERECEPTACLE SHELF1) => ['examine shelf 1']
        #     parts = act.split()
        #     if len(parts) >= 2:
        #         receptacle = parts[1]
        #         formatted_receptacle = re.sub(r"(\D+)(\d+)", r"\1 \2", receptacle)
        #         action_lst.append(f"examine {formatted_receptacle}")
        elif "useobject" in act: # (USEOBJECT DESKLAMP1) => ['use desklamp 1']
            parts = act.split()
            if len(parts) >= 2:
                obj = parts[1]
                formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
                action_lst.append(f"use {formatted_obj}")
        elif "heatobject" in act: # (HEATOBJECT BREAD1 MICROWAVE1) => ['heat bread 1 with microwave 1']
            parts = act.split()
            if len(parts) >= 3:
                obj = parts[1]
                receptacle = parts[2]
                formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
                formatted_receptacle = re.sub(r"(\D+)(\d+)", r"\1 \2", receptacle)
                action_lst.append(f"heat {formatted_obj} with {formatted_receptacle}")
        elif "cleanobject" in act: # (CLEANOBJECT FORK1 SINKBASIN1) => ['clean fork 1 with sinkbasin 1']
            parts = act.split()
            if len(parts) >= 3:
                obj = parts[1]
                receptacle = parts[2]
                formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
                formatted_receptacle = re.sub(r"(\D+)(\d+)", r"\1 \2", receptacle)
                action_lst.append(f"clean {formatted_obj} with {formatted_receptacle}")
        elif "coolobject" in act: # (COOLOBJECT WINEBOTTLE1 FRIDGE1) => ['cool winebottle 1 with fridge 1']
            parts = act.split()
            if len(parts) >= 3:
                obj = parts[1]
                receptacle = parts[2]
                formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
                formatted_receptacle = re.sub(r"(\D+)(\d+)", r"\1 \2", receptacle)
                action_lst.append(f"cool {formatted_obj} with {formatted_receptacle}")
        elif "sliceobject" in act: # (SLICEOBJECT COUNTERTOP1 BREAD1 KNIFE1) => ['slice bread 1 with knife 1']
            parts = act.split()
            if len(parts) >= 4:
                # Ignore the location (parts[1], "COUNTERTOP1") and take parts[2] (the object) and parts[3] (the sharp tool).
                obj = parts[2]
                sharp_obj = parts[3]
                sharp_obj = sharp_obj.replace(")", "")
                formatted_obj = re.sub(r"(\D+)(\d+)", r"\1 \2", obj)
                formatted_sharp = re.sub(r"(\D+)(\d+)", r"\1 \2", sharp_obj)
                action_lst.append(f"slice {formatted_obj} with {formatted_sharp}")

    if len(action_lst) == 0:
        return None
    return action_lst

def detect_duplicates(action_lst, threshold):
    n = len(action_lst)
    
    for seq_len in range(1, n // 2 + 1):
        # Get the last sequence of this length
        sequence = action_lst[-seq_len:]
        
        # Count how many times this sequence appears continuously at the end
        count = 1
        for i in range(2, threshold + 1):
            if action_lst[-i * seq_len: - (i - 1) * seq_len] == sequence:
                count += 1
            else:
                break
        
        # If the sequence repeats at least 'threshold' times, return True
        if count >= threshold:
            return True

    # If no sequence repeats up to the threshold, return False
    return False


def llm_to_pddl(model_name, brief_obs, goal, goal_type='detailed', prev_df="", prev_pf="", prev_err="", prev_err_2=None, have_error=False, have_duplicate=False, edit=False, overall_memory=None, large_loop_error_message=None):
    prompt_format = f"""
        Please provide the output in strict JSON format, without any additional text or explanation, including a PDDL domain file as 'df' and a PDDL problem file as 'pf'. 
        The format should strictly be:
            {{
            "df": "...",
            "pf": "..."
            }}
    """

    prompt_edit = """
        Please provide the output in JSON format, including the edit suggestions for a domain file as 'df' and the edit suggestions for a problem file as 'pf'. 
        The output format should be: {{"df": "...", "pf": "..."}}
        You will modify the following df and pf using add, delate, and replace operations (in a JSON format). 
        You SHOULD NOT provide a domain file and a problem file directly.
        This is the structure for df edit file, remember to add bracket:
        {
        "predicates": {
            "add": ["(predicates to add)"],
            "replace": {"(old)": "(new)"},
            "delete": ["(predicates to delete)"]
            },
        "action": {
            "open-door": {
                "precondition": ["(entire full new precondition for open-door)"], # directly replace the whole precondition
                "effect": ["(entire full new effect for open-door)"] # so as effect
                },
            "move": {
                "precondition": []
                "effect": []
                }
            }
        }
        This is the structure for pf edit file:
        {
        "objects": {
            "add": [],
            "replace": {},
            "delete": []
            },
        "init": {
            "add": [],
            "replace": {},
            "delete": []
            },
        "goal": ["(entire full new goal)"]
        }
    """

    prompt_obs_action_general_goal = f"""
        You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

        The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle.
        
        Now, {goal}
        Here are your current observations: {brief_obs}

        The following actions are allowed: 
        1. go to a receptacle
            :action GotoLocation
            :parameters (?from - receptacle ?to - receptacle)
        2. open a receptacle if it is closed
            :action OpenObject
            :parameters (?r - receptacle)
        3. close a receptacle
            :action CloseObject
            :parameters (?r - receptacle)
        4. take an object from another receptacle
            :action PickupObject
            :parameters (?o - object ?r - receptacle)
        5. put object into/on/in another receptacle
            :action PutObject
            :parameters (?o - object ?r - receptacle)
        6. using an object/receptacle by turning it on/off with a switch
            :action useObject
            :parameters (?o - object)
        7. heat an object using a receptacle
            :action HeatObject
            :parameters (?o - object ?r - microwaveReceptacle)
        8. clean an object using a receptacle
            :action CleanObject
            :parameters (?o - object ?r - receptacle)
        9. cool an object using a receptacle
            :action CoolObject
            :parameters (?o - object ?r - fridgeReceptacle)
        10. slice an object using a sharp object
            :action SliceObject
            :parameters (?r - receptacle ?co - object ?sharp_o - object)

        You are in an unfamiliar environment. Your task is to complete the given objective by observing your surroundings and interacting with objects and receptacles.
        You must build and update PDDL files solely based on what you directly observe. Do not make assumptions.

        General Goal: Find the necessary object(s) and use them appropriately to accomplish the task.
        Use the allowed actions to explore, discover, and act purposefully. Plan each step based on what you know so far.
        
        Constraints:
            1. Do not assume unseen objects or relationships.
            2. Receptacle names must be preserved exactly.
            3. Do not proceed to Stage 2 before completing Stage 1.
    """ 

    prompt_obs_action_subgoal = f"""
        You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

        The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.
        
        Now, {goal}
        Here are your current observations: {brief_obs}

        The following actions are allowed: (There are only two types: object and receptacle)
        1. go to a receptacle
            :action GotoLocation
            :parameters (?from - receptacle ?to - receptacle)
        2. open a receptacle if it is closed
            :action OpenObject
            :parameters (?r - receptacle)
        3. close a receptacle
            :action CloseObject
            :parameters (?r - receptacle)
        4. take an object from another receptacle
            :action PickupObject
            :parameters (?o - object ?r - receptacle)
        5. put object into/on/in another receptacle
            :action PutObject
            :parameters (?o - object ?r - receptacle)
        6. using an object/receptacle by turning it on/off with a switch
            :action useObject
            :parameters (?o - object)
        7. heat an object using a receptacle
            :action HeatObject
            :parameters (?o - object ?r - microwaveReceptacle)
        8. clean an object using a receptacle
            :action CleanObject
            :parameters (?o - object ?r - receptacle)
        9. cool an object using a receptacle
            :action CoolObject
            :parameters (?o - object ?r - fridgeReceptacle)
        10. slice an object using a sharp object
            :action SliceObject
            :parameters (?r - receptacle ?co - object ?sharp_o - object)

        Your process involves two main stages with the following subgoals:

        Stage 1: Search for the Target Object
            Goal 1.1: Move to a new, unvisited receptacle using the GotoLocation action.
            Goal 1.2: If the receptacle is closed, use the OpenObject action to reveal its contents.

        Stage 2: Use the Object to Complete the Task
            Goal 2.1: Pick up the target object using the PickupObject action.
            Goal 2.2: Move to the appropriate location needed to fulfill the task.
            Goal 2.3: Interact with relevant objects or receptacles (e.g., heat, clean, cool, slice, or use) to accomplish the task.

        Constraints:
            1. Do not assume unseen objects or relationships.
            2. Receptacle names must be preserved exactly.
            3. Do not proceed to Stage 2 before completing Stage 1.
        
        Note: Always include :negative-preconditions in your :requirements whenever you use (not …) or delete effects, and never leave an :precondition or :effect block empty—either omit it or include at least one literal.
    """ 

    prompt_obs_action_without_detailed_goal = f"""
        You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

        The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.
        
        Now, {goal}
        Here are your current observations: {brief_obs}

        Only the following actions are allowed: (There are only two types: object and receptacle)
        1. go to a receptacle
            :action GotoLocation
            :parameters (?from - receptacle ?to - receptacle)
        2. open a receptacle if it is closed
            :action OpenObject
            :parameters (?r - receptacle)
        3. close a receptacle
            :action CloseObject
            :parameters (?r - receptacle)
        4. take an object from another receptacle
            :action PickupObject
            :parameters (?o - object ?r - receptacle)
        5. put object into/on/in another receptacle
            :action PutObject
            :parameters (?o - object ?r - receptacle)
        6. using an object/receptacle by turning it on/off with a switch
            :action useObject
            :parameters (?o - object)
        7. heat an object using a receptacle
            :action HeatObject
            :parameters (?o - object ?r - microwaveReceptacle)
        8. clean an object using a receptacle
            :action CleanObject
            :parameters (?o - object ?r - sinkbasinReceptacle)
        9. cool an object using a receptacle
            :action CoolObject
            :parameters (?o - object ?r - fridgeReceptacle)
        10. slice an object using a sharp object
            :action SliceObject
            :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)

        You must go to a receptacle first in order to use/open it or take/put objects from/on it.

        The process involves two main stages:
        Stage 1: Search for the Target Object
            In this stage, your goal is to go to and may need to open new, unvisited recepatacles until you find the object mentioned in the task. Some receptacles cannot be opened so you can directly see what objects after you go to that receptacle.

            You can only use the GotoLocation action to travel to a new location and the OpenObject action (if the receptacle is closed) to verify whether it contains the target object.

            Goal 1.1: Reach a location that has not been visited (the location should be a receptacle) using the GotoLocation action. 
            Goal 1.2: If you already go to the recepatacle and found the recepatacle is closed, use the OpenObject action to open it and inspect the contents. 


        Stage 2: Use the Object to Complete the Task
            After you have located the object (the object may have some numbers added), you should always first pick up the object from that receptacle and update your goal to focus on how the object is used to complete the task. Remember your goal is {goal}. Based on different adjectives, you may need to perform different actions for the object in different ways.

            This may involve more than simply transferring it from one place to another.
            For example: You might examine the object or a nearby receptacle to gather information. You may need to use another tool or device (like a lamp or a switch). Some tasks require you to slice, heat, cool, or clean the object using an appropriate receptacle (e.g., microwave, sink, fridge).

            If necessary, use the PickupObject action to retrieve the item, and the GotoLocation action to move to the correct place.
            Then, apply the object in a purposeful way — not just move it — but interact with the environment to fulfill the task’s actual goal.

            Hint: 
            1. If you want to heat, clean, and cool an object, after you go to that aim receptacle, do not put the object in the receptacle but do the action directly. For example, go to fridge, then cool the object with receptacle.
            2. If you want to slice an object, you should first go to the receptacle where both the sharp object and the aim object are located and ONLY pick up the sharp object then do the slice action. Don't forget to put the sharp object back to the receptacle after you finish slicing.
            3. If you want to examine or look at an object with a lamp, you should first go to the receptacle where the object is located and then pick it up and take the USE action of the lamp. You don't need to take the lamp but directly use it.
            4. If there are multiple actions needed to complete the task, you can break them down into smaller subgoals. For example, if you need to slice and then heat an object, first focus on slicing it, and then move on to heating it.

        In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.
        
        Note: 
        1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
        2. Your initial goal should always be to go to a new location instead of put something into somewhere.
        3. Do not enter stage 2 when not finishing stage 1.

        Note: Always include :negative-preconditions in your :requirements whenever you use (not …) or delete effects, and never leave an :precondition or :effect block empty—either omit it or include at least one literal.
    """ 

    prompt_obs_action_without_hint = f"""
        You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

        The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.
        
        Now, {goal}
        Here are your current observations: {brief_obs}

        Only the following actions are allowed: (There are only two types: object and receptacle)
        1. go to a receptacle
            :action GotoLocation
            :parameters (?from - receptacle ?to - receptacle)
        2. open a receptacle if it is closed
            :action OpenObject
            :parameters (?r - receptacle)
        3. close a receptacle
            :action CloseObject
            :parameters (?r - receptacle)
        4. take an object from another receptacle
            :action PickupObject
            :parameters (?o - object ?r - receptacle)
        5. put object into/on/in another receptacle
            :action PutObject
            :parameters (?o - object ?r - receptacle)
        6. using an object/receptacle by turning it on/off with a switch
            :action useObject
            :parameters (?o - object)
        7. heat an object using a receptacle
            :action HeatObject
            :parameters (?o - object ?r - microwaveReceptacle)
        8. clean an object using a receptacle
            :action CleanObject
            :parameters (?o - object ?r - sinkbasinReceptacle)
        9. cool an object using a receptacle
            :action CoolObject
            :parameters (?o - object ?r - fridgeReceptacle)
        10. slice an object using a sharp object
            :action SliceObject
            :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)

        You must go to a receptacle first in order to use/open it or take/put objects from/on it.

        The process involves two main stages:

        Stage 1: Search for the Target Object
            Goal 1.1: Move to a new, unvisited receptacle using the GotoLocation action.
                You goal should look like this:
                    (:goal 
                        (at ?recepatacle)
                    ) where recepatacle should be somewhere or some recepatacles not visited.
            Goal 1.2: If the receptacle is closed, use the OpenObject action to reveal its contents.
                Your goal should look like this:
                    (:goal 
                        (opened ?recepatacle)
                    ) where recepatacle should be the recepatacle you want to open.

        Stage 2: Use the Object to Complete the Task
            Goal 2.1: Pick up the target object using the PickupObject action.
            Goal 2.2: Move to the appropriate location needed to fulfill the task.
            Goal 2.3: Interact with relevant objects or receptacles (e.g., heat, clean, cool, slice, or use) to accomplish the task.

        Constraints:
            1. Do not assume unseen objects or relationships.
            2. Receptacle names must be preserved exactly.
            3. Do not proceed to Stage 2 before completing Stage 1.

        Note: Always include :negative-preconditions in your :requirements whenever you use (not …) or delete effects, and never leave an :precondition or :effect block empty—either omit it or include at least one literal.
    """ 

    prompt_obs_action_detailed = f"""
        You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

        The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.
        
        Now, {goal}
        Here are your current observations: {brief_obs}

        Only the following actions are allowed: (There are only two types: object and receptacle)
        1. go to a receptacle
            :action GotoLocation
            :parameters (?from - receptacle ?to - receptacle)
        2. open a receptacle if it is closed
            :action OpenObject
            :parameters (?r - receptacle)
        3. close a receptacle
            :action CloseObject
            :parameters (?r - receptacle)
        4. take an object from another receptacle
            :action PickupObject
            :parameters (?o - object ?r - receptacle)
        5. put object into/on/in another receptacle
            :action PutObject
            :parameters (?o - object ?r - receptacle)
        6. using an object/receptacle by turning it on/off with a switch
            :action useObject
            :parameters (?o - object)
        7. heat an object using a receptacle
            :action HeatObject
            :parameters (?o - object ?r - microwaveReceptacle)
        8. clean an object using a receptacle
            :action CleanObject
            :parameters (?o - object ?r - sinkbasinReceptacle)
        9. cool an object using a receptacle
            :action CoolObject
            :parameters (?o - object ?r - fridgeReceptacle)
        10. slice an object using a sharp object
            :action SliceObject
            :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)

        You must go to a receptacle first in order to use/open it or take/put objects from/on it.

        The process involves two main stages:

        1. Always searching for the aim Object first!!!
            In this stage, your goal is to go to and may need to open new, unvisited recepatacles until you find the object mentioned in the task. Some receptacles cannot be opened so you can directly see what objects after you go to that receptacle.

            You can only use the GotoLocation action to travel to a new location and the OpenObject action (if the receptacle is closed) to verify whether it contains the target object.

            Goal 1.1: Reach a location that has not been visited (the location should be a receptacle) using the GotoLocation action. 
                You goal should look like this:
                (:goal 
                    (at ?recepatacle)
                ) where recepatacle should be somewhere or some recepatacles not visited.

            Goal 1.2: If you already go to the recepatacle and found the recepatacle is closed, use the OpenObject action to open it and inspect the contents. 
                Your goal should look like this:
                (:goal 
                    (opened ?recepatacle)
                ) where recepatacle should be the recepatacle you want to open.

        2. After you seeing the aim object in any receptacle, using the Object to Complete the Task:
            After you have located the object (the object may have some numbers added), you should always first pick up the object from that receptacle and update your goal to focus on how the object is used to complete the task. Remember your goal is {goal}. Based on different adjectives, you may need to perform different actions for the object in different ways.

            This may involve more than simply transferring it from one place to another.
            For example: You might examine the object or a nearby receptacle to gather information. You may need to use another tool or device (like a lamp or a switch). Some tasks require you to slice, heat, cool, or clean the object using an appropriate receptacle (e.g., microwave, sink, fridge).

            If necessary, use the PickupObject action to retrieve the item, and the GotoLocation action to move to the correct place.
            Then, apply the object in a purposeful way — not just move it — but interact with the environment to fulfill the task’s actual goal.

            Hint: 
            1. If you want to heat, clean, and cool an object, after you go to that aim receptacle, do not put the object in the receptacle but do the action directly. For example, go to fridge, then cool the object with receptacle.
            2. If you want to slice an object, you should first go to the receptacle where both the sharp object and the aim object are located and ONLY pick up the sharp object then do the slice action. Don't forget to put the sharp object back to the receptacle after you finish slicing.
            3. If you want to examine or look at an object with a lamp, you should first go to the receptacle where the object is located and then pick it up and take the USE action of the lamp. You don't need to take the lamp but directly use it.
            4. If there are multiple actions needed to complete the task, you can break them down into smaller subgoals. For example, if you need to slice and then heat an object, first focus on slicing it, and then move on to heating it.

        In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.
        
        Note: 
        1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
        2. Your initial goal should always be to go to a new location instead of put something into somewhere.
        3. Do not enter stage 2 when not finishing stage 1.

        Note: Always include :negative-preconditions in your :requirements whenever you use (not …) or delete effects, and never leave an :precondition or :effect block empty—either omit it or include at least one literal.
    """ 

    prompt_prev_files = f"""
        This is previous domain file: {prev_df}
        This is previous problem file: {prev_pf}
        This is all the memory you have in this game including each action and its corresponding observations: {overall_memory}
    """

    prompt_new_obs = f"""
        Now modify those two files according to the new observations and notes. Fix any errors you made in the previous setting according to the new observation.
        Generate updated files based on your new observation.
    """

    # error from Parser(df, pf)
    prompt_error_parser = f"""
        You made some mistakes when generating those files. Here is the error message: {prev_err}; {prev_err_2}
        Now modify those two files according to the error message.
    """

    # error from simulation environment
    prompt_simulation_error = f"""
        You have already generate files according to the observations. The df and pf can generate actions but after simulating,
        it got those errors: {large_loop_error_message}. Please review both files and fix them.
        Now modify those two files according to the error message.
    """

    prompt_duplicate_note = """
        You are repeating the same sequence of actions for at least three times. You may stuck in one location or have the wrong goal.
        You should revise your problem file to avoid the repeat.
        Remember your goal is always to keep exploration and go to a location you have not visited yet, i.e. your goal should be go to other not visited location but shouldn't be at one fixed location.
    """

    if not edit:
        prompt = prompt_format
    else:
        prompt = prompt_edit

    # all prompts should have observations and actions
    # select which prompt version to use: prompt_obs_action_detailed or prompt_obs_action_general_goal or prompt_obs_action_subgoal
    # default is detailed
    if goal_type == 'detailed':
        prompt += prompt_obs_action_detailed
    elif goal_type == 'subgoal':
        prompt += prompt_obs_action_subgoal
    elif goal_type == 'without_hint':
        prompt += prompt_obs_action_without_hint
    elif goal_type == 'without_detailed_goal':
        prompt += prompt_obs_action_without_detailed_goal
    else:
        prompt += prompt_obs_action_general_goal

    if prev_df and prev_pf:
        prompt += prompt_prev_files

        if not have_error:
            prompt += prompt_new_obs
        else:
            prompt += prompt_error_parser
        
        if large_loop_error_message:
            prompt += prompt_simulation_error

    if have_duplicate:
        # print('You have duplicated error message!!')
        prompt += prompt_duplicate_note


    # if edit:
    #     edit_json_df, edit_json_pf = run_llm(prompt, model_name)
    #     # print("Edit json:",edit_json_df, edit_json_pf)
    #     df, pf = apply_edit(prev_df, prev_pf, edit_json_df, edit_json_pf)
    #     # print("New df and pf:", df, pf)
    # else:
    #     df, pf = run_llm(prompt, model_name)
    df, pf = run_llm(prompt, model_name)

    err = None #error_message(df, pf)
    return df, pf, err, prompt


def llm_to_actions_baseline(model_name, brief_obs, goal, overall_memory=None, large_loop_error_message=None):
    # prompt_general = f"""
    #     You are in an environment that you explore step by step. Based on your observations, generate a series of valid actions to progress in the environment.
    #     Your task is to interact with objects and receptacles to complete a goal step by step.

    #     Your specific task goal: {goal if goal else "Explore and interact meaningfully based on available observations."}

    #     Here are your current observations: {brief_obs}

    #     Valid actions you can take (follow exactly this format, replacing the parts in brackets with actual object and location names. Do not include any brackets in your output):
    #         - go to [towelholder 1]
    #         - open [cabinet 2]
    #         - take [cloth 1] from [cabinet 3]
    #         - move [soap bar 1] to [sink basin 2]
    #         - use [desk lamp 1]
    #         - heat [bread 1] with [microwave 1]
    #         - clean [fork 1] with [sink basin 1]
    #         - cool [wine bottle 1] with [fridge 1]
    #         - slice [bread 1] with [knife 1]
    #     Only replace the object and receptacle names (the words that were inside the brackets) with the actual values from the game environment. Do not change the structure. Do not include brackets.

    #     Your actions should exactly follow this phrasing — do not invent new formats. Every action must correspond to a valid command from the environment.

    #     You are in an unfamiliar environment. Your task is to complete the given objective by observing your surroundings and interacting with objects and receptacles.
    #     You must build and update PDDL files solely based on what you directly observe. Do not make assumptions.

    #     General Goal: Find the necessary object(s) and use them appropriately to accomplish the task.
    #     Use the allowed actions to explore, discover, and act purposefully. Plan each step based on what you know so far.
        
    #     Constraints:
    #         1. Do not assume unseen objects or relationships.
    #         2. Receptacle names must be preserved exactly.
    #         3. Do not proceed to Stage 2 before completing Stage 1.
        
    #     Memory of past steps:
    #     {overall_memory if overall_memory else "No additional memory available."}

    #     If there are errors or obstacles, here is the message:
    #     {large_loop_error_message if large_loop_error_message else "No errors or obstacles mentioned."}

    #     Provide the output in strict JSON format like this:
    #     {{
    #         "actions": ["action1", "action2", ...]
    #     }}
    # """

    # prompt_subgoal = f"""
    #     You are in an environment that you explore step by step. Based on your observations, generate a series of valid actions to progress in the environment.
    #     Your task is to interact with objects and receptacles to complete a goal step by step.

    #     Your specific task goal: {goal if goal else "Explore and interact meaningfully based on available observations."}

    #     Here are your current observations: {brief_obs}

    #     Valid actions you can take (follow exactly this format, replacing the parts in brackets with actual object and location names. Do not include any brackets in your output):
    #         - go to [towelholder 1]
    #         - open [cabinet 2]
    #         - take [cloth 1] from [cabinet 3]
    #         - move [soap bar 1] to [sink basin 2]
    #         - use [desk lamp 1]
    #         - heat [bread 1] with [microwave 1]
    #         - clean [fork 1] with [sink basin 1]
    #         - cool [wine bottle 1] with [fridge 1]
    #         - slice [bread 1] with [knife 1]
    #     Only replace the object and receptacle names (the words that were inside the brackets) with the actual values from the game environment. Do not change the structure. Do not include brackets.

    #     Your actions should exactly follow this phrasing — do not invent new formats. Every action must correspond to a valid command from the environment.

    #     Your process involves two main stages with the following subgoals:

    #     Stage 1: Search for the Target Object
    #         Goal 1.1: Move to a new, unvisited receptacle using the GotoLocation action.
    #         Goal 1.2: If the receptacle is closed, use the OpenObject action to reveal its contents.

    #     Stage 2: Use the Object to Complete the Task
    #         Goal 2.1: Pick up the target object using the PickupObject action.
    #         Goal 2.2: Move to the appropriate location needed to fulfill the task.
    #         Goal 2.3: Interact with relevant objects or receptacles (e.g., heat, clean, cool, slice, or use) to accomplish the task.

    #     Constraints:
    #         1. Do not assume unseen objects or relationships.
    #         2. Receptacle names must be preserved exactly.
    #         3. Do not proceed to Stage 2 before completing Stage 1.

    #     Memory of past steps:
    #     {overall_memory if overall_memory else "No additional memory available."}

    #     If there are errors or obstacles, here is the message:
    #     {large_loop_error_message if large_loop_error_message else "No errors or obstacles mentioned."}

    #     Provide the output in strict JSON format like this:
    #     {{
    #         "actions": ["action1", "action2", ...]
    #     }}
    # """

    prompt_detailed = f"""
        You are in an environment that you explore step by step. Based on your observations, generate one valid action at a time to progress in the environment.
        Your task is to interact with objects and receptacles to complete a goal step by step.

        Your specific task goal: {goal if goal else "Explore and interact meaningfully based on available observations."}

        Here are your current observations: {brief_obs}

        Valid actions you can take (follow exactly this format, replacing the parts in brackets with actual object and location names. Do not include any brackets in your output):
            - go to [towelholder 1]
            - open [cabinet 2]
            - take [cloth 1] from [cabinet 3]
            - move [soap bar 1] to [sink basin 2]
            - use [desk lamp 1]
            - heat [bread 1] with [microwave 1]
            - clean [fork 1] with [sink basin 1]
            - cool [wine bottle 1] with [fridge 1]
            - slice [bread 1] with [knife 1]
        Only replace the object and receptacle names (the words that were inside the brackets) with the actual values from the game environment. Do not change the structure. Do not include brackets.

        Your actions should exactly follow this phrasing — do not invent new formats. Every action must correspond to a valid command from the environment.

        You must go to a receptacle first in order to use/open it or take/put objects from/on it. You can assume all receptacles are freely reachable.

        The process involves two main stages:

        1. Always searching for the aim Object first!!!
            In this stage, your goal is to go to and may need to open new, unvisited recepatacles until you find the object mentioned in the task. Some receptacles cannot be opened so you can directly see what objects after you go to that receptacle.

            You can only use the GotoLocation action to travel to a new location and the OpenObject action (if the receptacle is closed) to verify whether it contains the target object.

        2. Using the Object to Complete the Task:
            Once you have located and picked up the object, update your goal to focus on how the object is used to complete the task. This may involve more than simply transferring it from one place to another.
            For example: You might examine the object or a nearby receptacle to gather information. You may need to use another tool or device (like a lamp or a switch). Some tasks require you to slice, heat, cool, or clean the object using an appropriate receptacle (e.g., microwave, sink, fridge).

            If necessary, use the PickupObject action to retrieve the item, and the GotoLocation action to move to the correct place.
            Then, apply the object in a purposeful way — not just move it — but interact with the environment to fulfill the task’s actual goal.

            Note: if you want to heat, clean, and cool an object, you can go to that receptacle then do the action directly without put the object into that receptacle.
                But if you want to slice an object, you should first go to the receptacle and pick up the sharp object then do the slice action.

        In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.
        
        Note: 
        1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
        2. Your initial goal should always be to go to a new location instead of put something into somewhere.
        3. Do not enter stage 2 when not finishing stage 1.

        Memory of past steps:
        {overall_memory if overall_memory else "No additional memory available."}

        If there are errors or obstacles, here is the message:
        {large_loop_error_message if large_loop_error_message else "No errors or obstacles mentioned."}

        Provide the output in strict JSON format like this while you should only generate one action at a time:
        {{
            "actions": ["action1"]
        }}
    """

    prompt = prompt_detailed

    actions = run_llm_for_actions_baseline(prompt, model_name)
    return actions, prompt


def llm_to_pddl_fixed_df(model_name, brief_obs, goal, goal_type, df, prev_df="", prev_pf="", prev_err="", prev_err_2=None, have_error=False, have_duplicate=False, edit=False, overall_memory=None, large_loop_error_message=None):
    prompt = f"""
        Please provide the output in strict JSON format, without any additional text or explanation, including  a PDDL problem file as 'pf'. The domain file is fixed and will be provided below. It should not be modified.
        The format should strictly be:
            {{
            "pf": "..."
            }}

        You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

        The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.
        
        Now, {goal}
        Here are your current observations: {brief_obs}

        Only the following actions are allowed: (There are only two types: object and receptacle)
        1. go to a receptacle
            :action GotoLocation
            :parameters (?from - receptacle ?to - receptacle)
        2. open a receptacle if it is closed
            :action OpenObject
            :parameters (?r - receptacle)
        3. close a receptacle
            :action CloseObject
            :parameters (?r - receptacle)
        4. take an object from another receptacle
            :action PickupObject
            :parameters (?o - object ?r - receptacle)
        5. put object into/on/in another receptacle
            :action PutObject
            :parameters (?o - object ?r - receptacle)
        6. using an object/receptacle by turning it on/off with a switch
            :action useObject
            :parameters (?o - object)
        7. heat an object using a receptacle
            :action HeatObject
            :parameters (?o - object ?r - microwaveReceptacle)
        8. clean an object using a receptacle
            :action CleanObject
            :parameters (?o - object ?r - sinkbasinReceptacle)
        9. cool an object using a receptacle
            :action CoolObject
            :parameters (?o - object ?r - fridgeReceptacle)
        10. slice an object using a sharp object
            :action SliceObject
            :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)

        You must go to a receptacle first in order to use/open it or take/put objects from/on it.

        The process involves two main stages:

        1. Always searching for the aim Object first!!!
            In this stage, your goal is to go to and may need to open new, unvisited recepatacles until you find the object mentioned in the task. Some receptacles cannot be opened so you can directly see what objects after you go to that receptacle.

            You can only use the GotoLocation action to travel to a new location and the OpenObject action (if the receptacle is closed) to verify whether it contains the target object.

            Goal 1.1: Reach a location that has not been visited (the location should be a receptacle) using the GotoLocation action. 
                You goal should look like this:
                (:goal 
                    (at ?recepatacle)
                ) where recepatacle should be somewhere or some recepatacles not visited.

            Goal 1.2: If you already go to the recepatacle and found the recepatacle is closed, use the OpenObject action to open it and inspect the contents. 
                Your goal should look like this:
                (:goal 
                    (opened ?recepatacle)
                ) where recepatacle should be the recepatacle you want to open.

        2. After you seeing the aim object in any receptacle, using the Object to Complete the Task:
            After you have located the object (the object may have some numbers added), you should always first pick up the object from that receptacle and update your goal to focus on how the object is used to complete the task. Remember your goal is {goal}. Based on different adjectives, you may need to perform different actions for the object in different ways.

            This may involve more than simply transferring it from one place to another.
            For example: You might examine the object or a nearby receptacle to gather information. You may need to use another tool or device (like a lamp or a switch). Some tasks require you to slice, heat, cool, or clean the object using an appropriate receptacle (e.g., microwave, sink, fridge).

            If necessary, use the PickupObject action to retrieve the item, and the GotoLocation action to move to the correct place.
            Then, apply the object in a purposeful way — not just move it — but interact with the environment to fulfill the task’s actual goal.

            Hint: 
            1. If you want to heat, clean, and cool an object, after you go to that aim receptacle, do not put the object in the receptacle but do the action directly. For example, go to fridge, then cool the object with receptacle.
            2. If you want to slice an object, you should first go to the receptacle where both the sharp object and the aim object are located and ONLY pick up the sharp object then do the slice action. Don't forget to put the sharp object back to the receptacle after you finish slicing.
            3. If you want to examine or look at an object with a lamp, you should first go to the receptacle where the object is located and then pick it up and take the USE action of the lamp. You don't need to take the lamp but directly use it.
            4. If there are multiple actions needed to complete the task, you can break them down into smaller subgoals. For example, if you need to slice and then heat an object, first focus on slicing it, and then move on to heating it.

        In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.
        
        Note: 
        1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
        2. Your initial goal should always be to go to a new location instead of put something into somewhere.
        3. Do not enter stage 2 when not finishing stage 1.

        Note: Always include :negative-preconditions in your :requirements whenever you use (not …) or delete effects, and never leave an :precondition or :effect block empty—either omit it or include at least one literal.

        This is the fixed domain file and you should not modify it: 
        {df}
    """

    prompt_prev_files = f"""
        This is previous problem file: {prev_pf}
        This is all the memory you have in this game including each action and its corresponding observations: {overall_memory}
    """

    prompt_new_obs = f"""
        Now modify those two files according to the new observations and notes. Fix any errors you made in the previous setting according to the new observation.
        Generate updated files based on your new observation.
    """

    # error from Parser(df, pf)
    prompt_error_parser = f"""
        You made some mistakes when generating those files. Here is the error message: {prev_err}; {prev_err_2}
        Now modify those two files according to the error message.
    """

    # error from simulation environment
    prompt_simulation_error = f"""
        You have already generate files according to the observations. The df and pf can generate actions but after simulating,
        it got those errors: {large_loop_error_message}. Please review both files and fix them.
        Now modify those two files according to the error message.
    """

    prompt_duplicate_note = """
        You are repeating the same sequence of actions for at least three times. You may stuck in one location or have the wrong goal.
        You should revise your problem file to avoid the repeat.
        Remember your goal is always to keep exploration and go to a location you have not visited yet, i.e. your goal should be go to other not visited location but shouldn't be at one fixed location.
    """

    if prev_pf:
        prompt += prompt_prev_files

        if not have_error:
            prompt += prompt_new_obs
        else:
            prompt += prompt_error_parser
        
        if large_loop_error_message:
            prompt += prompt_simulation_error

    if have_duplicate:
        prompt += prompt_duplicate_note


    # if edit:
    #     edit_json_df, edit_json_pf = run_llm(prompt, model_name)
    #     # print("Edit json:",edit_json_df, edit_json_pf)
    #     df, pf = apply_edit(prev_df, prev_pf, edit_json_df, edit_json_pf)
    #     # print("New df and pf:", df, pf)
    # else:
    #     df_None, pf = run_llm(prompt, model_name)

    df_None, pf = run_llm(prompt, model_name)

    err = None #error_message(df, pf)
    return pf, err, prompt


# ===== Main functions here =====
def run_iterative_model(model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", start_trial = 0, end_trial = 11, folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed"):
    # trial_record = 
    # structured_info_record = "output/summary"
    for trial in range(start_trial, end_trial):
        retry = 0
        while retry < 2:  # allow up to 2 attempts per trial
            try:
                succeed = False
                today = date.today()
                fixed_model_name = model_name.replace("/","_")
                folder_path = f"output/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_name = f"{folder_path}/{today}_{fixed_model_name}_PDDL_{goal_type}_{trial}.txt"

                if os.path.exists(file_name): # retry == 1 and 
                    open(file_name, 'w').close()  # empty file
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                trial_record = []
                
                # each trial reset environment ===================
                problem_id = random.randint(0, 9)
                problem = os.path.dirname(PROBLEMS[problem_id])
                problem_type_dic = {0: 'clean', 1: 'basic', 2: 'basic', 3:'slice & heat', 4: 'heat',\
                    5:'use', 6:'clean', 7: 'use', 8: 'basic', 9:'cool'}
                game_type = problem_type_dic[problem_id] # set game_type here!
                print(f"Playing {problem_id}: {problem}")
                domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl") # domain file
                grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
                GAME_LOGIC = {
                        "pddl_domain": open(domain).read(),
                        "grammar": open(grammar).read(),
                    }
                pddl_file = os.path.join(problem, 'initial_state.pddl') # problem file
                json_file = os.path.join(problem, 'traj_data.json')
                with open(json_file, 'r') as f:
                    traj_data = json.load(f)
                GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)
                gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
                gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
                json.dump(gamedata, open(gamefile, "w"))
                expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

                request_infos = textworld.EnvInfos(
                    won=True,
                    admissible_commands=True,
                    score=True,
                    max_score=True,
                    intermediate_reward=True,
                    extras=["expert_plan"]
                )
                env_id = textworld.gym.register_game(
                    gamefile,
                    request_infos,
                    max_episode_steps=1000000,
                    wrappers=[AlfredDemangler(), expert]
                )
                env = textworld.gym.make(env_id)

                # reset environment
                obs, infos = env.reset()
                init_obs = obs.split('\n')[2]
                goal = obs.split('\n')[-1]
                valid_actions = infos["admissible_commands"]

                with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
                    f.write(f"Playing {problem_id}: {problem} \n")
                    f.write(f"Observations: {init_obs} \n") 
                    f.write(f"Valid Actions: {valid_actions} \n")
                    f.write(f"taskDescription: {goal} \n")

                # task_description = infos['taskDescription']
                valid_actions = sorted(valid_actions)
                valid_actions.remove('look')
                valid_actions.remove('inventory')
                valid_actions.remove('help') # add help?

                MAX_STEPS = 50

                brief_obs = "Action: look around\n" + summarize_obs(init_obs)+'\n' # initial definition
                with open(file_name, "a") as f:
                    f.write(f"brief_obs: {brief_obs} \n") 

                action_queue = []
                obs_queue = []
                df = ""
                pf = ""
                all_actions = []
                successful_actions = []
                edit = False
                end_game = False

                overall_memory = brief_obs

                for step_id in range(0, MAX_STEPS):
                    with open(file_name, "a") as f:
                        f.write(f"\n\n====Step {step_id}==== \n")

                    trial_step_record = []
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""

                    # Under step#, it should repeat until run all actions and found no error
                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a") as f:
                            f.write(f'\n----Larger Loop No. {within_step_tries}---- \n') 
                            f.write(f'successful_actions: {successful_actions} \n')

                        within_step_tries += 1

                        if within_step_tries > 1: 
                            env_id = textworld.gym.register_game(
                                gamefile,
                                request_infos,
                                max_episode_steps=1000000,
                                wrappers=[AlfredDemangler(), expert]
                            )
                            env = textworld.gym.make(env_id)

                            # reset environment
                            obs, infos = env.reset()
                            for successful_action in successful_actions:
                                obs, score, done, infos = env.step(successful_action)

                        action_queue = [] # reset action_queue
                        tem_action_queue = []
                        tem_memory = ""

                        start_checkpoint = True
                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f'Small Loop, action_queue: {action_queue} \n')
                            start_checkpoint = False

                            if not action_queue:
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                action = ""
                                
                                if not df and not pf: # First step no need duplicates detection
                                    num_tries = 0
                                    df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type) # error 1 here
                                    action, err_2 = get_action_from_pddl(df, pf) # error 2 here
                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Error: {err} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")

                                    while not action and num_tries < 5:
                                        df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type, df, pf, err, err_2, True, False, edit)
                                        action, err_2 = get_action_from_pddl(df, pf)
                                        num_tries += 1
                                        
                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Error: {err} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                else:
                                    num_tries = 0
                                    # Every time read new error message from larger loop
                                    # In llm_to_pddl, detect if new large loop error message exists
                                    df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type, df, pf, err, None, False, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message) # need to add new error message
                                    action, err_2 = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Error: {err} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")

                                    while not action and num_tries < 5:
                                        df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type, df, pf, err, err_2, True, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message)
                                        action, err_2 = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Error: {err} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")

                                # append which loop it stops
                                # Must be under first time generating the actions
                                trial_step_record.append([within_step_tries, num_tries])

                                if action:
                                    action_queue.extend(action)
                                    tem_action_queue.extend(action) # temporary action queue to put in successful_actions
                                    all_actions.extend(action) # to detect duplicated
                                else:
                                    end_game = True
                                    break

                            with open(file_name, "a") as f:
                                f.write(f"Current action_queue: {action_queue} \n")
                            
                            taken_action = action_queue.pop(0)

                            obs, reward, done, infos = env.step(taken_action)
                            
                            # Directly end the game if Done!
                            if infos["won"]:
                                end_game = True
                                succeed = True
                                with open(file_name, "a") as f:
                                    f.write('Done!')
                                break
                            
                            action_text = "Action: " + taken_action + "\n"
                            obs_text = summarize_obs(obs) + "\n"

                            brief_obs = action_text + obs_text

                            obs_queue.append(brief_obs)
                            with open(file_name, "a") as f:
                                f.write(f"> {brief_obs} \n")
                                f.write(f"After taking action '{taken_action}', you have the following valid actions: {infos['admissible_commands']} \n")


                            if "Nothing happens." in brief_obs:
                                large_loop_error_message = f"""In this step, you take the following actions and observations from those actions:
                                {''.join(obs_queue)}"""
                                if "go to" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to go to a receptacle but nothing happens. 
                                    You may already been at this receptacle, in other words, you have already went to this place and do not need to go to this receptacle again.
                                    Otherwise, there is no the receptacle you are aiming to."""
                                    continue
                                elif "open" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to open a receptacle but nothing happens. 
                                    You should first go to this receptacle to open it. 
                                    But if you have already go to this receptacle and still seeing this error message, it means that this receptacle cannot be opened and you can directly see objects after you go to it. Do not try to open it!!"""
                                elif "take" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to take something from a receptacle.
                                    You should first go to this receptacle to take the object.
                                    But if you have already go to this receptacle and still seeing this error message, it means that this receptacle doesn't have this object.
                                    You should go to other receptacle to find your aim object. Remember do not assume you can take the object from the receptable but should always set the initial goal as finding that aim object."""
                                elif "move" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}.
                                    You want to move some object to a receptacle but failed. You should first find that object somewhere by going to an unvisited receptacle and open if necessary.
                                    Then pick up the aiming object so that you can go to your aim receptacle and put it there.
                                    """
                                elif "slice" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to slice an object with a sharp object.
                                    You should first pickup the sharp object (this should be the only object you pick up) then take the slice action directly without picking up the aim object!
                                    Don't forget to put the sharp object back to the receptacle after you finish slicing."""
                                elif "cool" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to cool an object with a fridge. 
                                    You need to find the object and pick it up from other receptacle. Then go to frige and cool the object directly. Notice: do not move the object to the fridge but cool directly!"""
                                elif ("fridge" in taken_action or "sinkbasin" in taken_action or "microwave" in taken_action) and ("move" in taken_action or "take" in taken_action): # pass this
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to move or take an object to or from a fridge. 
                                    You don't need to take this action! You should go to fridge receptacle, cool the object, go to another receptacle"""
                                    continue
                                elif "use" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to use an object.
                                    You can only use a lamp to turn it on and look at or examine other objects. Note: to look at or examine other objects, you should first pick it up."""
                                break

                            # append into overall memory and dictionary format
                            tem_memory += brief_obs

                            # It should be the last step and passed all actions
                            if not action_queue:
                                action_passed = True
                                successful_actions.extend(tem_action_queue)
                                overall_memory += tem_memory

                        if (within_step_tries == 5 and not action_passed) or end_game:
                            end_game = True
                            break

                    trial_record.append(trial_step_record)

                    if end_game:
                        break
                
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile: 
                    # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
                    model_type = 'PDDL'
                    data_row = [today, model_name, model_type, game_type, goal_type, trial, succeed, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    writer = csv.writer(csvfile)
                    writer.writerow(data_row)

                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    log_message = (
                        f"[PDDLego+] Trial {trial} (Attempt {retry+1}) | "
                        f"Model: {model_name} | Goal Type: {goal_type} | "
                        f"Failed: {str(e)}\n"
                    )
                    f.write(log_message)
                retry += 1

def run_iterative_model_50(model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed", trials_to_run=None):
    trial = 0
    for game_type, game_lst in GAME_DICT.items():
        game_lst_sep = game_lst[:4] # only run 4 trials for each game type
        # game_lst_sep = game_lst*2
        for problem_id in game_lst_sep: # extra indent
            trial += 1
            # if trial < 86: 
            #     continue
            # print(f"Trial {trial} for {game_type} game type")

            # if trials_to_run and trial not in trials_to_run: # skip trials not in the list
            #     continue

            retry = 0
            while retry < 2:  # allow up to 2 attempts per trial
                try:
                    succeed = False
                    today = date.today()
                    fixed_model_name = model_name.replace("/","_")
                    folder_path = f"output/{folder_name}"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    file_name = f"{folder_path}/{today}_{fixed_model_name}_PDDL_{goal_type}_{game_type}_{trial}.txt"

                    if os.path.exists(file_name): # retry == 1 and 
                        open(file_name, 'w').close()  # empty file
                        print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                    trial_record = []
                    
                    # each trial reset environment ===================
                    problem = os.path.dirname(PROBLEMS[problem_id])
                    print(f"Playing {problem_id}: {problem}")
                    domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
                    grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
                    GAME_LOGIC = {
                            "pddl_domain": open(domain).read(),
                            "grammar": open(grammar).read(),
                        }
                    pddl_file = os.path.join(problem, 'initial_state.pddl')
                    json_file = os.path.join(problem, 'traj_data.json')
                    with open(json_file, 'r') as f:
                        traj_data = json.load(f)
                    GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)
                    gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
                    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
                    json.dump(gamedata, open(gamefile, "w"))
                    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

                    request_infos = textworld.EnvInfos(
                        won=True,
                        admissible_commands=True,
                        score=True,
                        max_score=True,
                        intermediate_reward=True,
                        extras=["expert_plan"]
                    )
                    env_id = textworld.gym.register_game(
                        gamefile,
                        request_infos,
                        max_episode_steps=1000000,
                        wrappers=[AlfredDemangler(), expert]
                    )
                    env = textworld.gym.make(env_id)

                    # reset environment
                    obs, infos = env.reset()
                    init_obs = obs.split('\n')[2]
                    goal = obs.split('\n')[-1]
                    valid_actions = infos["admissible_commands"]

                    with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
                        f.write(f"Playing {problem_id}: {problem} \n")
                        f.write(f"Observations: {init_obs} \n") 
                        f.write(f"Valid Actions: {valid_actions} \n")
                        f.write(f"taskDescription: {goal} \n")

                    # task_description = infos['taskDescription']
                    valid_actions = sorted(valid_actions)
                    valid_actions.remove('look')
                    valid_actions.remove('inventory')
                    valid_actions.remove('help') # add help?

                    MAX_STEPS = 50

                    brief_obs = "Action: look around\n" + summarize_obs(init_obs)+'\n' # initial definition
                    with open(file_name, "a") as f:
                        f.write(f"brief_obs: {brief_obs} \n") 

                    action_queue = []
                    obs_queue = []
                    df = ""
                    pf = ""
                    all_actions = []
                    successful_actions = []
                    edit = False
                    end_game = False

                    overall_memory = brief_obs

                    for step_id in range(0, MAX_STEPS):
                        with open(file_name, "a") as f:
                            f.write(f"\n\n====Step {step_id}==== \n")

                        trial_step_record = []
                        within_step_tries = 0
                        action_passed = False
                        large_loop_error_message = ""

                        # Under step#, it should repeat until run all actions and found no error
                        while within_step_tries < 5 and not action_passed:
                            with open(file_name, "a") as f:
                                f.write(f'\n----Larger Loop No. {within_step_tries}---- \n') 
                                f.write(f'successful_actions: {successful_actions} \n')

                            within_step_tries += 1

                            if within_step_tries > 1: 
                                env_id = textworld.gym.register_game(
                                    gamefile,
                                    request_infos,
                                    max_episode_steps=1000000,
                                    wrappers=[AlfredDemangler(), expert]
                                )
                                env = textworld.gym.make(env_id)

                                # reset environment
                                obs, infos = env.reset()
                                for successful_action in successful_actions:
                                    obs, score, done, infos = env.step(successful_action)

                            action_queue = [] # reset action_queue
                            tem_action_queue = []
                            tem_memory = ""

                            start_checkpoint = True
                            while start_checkpoint or action_queue:
                                with open(file_name, "a") as f:
                                    f.write(f'Small Loop, action_queue: {action_queue} \n')
                                start_checkpoint = False

                                if not action_queue:
                                    if obs_queue:
                                        brief_obs = "\n".join(obs_queue)
                                        obs_queue = []
                                    action = ""
                                    
                                    if not df and not pf: # First step no need duplicates detection
                                        num_tries = 0
                                        df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal_type=goal_type, goal=goal) # error 1 here
                                        action, err_2 = get_action_from_pddl(df, pf) # error 2 here
                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Error: {err} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")

                                        while not action and num_tries < 5:
                                            df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, df, pf, err, err_2, True, False, edit, goal_type=goal_type, goal=goal)
                                            action, err_2 = get_action_from_pddl(df, pf)
                                            num_tries += 1
                                            
                                            with open(file_name, "a") as f:
                                                f.write(f"--Small Loop--: {num_tries} \n")
                                                f.write(f"Error: {err} \n")
                                                f.write(f"Prompt: {prompt} \n") 
                                                f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                                f.write(f"Actions from solver(df, pf): {action} \n")
                                    else:
                                        num_tries = 0
                                        # Every time read new error message from larger loop
                                        # In llm_to_pddl, detect if new large loop error message exists
                                        df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, df, pf, err, None, False, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message, goal_type=goal_type, goal=goal) # need to add new error message
                                        action, err_2 = get_action_from_pddl(df, pf)

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Error: {err} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")

                                        while not action and num_tries < 5:
                                            df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, df, pf, err, err_2, True, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message, goal_type=goal_type, goal=goal)
                                            action, err_2 = get_action_from_pddl(df, pf)
                                            num_tries += 1

                                            with open(file_name, "a") as f:
                                                f.write(f"--Small Loop--: {num_tries} \n")
                                                f.write(f"Error: {err} \n")
                                                f.write(f"Prompt: {prompt} \n") 
                                                f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                                f.write(f"Actions from solver(df, pf): {action} \n")

                                    # append which loop it stops
                                    # Must be under first time generating the actions
                                    trial_step_record.append([within_step_tries, num_tries])

                                    if action:
                                        action_queue.extend(action)
                                        tem_action_queue.extend(action) # temporary action queue to put in successful_actions
                                        all_actions.extend(action) # to detect duplicated
                                    else:
                                        end_game = True
                                        break

                                with open(file_name, "a") as f:
                                    f.write(f"Current action_queue: {action_queue} \n")
                                
                                taken_action = action_queue.pop(0)

                                obs, reward, done, infos = env.step(taken_action)
                                
                                # Directly end the game if Done!
                                if infos["won"]:
                                    end_game = True
                                    succeed = True
                                    with open(file_name, "a") as f:
                                        f.write('Done!')
                                    break
                                
                                action_text = "Action: " + taken_action + "\n"
                                obs_text = summarize_obs(obs) + "\n"

                                brief_obs = action_text + obs_text

                                obs_queue.append(brief_obs)
                                with open(file_name, "a") as f:
                                    f.write(f"> {brief_obs} \n")
                                    f.write(f"After taking action '{taken_action}', you have the following valid actions: {infos['admissible_commands']} \n")


                                if "Nothing happens." in brief_obs:
                                    large_loop_error_message = f"""In this step, you take the following actions and observations from those actions:
                                    {''.join(obs_queue)}"""
                                    if "go to" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to go to a receptacle but nothing happens. 
                                        You may already been at this receptacle, in other words, you have already went to this place and do not need to go to this receptacle again.
                                        Otherwise, there is no the receptacle you are aiming to."""
                                        continue
                                    elif "open" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to open a receptacle but nothing happens. 
                                        You should first go to this receptacle to open it. 
                                        But if you have already go to this receptacle and still seeing this error message, it means that this receptacle cannot be opened and you can directly see objects after you go to it. Do not try to open it!!"""
                                    elif "take" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to take something from a receptacle.
                                        You should first go to this receptacle to take the object.
                                        But if you have already go to this receptacle and still seeing this error message, it means that this receptacle doesn't have this object.
                                        You should go to other receptacle to find your aim object. Remember do not assume you can take the object from the receptable but should always set the initial goal as finding that aim object."""
                                    elif "move" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}.
                                        You want to move some object to a receptacle but failed. You should first find that object somewhere by going to an unvisited receptacle and open if necessary.
                                        Then pick up the aiming object so that you can go to your aim receptacle and put it there.
                                        """
                                    elif "slice" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to slice an object with a sharp object.
                                        You should first pickup the sharp object (this should be the only object you pick up) then take the slice action directly without picking up the aim object!
                                        Don't forget to put the sharp object back to the receptacle after you finish slicing."""
                                    elif "cool" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to cool an object with a fridge. 
                                        You need to find the object and pick it up from other receptacle. Then go to frige and cool the object directly. Notice: do not move the object to the fridge but cool directly!"""
                                    elif ("fridge" in taken_action or "sinkbasin" in taken_action or "microwave" in taken_action) and ("move" in taken_action or "take" in taken_action): # pass this
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to move or take an object to or from a fridge. 
                                        You don't need to take this action! You should go to fridge receptacle, cool the object, go to another receptacle"""
                                        continue
                                    elif "use" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to use an object.
                                        You can only use a lamp to turn it on and look at or examine other objects. Note: to look at or examine other objects, you should first pick it up."""
                                    break

                                # append into overall memory and dictionary format
                                tem_memory += brief_obs

                                # It should be the last step and passed all actions
                                if not action_queue:
                                    action_passed = True
                                    successful_actions.extend(tem_action_queue)
                                    overall_memory += tem_memory

                            if (within_step_tries == 5 and not action_passed) or end_game:
                                end_game = True
                                break

                        trial_record.append(trial_step_record)

                        if end_game:
                            break
                    
                    with open(f"output/{result_name}.csv", "a", newline="") as csvfile: 
                        # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
                        model_type = 'PDDL'
                        data_row = [today, model_name, model_type, game_type, goal_type, trial, succeed, len(trial_record)-1,trial_record[-1][-1], trial_record]
                        writer = csv.writer(csvfile)
                        writer.writerow(data_row)

                    break
                except Exception as e:
                    error_log_path = f"output/{folder_name}/errors.txt"
                    with open(error_log_path, "a") as f:
                        log_message = (
                            f"[PDDLego+ 50] Trial {trial} (Attempt {retry+1}) | "
                            f"Game Type: {game_type} | Model: {model_name} | Goal Type: {goal_type} | "
                            f"Failed: {str(e)}\n"
                        )
                        f.write(log_message)
                    retry += 1

def run_iterative_model_fixed_df(model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed", trials_to_run=None):
    trial = 0
    domain_files = [f"df_cache/df_AlfW_{i}.pddl" for i in range(1, 11)]
    for domain_idx, df_path in enumerate(domain_files, start=1):
        with open(df_path, "r") as f:
            df = f.read()
        for game_type, game_lst in GAME_DICT.items():
            game_lst = game_lst[:2]
            for problem_id in game_lst: # extra indent
                trial += 1
                if trial < 49: 
                    continue
                print(f"Trial {trial} for {game_type} game type")

                # if trials_to_run and trial not in trials_to_run: # skip trials not in the list
                #     continue

                retry = 0
                while retry < 2:  # allow up to 2 attempts per trial
                    try:
                        succeed = False
                        today = date.today()
                        fixed_model_name = model_name.replace("/","_")
                        folder_path = f"output/{folder_name}"
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        file_name = f"{folder_path}/{today}_{fixed_model_name}_PDDL_fixed_df_{goal_type}_{game_type}_{trial}.txt"

                        if os.path.exists(file_name): # retry == 1 and 
                            open(file_name, 'w').close()  # empty file
                            print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                        trial_record = []
                        
                        # each trial reset environment ===================
                        problem = os.path.dirname(PROBLEMS[problem_id])
                        print(f"Playing {problem_id}: {problem}")
                        domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
                        grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
                        GAME_LOGIC = {
                                "pddl_domain": open(domain).read(),
                                "grammar": open(grammar).read(),
                            }
                        pddl_file = os.path.join(problem, 'initial_state.pddl')
                        json_file = os.path.join(problem, 'traj_data.json')
                        with open(json_file, 'r') as f:
                            traj_data = json.load(f)
                        GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)
                        gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
                        gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
                        json.dump(gamedata, open(gamefile, "w"))
                        expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

                        request_infos = textworld.EnvInfos(
                            won=True,
                            admissible_commands=True,
                            score=True,
                            max_score=True,
                            intermediate_reward=True,
                            extras=["expert_plan"]
                        )
                        env_id = textworld.gym.register_game(
                            gamefile,
                            request_infos,
                            max_episode_steps=1000000,
                            wrappers=[AlfredDemangler(), expert]
                        )
                        env = textworld.gym.make(env_id)

                        # reset environment
                        obs, infos = env.reset()
                        init_obs = obs.split('\n')[2]
                        goal = obs.split('\n')[-1]
                        valid_actions = infos["admissible_commands"]

                        with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
                            f.write(f"Playing {problem_id}: {problem} \n")
                            f.write(f"Observations: {init_obs} \n") 
                            f.write(f"Valid Actions: {valid_actions} \n")
                            f.write(f"taskDescription: {goal} \n")

                        # task_description = infos['taskDescription']
                        valid_actions = sorted(valid_actions)
                        valid_actions.remove('look')
                        valid_actions.remove('inventory')
                        valid_actions.remove('help') # add help?

                        MAX_STEPS = 50

                        brief_obs = "Action: look around\n" + summarize_obs(init_obs)+'\n' # initial definition
                        with open(file_name, "a") as f:
                            f.write(f"brief_obs: {brief_obs} \n") 

                        action_queue = []
                        obs_queue = []
                        pf = ""
                        all_actions = []
                        successful_actions = []
                        edit = False
                        end_game = False

                        overall_memory = brief_obs

                        for step_id in range(0, MAX_STEPS):
                            with open(file_name, "a") as f:
                                f.write(f"\n\n====Step {step_id}==== \n")

                            trial_step_record = []
                            within_step_tries = 0
                            action_passed = False
                            large_loop_error_message = ""

                            # Under step#, it should repeat until run all actions and found no error
                            while within_step_tries < 5 and not action_passed:
                                with open(file_name, "a") as f:
                                    f.write(f'\n----Larger Loop No. {within_step_tries}---- \n') 
                                    f.write(f'successful_actions: {successful_actions} \n')

                                within_step_tries += 1

                                if within_step_tries > 1: 
                                    env_id = textworld.gym.register_game(
                                        gamefile,
                                        request_infos,
                                        max_episode_steps=1000000,
                                        wrappers=[AlfredDemangler(), expert]
                                    )
                                    env = textworld.gym.make(env_id)

                                    # reset environment
                                    obs, infos = env.reset()
                                    for successful_action in successful_actions:
                                        obs, score, done, infos = env.step(successful_action)

                                action_queue = [] # reset action_queue
                                tem_action_queue = []
                                tem_memory = ""

                                start_checkpoint = True
                                while start_checkpoint or action_queue:
                                    with open(file_name, "a") as f:
                                        f.write(f'Small Loop, action_queue: {action_queue} \n')
                                    start_checkpoint = False

                                    if not action_queue:
                                        if obs_queue:
                                            brief_obs = "\n".join(obs_queue)
                                            obs_queue = []
                                        action = ""
                                        
                                        if not pf: # First step no need duplicates detection
                                            num_tries = 0
                                            pf, err, prompt = llm_to_pddl_fixed_df(model_name, brief_obs,goal, goal_type, df) # error 1 here
                                            action, err_2 = get_action_from_pddl(df, pf) # error 2 here
                                            with open(file_name, "a") as f:
                                                f.write(f"--Small Loop--: {num_tries} \n")
                                                f.write(f"Error: {err} \n")
                                                f.write(f"Prompt: {prompt} \n") 
                                                f.write(f"Generated pf: \n {pf} \n") 
                                                f.write(f"Actions from solver(df, pf): {action} \n")

                                            while not action and num_tries < 5:
                                                pf, err, prompt = llm_to_pddl_fixed_df(model_name, brief_obs, goal, goal_type, df, df, pf, err, err_2, True, False, edit)
                                                action, err_2 = get_action_from_pddl(df, pf)
                                                num_tries += 1
                                                
                                                with open(file_name, "a") as f:
                                                    f.write(f"--Small Loop--: {num_tries} \n")
                                                    f.write(f"Error: {err} \n")
                                                    f.write(f"Prompt: {prompt} \n") 
                                                    f.write(f"Generated pf: \n {pf} \n") 
                                                    f.write(f"Actions from solver(df, pf): {action} \n")
                                        else:
                                            num_tries = 0
                                            # Every time read new error message from larger loop
                                            # In llm_to_pddl, detect if new large loop error message exists
                                            pf, err, prompt = llm_to_pddl_fixed_df(model_name, brief_obs, goal, goal_type, df, df, pf, err, None, False, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message) # need to add new error message
                                            action, err_2 = get_action_from_pddl(df, pf)

                                            with open(file_name, "a") as f:
                                                f.write(f"--Small Loop--: {num_tries} \n")
                                                f.write(f"Error: {err} \n")
                                                f.write(f"Prompt: {prompt} \n") 
                                                f.write(f"Generated pf: \n {pf} \n") 
                                                f.write(f"Actions from solver(df, pf): {action} \n")

                                            while not action and num_tries < 5:
                                                pf, err, prompt = llm_to_pddl_fixed_df(model_name, brief_obs, goal, goal_type, df, df, pf, err, err_2, True, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message)
                                                action, err_2 = get_action_from_pddl(df, pf)
                                                num_tries += 1

                                                with open(file_name, "a") as f:
                                                    f.write(f"--Small Loop--: {num_tries} \n")
                                                    f.write(f"Error: {err} \n")
                                                    f.write(f"Prompt: {prompt} \n") 
                                                    f.write(f"Generated pf: \n {pf} \n") 
                                                    f.write(f"Actions from solver(df, pf): {action} \n")

                                        # append which loop it stops
                                        # Must be under first time generating the actions
                                        trial_step_record.append([within_step_tries, num_tries])

                                        if action:
                                            action_queue.extend(action)
                                            tem_action_queue.extend(action) # temporary action queue to put in successful_actions
                                            all_actions.extend(action) # to detect duplicated
                                        else:
                                            end_game = True
                                            break

                                    with open(file_name, "a") as f:
                                        f.write(f"Current action_queue: {action_queue} \n")
                                    
                                    taken_action = action_queue.pop(0)

                                    obs, reward, done, infos = env.step(taken_action)
                                    
                                    # Directly end the game if Done!
                                    if infos["won"]:
                                        end_game = True
                                        succeed = True
                                        with open(file_name, "a") as f:
                                            f.write('Done!')
                                        break
                                    
                                    action_text = "Action: " + taken_action + "\n"
                                    obs_text = summarize_obs(obs) + "\n"

                                    brief_obs = action_text + obs_text

                                    obs_queue.append(brief_obs)
                                    with open(file_name, "a") as f:
                                        f.write(f"> {brief_obs} \n")
                                        f.write(f"After taking action '{taken_action}', you have the following valid actions: {infos['admissible_commands']} \n")


                                    if "Nothing happens." in brief_obs:
                                        large_loop_error_message = f"""In this step, you take the following actions and observations from those actions:
                                        {''.join(obs_queue)}"""
                                        if "go to" in taken_action:
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to go to a receptacle but nothing happens. 
                                            You may already been at this receptacle, in other words, you have already went to this place and do not need to go to this receptacle again.
                                            Otherwise, there is no the receptacle you are aiming to."""
                                            continue
                                        elif "open" in taken_action:
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to open a receptacle but nothing happens. 
                                            You should first go to this receptacle to open it. 
                                            But if you have already go to this receptacle and still seeing this error message, it means that this receptacle cannot be opened and you can directly see objects after you go to it. Do not try to open it!!"""
                                        elif "take" in taken_action:
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to take something from a receptacle.
                                            You should first go to this receptacle to take the object.
                                            But if you have already go to this receptacle and still seeing this error message, it means that this receptacle doesn't have this object.
                                            You should go to other receptacle to find your aim object. Remember do not assume you can take the object from the receptable but should always set the initial goal as finding that aim object."""
                                        elif "move" in taken_action:
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}.
                                            You want to move some object to a receptacle but failed. You should first find that object somewhere by going to an unvisited receptacle and open if necessary.
                                            Then pick up the aiming object so that you can go to your aim receptacle and put it there.
                                            """
                                        elif "slice" in taken_action:
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to slice an object with a sharp object.
                                            You should first pickup the sharp object (this should be the only object you pick up) then take the slice action directly without picking up the aim object!
                                            Don't forget to put the sharp object back to the receptacle after you finish slicing."""
                                        elif "cool" in taken_action:
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to cool an object with a fridge. 
                                            You need to find the object and pick it up from other receptacle. Then go to frige and cool the object directly. Notice: do not move the object to the fridge but cool directly!"""
                                        elif ("fridge" in taken_action or "sinkbasin" in taken_action or "microwave" in taken_action) and ("move" in taken_action or "take" in taken_action): # pass this
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to move or take an object to or from a fridge. 
                                            You don't need to take this action! You should go to fridge receptacle, cool the object, go to another receptacle"""
                                            continue
                                        elif "use" in taken_action:
                                            large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to use an object.
                                            You can only use a lamp to turn it on and look at or examine other objects. Note: to look at or examine other objects, you should first pick it up."""
                                        break

                                    # append into overall memory and dictionary format
                                    tem_memory += brief_obs

                                    # It should be the last step and passed all actions
                                    if not action_queue:
                                        action_passed = True
                                        successful_actions.extend(tem_action_queue)
                                        overall_memory += tem_memory

                                if (within_step_tries == 5 and not action_passed) or end_game:
                                    end_game = True
                                    break

                            trial_record.append(trial_step_record)

                            if end_game:
                                break
                        
                        with open(f"output/{result_name}.csv", "a", newline="") as csvfile: 
                            # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
                            model_type = 'PDDL_fixed_df'
                            data_row = [today, model_name, model_type, game_type, goal_type, trial, succeed, len(trial_record)-1,trial_record[-1][-1], trial_record]
                            writer = csv.writer(csvfile)
                            writer.writerow(data_row)

                        break
                    
                    except Exception as e:
                        error_log_path = f"output/{folder_name}/errors.txt"
                        with open(error_log_path, "a") as f:
                            # Add a specific prefix and more context
                            log_message = (
                                f"[PDDLego+ Fixed DF] Trial {trial} (Attempt {retry+1}) | "
                                f"Game Type: {game_type} | Model: {model_name} | Goal Type: {goal_type} | "
                                f"Failed: {str(e)}\n"
                            )
                            f.write(log_message)
                        retry += 1

def run_baseline_alfworld(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", start_trial=0, end_trial=5, folder_name="08_031825_alfworld", result_name="alfworld_results"):
    for trial in range(start_trial, end_trial):
        retry = 0
        while retry < 2:  # allow up to 2 attempts per trial
            try:
                succeed = False
                today = date.today()
                fixed_model_name = model_name.replace("/", "_")
                folder_path = f"output/{folder_name}"
                os.makedirs(folder_path, exist_ok=True)
                file_name = f"{folder_path}/{today}_{fixed_model_name}_baseline_{trial}.txt"

                if retry == 1 and os.path.exists(file_name):
                    open(file_name, 'w').close()  # empty file
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")
                trial_record = []

                # each trial reset environment ===================
                problem_id = random.randint(0, 9)
                problem = os.path.dirname(PROBLEMS[problem_id])
                problem_type_dic = {0: 'clean', 1: 'basic', 2: 'basic', 3:'slice & heat', 4: 'heat',\
                    5:'use', 6:'clean', 7: 'use', 8: 'basic', 9:'cool'}
                game_type = problem_type_dic[problem_id] # set game_type here!
                print(f"Playing {problem_id}: {problem}")
                domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
                grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
                GAME_LOGIC = {
                        "pddl_domain": open(domain).read(),
                        "grammar": open(grammar).read(),
                    }
                pddl_file = os.path.join(problem, 'initial_state.pddl')
                json_file = os.path.join(problem, 'traj_data.json')
                with open(json_file, 'r') as f:
                    traj_data = json.load(f)
                GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)
                gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
                gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
                json.dump(gamedata, open(gamefile, "w"))
                expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

                request_infos = textworld.EnvInfos(
                    won=True,
                    admissible_commands=True,
                    score=True,
                    max_score=True,
                    intermediate_reward=True,
                    extras=["expert_plan"]
                )

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
                valid_actions = sorted(set(valid_actions) - {'look', 'inventory', 'help'})

                with open(file_name, "a") as f:
                    f.write(f"Trial {trial} - {model_name}\n")
                    f.write(f"Initial Observation: {init_obs}\n")
                    f.write(f"Goal: {goal}\n")
                    f.write(f"Valid Actions: {valid_actions}\n")

                brief_obs = "Action: look around\n" + summarize_obs(init_obs) + "\n"
                overall_memory = brief_obs
                MAX_STEPS = 20
                all_actions = []
                successful_actions = []
                obs_queue = []

                for step_id in range(MAX_STEPS):
                    with open(file_name, "a") as f:
                        f.write(f"\n==== Step {step_id} ====\n")
                    trial_step_record = []
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""

                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a") as f:
                            f.write(f"\n---- Larger Loop No. {within_step_tries} ----\n")
                            f.write(f"successful_actions: {successful_actions}\n")

                        within_step_tries += 1

                        if within_step_tries > 1:
                            env_id = textworld.gym.register_game(
                                gamefile,
                                request_infos,
                                max_episode_steps=1000000,
                                wrappers=[AlfredDemangler(), expert]
                            )
                            env = textworld.gym.make(env_id)
                            obs, infos = env.reset()
                            goal = obs.split('\n')[-1]
                            valid_actions = infos["admissible_commands"]
                            for act in successful_actions:
                                obs, _, done, infos = env.step(act)

                        actions, prompt = llm_to_actions_baseline(
                            model_name,
                            brief_obs,
                            goal,
                            overall_memory,
                            large_loop_error_message
                        )

                        with open(file_name, "a") as f:
                            f.write(f"Prompt: {prompt}\n")
                            f.write(f"Generated Actions: {actions}\n")

                        if not actions:
                            break

                        action_queue = list(actions)
                        tem_action_queue = []
                        tem_memory = ""

                        for act in action_queue:
                            _ = action_queue.pop(0)
                            obs, reward, done, infos = env.step(act)
                            action_text = "Action: " + act + "\n"
                            obs_text = summarize_obs(obs) + "\n"
                            brief_obs = action_text + obs_text
                            tem_memory += brief_obs
                            all_actions.append(act)

                            with open(file_name, "a") as f:
                                f.write(f"> {act}\n{brief_obs}\n")
                                f.write(f"After action '{act}', valid actions: {infos['admissible_commands']}\n")

                            if infos["won"]:
                                succeed = True
                                action_passed = True
                                with open(file_name, "a") as f:
                                    f.write("Success! Task completed.\n")
                                break

                            taken_action = act
                            if "Nothing happens." in brief_obs:
                                large_loop_error_message = f"""In this step, you take the following actions and observations from those actions:
                                    {''.join(obs_queue)}"""
                                if "go to" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to go to a receptacle but nothing happens. 
                                    You may already been at this receptacle, in other words, you have already went to this place and do not need to go to this receptacle again.
                                    Otherwise, there is no the receptacle you are aiming to."""
                                    continue
                                elif "open" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to open a receptacle but nothing happens. 
                                    You should first go to this receptacle to open it. 
                                    But if you have already go to this receptacle and still seeing this error message, it means that this receptacle cannot be opened and you can directly see objects after you go to it. Do not try to open it!!"""
                                elif "take" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to take something from a receptacle.
                                    You should first go to this receptacle to take the object.
                                    But if you have already go to this receptacle and still seeing this error message, it means that this receptacle doesn't have this object.
                                    You should go to other receptacle to find your aim object. Remember do not assume you can take the object from the receptable but should always set the initial goal as finding that aim object."""
                                elif "move" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}.
                                    You want to move some object to a receptacle but failed. You should first find that object somewhere by going to an unvisited receptacle and open if necessary.
                                    Then pick up the aiming object so that you can go to your aim receptacle and put it there.
                                    """
                                elif "slice" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to slice an object with a sharp object.
                                    You should first pickup the sharp object (this should be the only object you pick up) then take the slice action directly without picking up the aim object!
                                    Don't forget to put the sharp object back to the receptacle after you finish slicing."""
                                elif "cool" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to cool an object with a fridge. 
                                    You need to find the object and pick it up from other receptacle. Then go to frige and cool the object directly. Notice: do not move the object to the fridge but cool directly!"""
                                elif ("fridge" in taken_action or "sinkbasin" in taken_action or "microwave" in taken_action) and ("move" in taken_action or "take" in taken_action): # pass this
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to move or take an object to or from a fridge. 
                                    You don't need to take this action! You should go to fridge receptacle, cool the object, go to another receptacle"""
                                    continue
                                elif "use" in taken_action:
                                    large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to use an object.
                                    You can only use a lamp to turn it on and look at or examine other objects. Note: to look at or examine other objects, you should first pick it up."""
                                break

                            if not action_queue:
                                action_passed = True
                                successful_actions.extend(tem_action_queue)
                                overall_memory += tem_memory
                        # if action_passed:
                        #     successful_actions.extend(actions)
                        #     overall_memory += tem_memory
                        #     break

                    trial_step_record.append(within_step_tries)
                    trial_record.append(trial_step_record)

                    if within_step_tries == 5 or succeed:
                        break

                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    model_type = 'baseline' # PDDL
                    goal_type = 'detailed'
                    data_row = [today, model_name, model_type, game_type, goal_type, trial, succeed, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    writer.writerow(data_row)

                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    log_message = (
                        f"[Baseline] Trial {trial} (Attempt {retry+1}) | "
                        f"Model: {model_name} | "
                        f"Failed: {str(e)}\n"
                    )
                    f.write(log_message)
                retry += 1

def run_baseline_alfworld_50(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", folder_name="08_031825_alfworld", result_name="alfworld_results"):
    trial = 0
    for game_type, game_lst in GAME_DICT.items():
        # game_lst_sep = game_lst*2
        game_lst_sep = game_lst[:4]
        for problem_id in game_lst_sep:
            trial += 1
            # if trial < 13 and model_name == "o3-mini-2025-01-31":
            #     continue
            retry = 0
            while retry < 2:  # allow up to 2 attempts per trial
                try:
                    succeed = False
                    today = date.today()
                    fixed_model_name = model_name.replace("/", "_")
                    folder_path = f"output/{folder_name}"
                    os.makedirs(folder_path, exist_ok=True)
                    file_name = f"{folder_path}/{today}_{fixed_model_name}_baseline_{game_type}_{trial}.txt"

                    if retry == 1 and os.path.exists(file_name):
                        open(file_name, 'w').close()  # empty file
                        print(f"[Trial {trial}] Retrying: cleared file and retrying...")
                    trial_record = []

                    # each trial reset environment ===================
                    problem = os.path.dirname(PROBLEMS[problem_id])
                    print(f"Playing {problem_id}: {problem}")
                    domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
                    grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
                    GAME_LOGIC = {
                            "pddl_domain": open(domain).read(),
                            "grammar": open(grammar).read(),
                        }
                    pddl_file = os.path.join(problem, 'initial_state.pddl')
                    json_file = os.path.join(problem, 'traj_data.json')
                    with open(json_file, 'r') as f:
                        traj_data = json.load(f)
                    GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)
                    gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
                    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
                    json.dump(gamedata, open(gamefile, "w"))
                    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

                    request_infos = textworld.EnvInfos(
                        won=True,
                        admissible_commands=True,
                        score=True,
                        max_score=True,
                        intermediate_reward=True,
                        extras=["expert_plan"]
                    )

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
                    valid_actions = sorted(set(valid_actions) - {'look', 'inventory', 'help'})

                    with open(file_name, "a") as f:
                        f.write(f"Trial {trial} - {model_name}\n")
                        f.write(f"Initial Observation: {init_obs}\n")
                        f.write(f"Goal: {goal}\n")
                        f.write(f"Valid Actions: {valid_actions}\n")

                    brief_obs = "Action: look around\n" + summarize_obs(init_obs) + "\n"
                    overall_memory = brief_obs
                    MAX_STEPS = 20
                    all_actions = []
                    successful_actions = []
                    obs_queue = []

                    for step_id in range(MAX_STEPS):
                        with open(file_name, "a") as f:
                            f.write(f"\n==== Step {step_id} ====\n")
                        trial_step_record = []
                        within_step_tries = 0
                        action_passed = False
                        large_loop_error_message = ""

                        while within_step_tries < 5 and not action_passed:
                            with open(file_name, "a") as f:
                                f.write(f"\n---- Larger Loop No. {within_step_tries} ----\n")
                                f.write(f"successful_actions: {successful_actions}\n")

                            within_step_tries += 1

                            if within_step_tries > 1:
                                env_id = textworld.gym.register_game(
                                    gamefile,
                                    request_infos,
                                    max_episode_steps=1000000,
                                    wrappers=[AlfredDemangler(), expert]
                                )
                                env = textworld.gym.make(env_id)
                                obs, infos = env.reset()
                                for act in successful_actions:
                                    obs, _, done, infos = env.step(act)

                            actions, prompt = llm_to_actions_baseline(
                                model_name,
                                brief_obs,
                                valid_actions,
                                overall_memory,
                                large_loop_error_message,
                                goal=goal
                            )

                            with open(file_name, "a") as f:
                                f.write(f"Prompt: {prompt}\n")
                                f.write(f"Generated Actions: {actions}\n")

                            if not actions:
                                break

                            action_queue = list(actions)
                            tem_action_queue = []
                            tem_memory = ""

                            for act in action_queue:
                                _ = action_queue.pop(0)
                                obs, reward, done, infos = env.step(act)
                                action_text = "Action: " + act + "\n"
                                obs_text = summarize_obs(obs) + "\n"
                                brief_obs = action_text + obs_text
                                tem_memory += brief_obs
                                all_actions.append(act)

                                with open(file_name, "a") as f:
                                    f.write(f"> {act}\n{brief_obs}\n")
                                    f.write(f"After action '{act}', valid actions: {infos['admissible_commands']}\n")

                                if infos["won"]:
                                    succeed = True
                                    action_passed = True
                                    with open(file_name, "a") as f:
                                        f.write("Success! Task completed.\n")
                                    break

                                taken_action = act
                                if "Nothing happens." in brief_obs:
                                    large_loop_error_message = f"""In this step, you take the following actions and observations from those actions:
                                        {''.join(obs_queue)}"""
                                    if "go to" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to go to a receptacle but nothing happens. 
                                        You may already been at this receptacle, in other words, you have already went to this place and do not need to go to this receptacle again.
                                        Otherwise, there is no the receptacle you are aiming to."""
                                        continue
                                    elif "open" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to open a receptacle but nothing happens. 
                                        You should first go to this receptacle to open it. 
                                        But if you have already go to this receptacle and still seeing this error message, it means that this receptacle cannot be opened and you can directly see objects after you go to it. Do not try to open it!!"""
                                    elif "take" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to take something from a receptacle.
                                        You should first go to this receptacle to take the object.
                                        But if you have already go to this receptacle and still seeing this error message, it means that this receptacle doesn't have this object.
                                        You should go to other receptacle to find your aim object. Remember do not assume you can take the object from the receptable but should always set the initial goal as finding that aim object."""
                                    elif "move" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}.
                                        You want to move some object to a receptacle but failed. You should first find that object somewhere by going to an unvisited receptacle and open if necessary.
                                        Then pick up the aiming object so that you can go to your aim receptacle and put it there.
                                        """
                                    elif "slice" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to slice an object with a sharp object.
                                        You should first pickup the sharp object (this should be the only object you pick up) then take the slice action directly without picking up the aim object!
                                        Don't forget to put the sharp object back to the receptacle after you finish slicing."""
                                    elif "cool" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to cool an object with a fridge. 
                                        You need to find the object and pick it up from other receptacle. Then go to frige and cool the object directly. Notice: do not move the object to the fridge but cool directly!"""
                                    elif ("fridge" in taken_action or "sinkbasin" in taken_action or "microwave" in taken_action) and ("move" in taken_action or "take" in taken_action): # pass this
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to move or take an object to or from a fridge. 
                                        You don't need to take this action! You should go to fridge receptacle, cool the object, go to another receptacle"""
                                        continue
                                    elif "use" in taken_action:
                                        large_loop_error_message += f"""This is the action you take and got something wrong: {taken_action}. You are trying to use an object.
                                        You can only use a lamp to turn it on and look at or examine other objects. Note: to look at or examine other objects, you should first pick it up."""
                                    break

                                if not action_queue:
                                    action_passed = True
                                    successful_actions.extend(tem_action_queue)
                                    overall_memory += tem_memory
                            # if action_passed:
                            #     successful_actions.extend(actions)
                            #     overall_memory += tem_memory
                            #     break

                        trial_step_record.append(within_step_tries)
                        trial_record.append(trial_step_record)

                        if within_step_tries == 5 or succeed:
                            break

                    with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        model_type = 'baseline' # PDDL
                        data_row = [today, model_name, model_type, game_type, "detailed", trial, succeed, len(trial_record)-1,trial_record[-1][-1], trial_record]
                        writer.writerow(data_row)

                    break

                except Exception as e:
                    error_log_path = f"output/{folder_name}/errors.txt"
                    with open(error_log_path, "a") as f:
                        # Add a prefix and more context
                        log_message = (
                            f"[Baseline 50] Trial {trial} (Attempt {retry+1}) | "
                            f"Game Type: {game_type} | Model: {model_name} | "
                            f"Failed: {str(e)}\n"
                        )
                        f.write(log_message)
                    retry += 1

i = 0
num_trials = 2
# folder_name = "AlfW_o4_mini_high"
folder_name = "yewon_alfworld_0812_test"
result_name = folder_name
model_id1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_id2 = "Qwen/Qwen3-32B"
model_id3 = "meta-llama/Llama-3.3-70B-Instruct"
## Run PlanGen models
# run_baseline_alfworld(model_id1, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_iterative_model(model_id1, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id1)

# run_baseline_alfworld(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id2)

# run_baseline_alfworld(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_iterative_model(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id3)

# run_baseline_alfworld("gpt-4o-2024-05-13", i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld("gpt-4.1-2025-04-14", i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld("o4-mini-2025-04-16", i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld("deepseek", i, i+num_trials, folder_name=folder_name, result_name=result_name)

# run_baseline_alfworld_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld_50("deepseek", folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld_50("gpt-4o-2024-05-13", folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld_50("gpt-4.1-2025-04-14", folder_name=folder_name, result_name=result_name)
# run_baseline_alfworld_50("o4-mini-2025-04-16", folder_name=folder_name, result_name=result_name)


## Run PDDLego+ models
# run_iterative_model(model_id1, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
run_iterative_model("gpt-4o-2024-05-13", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("gpt-4o-2024-05-13", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("gpt-4.1-2025-04-14", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("o4-mini-2025-04-16", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("deepseek", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model("gpt-4o-2024-05-13", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model("deepseek", 7, 10, folder_name=folder_name, result_name=result_name, goal_type="subgoal")

# run_iterative_model_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_50("gpt-4o-2024-05-13", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_50("deepseek", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_50("gpt-4.1-2025-04-14", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_50("o4-mini-2025-04-16", folder_name=folder_name, result_name=result_name, goal_type="detailed")


## Run PDDLego+ with multiple prompts
# run_iterative_model_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="without_hint")
# run_iterative_model_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="without_detailed_goal")
# run_iterative_model_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model_50("gpt-4o-2024-05-13", folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model_50("deepseek", folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model_50("gpt-4.1-2025-04-14", folder_name=folder_name, result_name=result_name, goal_type="subgoal")


## Run PDDLego+ with fixed domain file
# run_iterative_model_fixed_df("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_fixed_df("gpt-4.1-2025-04-14", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_fixed_df("deepseek", folder_name=folder_name, result_name=result_name, goal_type="detailed")
