import sys
# print(sys.path)
import time
from datetime import date
import csv
import json
import asyncio
import re
print(1)
## module for server
# from kani import Kani
# from kani.engines.huggingface import HuggingEngine

import subprocess
import requests
print(2)
import os
import json
import glob
import random
import argparse
from os.path import join as pjoin
print(3)
from textworld_express import TextWorldExpressEnv
import textworld
import textworld.gym
print(4)
from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType

from openai import OpenAI
print('finishing import')


# Solver set up
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
            print(f"Error encountered: {e}. Retrying in 5 seconds... (Attempt {retries+1}/{max_retries})")
            retries += 1
            time.sleep(5)

    raise RuntimeError("Max retries exceeded. Failed to get result from solver.")

def get_action_from_pddl(df, pf):
    # run_fast_downward(path_to_df, path_to_pf)
    result = run_solver(df, pf, "dual-bfws-ffparser")
    action = result['output']['plan']
    print(f"action from solver: {action}")
    err_2 = result['stderr'] + result['stdout']
    return map_actions(action), err_2


# LLM set up
close_source_model_lists = ['gpt-4o-2024-05-13','o3-mini-2025-01-31',"gpt-4.1-2025-04-14","o4-mini-2025-04-16"]
def run_llm_model(prompt, model_name):

    if model_name in close_source_model_lists: # closed source LLMs
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            # max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        response_content = response.choices[0].message.content

        if response_content.startswith("```json"):
            response_content = response_content.lstrip("```json").rstrip("```").strip()

        result = json.loads(response_content)
        # try:
        #     result = json.loads(response_content)
        # except json.JSONDecodeError:
        #     return None, None

        df = result.get("df", None)
        pf = result.get("pf", None)

        # if df is None or pf is None:
        #     raise ValueError("Missing 'df' or 'pf' in the response. Check the prompt or the model output.")

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

        if response_content.startswith("```json"):
            response_content = response_content.lstrip("```json").rstrip("```").strip()

        result = json.loads(response_content)
        df = result.get("df", None)
        pf = result.get("pf", None)

        # if df is None or pf is None:
        #     raise ValueError("Missing 'df' or 'pf' in the response. Check the prompt or the model output.")

        return df, pf
    
    else: # Open source LLMs
        """
        Run a prompt against a HuggingFace model using Kani and parse out
        'df' and 'pf' from the JSON response. Raises ValueError if missing keys.
        """
        async def _ask_model(model_name, user_prompt):
            # Create Hugging Face engine
            engine = HuggingEngine(
                model_id=model_name,
                use_auth_token=True, 
                model_load_kwargs={
                    "device_map": "auto",
                    "trust_remote_code": True
                }
            )
            # Wrap in Kani
            ai = Kani(engine, system_prompt="")

            # Send the user prompt and get the response string
            response = await ai.chat_round_str(user_prompt)
            return response

        # Because Kani calls are async, we need to run them in an event loop
        response_content = asyncio.run(_ask_model(model_name, prompt))

        if '</think>' in response_content:
            response_content = response_content[response_content.find('</think>')+10:]

        if response_content.startswith("```json"):
            response_content = (
                response_content
                .lstrip("```json")
                .rstrip("```")
                .strip()
            )
        
        def extract_json_block(text):
            # This pattern captures everything between ```json and the next ```
            # (?s) makes '.' match newlines as well
            pattern = r"(?s)```json\s*(.*?)\s*```"
            
            match = re.search(pattern, text)
            if match:
                # match.group(1) contains the content between the backticks
                return match.group(1).strip()
            return text
        # print(response_content)
        response_content = extract_json_block(response_content)

        # Attempt to parse the JSON response
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            raise ValueError(
                f"Model response is not valid JSON:\n{response_content}"
            )

        # Extract fields
        df = result.get("df")
        pf = result.get("pf")

        if df is None or pf is None:
            raise ValueError(
                "Missing 'df' or 'pf' in the response. Check your prompt or the model's output."
            )

        return df, pf

# Set up baseline model: get actions directly from model
def run_gpt_for_actions_baseline(prompt, model_name):
    if model_name in close_source_model_lists: # closed source LLMs
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            # max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        response_content = response.choices[0].message.content

        if response_content.startswith("```json"):
            response_content = response_content.lstrip("```json").rstrip("```").strip()

        def extract_json_block(text):
            # This pattern captures everything between ```json and the next ```
            # (?s) makes '.' match newlines as well
            pattern = r"(?s)```json\s*(.*?)\s*```"
            
            match = re.search(pattern, text)
            if match:
                # match.group(1) contains the content between the backticks
                return match.group(1).strip()
            return text

        response_content = extract_json_block(response_content)

        result = json.loads(response_content)

        actions = result.get("actions", None)

        if actions is None:
            raise ValueError("Missing 'actions' in the response. Check the prompt or the model output.")

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

        if response_content.startswith("```json"):
            response_content = response_content.lstrip("```json").rstrip("```").strip()

        def extract_json_block(text):
            # This pattern captures everything between ```json and the next ```
            # (?s) makes '.' match newlines as well
            pattern = r"(?s)```json\s*(.*?)\s*```"
            
            match = re.search(pattern, text)
            if match:
                # match.group(1) contains the content between the backticks
                return match.group(1).strip()
            return text

        response_content = extract_json_block(response_content)
        result = json.loads(response_content)
        actions = result.get("actions", None)
        if actions is None:
            raise ValueError("Missing 'actions' in the response. Check the prompt or the model output.")
        return actions
        
    else: # Open source LLMs
        """
        Run a prompt against a HuggingFace model using Kani and parse out
        'df' and 'pf' from the JSON response. Raises ValueError if missing keys.
        """
        async def _ask_model(model_name, user_prompt):
            # Create Hugging Face engine
            engine = HuggingEngine(
                model_id=model_name,
                use_auth_token=True, 
                model_load_kwargs={
                    "device_map": "auto",
                    "trust_remote_code": True
                }
            )
            # Wrap in Kani
            ai = Kani(engine, system_prompt="")

            # Send the user prompt and get the response string
            response = await ai.chat_round_str(user_prompt)
            return response

        # Because Kani calls are async, we need to run them in an event loop
        response_content = asyncio.run(_ask_model(model_name, prompt))

        # deepseek-ai/DeepSeek-R1-Distill-Llama-70B
        # print(response_content)
        if '</think>' in response_content:
            response_content = response_content[response_content.find('</think>')+10:]

        if response_content.startswith("```json"):
            response_content = (
                response_content
                .lstrip("```json")
                .rstrip("```")
                .strip()
            )
        
        def extract_json_block(text):
            # This pattern captures everything between ```json and the next ```
            # (?s) makes '.' match newlines as well
            pattern = r"(?s)```json\s*(.*?)\s*```"
            
            match = re.search(pattern, text)
            if match:
                # match.group(1) contains the content between the backticks
                return match.group(1).strip()
            return text

        response_content = extract_json_block(response_content)

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            raise ValueError(
                f"Model response is not valid JSON:\n{response_content}"
            )

        actions = result.get("actions", None)

        if actions is None:
            raise ValueError("Missing 'actions' in the response. Check the prompt or the model output.")

        return actions



# VAL setup
# common_path = "/Users/krystalgong/Documents/GitHub/pddlego-df/"

def file_to_path(domain_content, problem_content, domain_filename="domain.pddl", problem_filename="problem.pddl"):
    with open(domain_filename, 'w') as domain_file:
        domain_file.write(domain_content)
    
    with open(problem_filename, 'w') as problem_file:
        problem_file.write(problem_content)

    path_to_df = domain_filename
    path_to_pf = problem_filename

    return path_to_df, path_to_pf

def plan_to_path(plan, plan_filename="plan.txt"):
    with open(plan_filename, 'w') as plan_file:
        plan_file.write(plan)

    path_to_plan = "plan.txt"

    return path_to_plan

def run_pddl_parser(domain_file, problem_file=None):
    # Define the path to your Parser executable
    parser_path = "VAL-master/build/macos64/Release/bin/Parser"
    domain_path, problem_path = file_to_path(domain_file, problem_file)
    
    # Check if both domain and problem files are provided
    if problem_file:
        command = [parser_path, domain_path, problem_path]
    else:
        command = [parser_path, domain_path]
    
    try:
        # Run the Parser and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if there is any error
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        # Return the stdout (output) of the parser
        return result.stdout
    
    except FileNotFoundError as e:
        print(f"Parser not found: {e}")
        return None

def validate_pddl(domain_file, problem_file, plan=None):
    # The path to the Validate executable
    validate_executable = "VAL-master/build/macos64/Release/bin/Validate"

    domain_path, problem_path = file_to_path(domain_file, problem_file)
    plan_path = plan_to_path(plan)
    
    # Construct the command
    command = [validate_executable, "-v", domain_path, problem_path]

    # plan should be a txt file
    if plan_path:
      command.append(plan_path)
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        print("Validation Output:\n", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("Error:\n", e.stderr)

# Currently ignoring err_1, i.e. not using this function. Something wrong with VAL...
def error_message(domain_file, problem_file):
    # Run Parser and get error message
    pass
    parser_output = run_pddl_parser(domain_file, problem_file)
    err_message = ''
    if parser_output:
        output_lst = parser_output.split('\n')
        # err_message = ''
        error = False
        for i in output_lst:
            if i.startswith('Errors:') or error:
                if "Warning" in i:
                    continue
                error = True
                err_message += i
                err_message += '\n'
                
    # err_message = err_message.replace('/Users/krystalgong/Documents/GitHub/pddlego-df/', '')
    return err_message


# Additional functions
def summarize_obs(obs):
    # If obs only has one line, return it
    if len(obs.split('\n')) == 1:
        return obs
    # Only keep where you are and location informtion
    else:
        return obs.split('\n')[0].split(". ")[0] + ". " + obs.split('\n')[1]

### === keep modifying this === ###
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

# Edit mode but NOT USING it now...
def apply_edit_domain(prev_df, edit_json):
    output = []
    predicate_section = False
    action_name = None
    
    for line in prev_df.split("\n"):
        stripped_line = line.strip()
        
        # Handle predicates
        if "(:predicates" in line:
            predicate_section = True
            output.append(line)
        elif predicate_section and stripped_line == ")":
            predicate_section = False
            # Add new predicates if specified
            if "predicates" in edit_json and "add" in edit_json["predicates"]:
                for pred in edit_json["predicates"]["add"]:
                    output.append("    " + pred)
            output.append(line)
        elif predicate_section:
            if "predicates" in edit_json:
                if "replace" in edit_json["predicates"] and stripped_line in edit_json["predicates"]["replace"]:
                    output.append("    " + edit_json["predicates"]["replace"][stripped_line])
                elif "delete" in edit_json["predicates"] and stripped_line in edit_json["predicates"]["delete"]:
                    continue
                else:
                    output.append(line)
            else:
                output.append(line)
        
        # Handle actions
        elif "(:action" in line:
            action_name = stripped_line.split()[1]
            output.append(line)
        elif action_name and ":precondition" in stripped_line:
            if "action" in edit_json and action_name in edit_json["action"] and "precondition" in edit_json["action"][action_name]:
                # Replace precondition
                output.append("        :precondition " + " ".join(edit_json["action"][action_name]["precondition"]))
                while ")" not in stripped_line:  # Skip lines until end of precondition
                    stripped_line = next(prev_df.split("\n")).strip()
            else:
                output.append(line)
        elif action_name and ":effect" in stripped_line:
            if "action" in edit_json and action_name in edit_json["action"] and "effect" in edit_json["action"][action_name]:
                # Replace effect
                output.append("        :effect " + " ".join(edit_json["action"][action_name]["effect"]))
                while ")" not in stripped_line:  # Skip lines until end of effect
                    stripped_line = next(prev_df.split("\n")).strip()
            else:
                output.append(line)
        else:
            output.append(line)
    
    return "\n".join(output)

def apply_edit_problem(prev_pf, edit_json):
    output = []
    obj_section = False
    init_section = False
    goal_section = False
    
    for line in prev_pf.split("\n"):
        stripped_line = line.strip()
        
        # Handle objects
        if "(:objects" in line:
            obj_section = True
            output.append(line)
        elif obj_section and stripped_line == ")":
            obj_section = False
            # Add new objects if specified
            if "objects" in edit_json and "add" in edit_json["objects"]:
                for obj in edit_json["objects"]["add"]:
                    output.append("        " + obj)
            output.append(line)
        elif obj_section:
            if "objects" in edit_json:
                if "replace" in edit_json["objects"] and stripped_line in edit_json["objects"]["replace"]:
                    output.append("        " + edit_json["objects"]["replace"][stripped_line])
                elif "delete" in edit_json["objects"] and stripped_line in edit_json["objects"]["delete"]:
                    continue
                else:
                    output.append(line)
            else:
                output.append(line)
        
        # Handle init
        elif "(:init" in line:
            init_section = True
            output.append(line)
        elif init_section and stripped_line == ")":
            init_section = False
            # Add new init statements if specified
            if "init" in edit_json and "add" in edit_json["init"]:
                for init in edit_json["init"]["add"]:
                    output.append("        " + init)
            output.append(line)
        elif init_section:
            if "init" in edit_json:
                if "replace" in edit_json["init"] and stripped_line in edit_json["init"]["replace"]:
                    output.append("        " + edit_json["init"]["replace"][stripped_line])
                elif "delete" in edit_json["init"] and stripped_line in edit_json["init"]["delete"]:
                    continue
                else:
                    output.append(line)
            else:
                output.append(line)
        
        # Handle goal
        elif "(:goal" in line:
            goal_section = True
            output.append(line)
        elif goal_section:
            goal_section = False
            if "goal" in edit_json:
                output.append("        " + " ".join(edit_json["goal"]))
                while ")" not in stripped_line:  # Skip lines until end of goal
                    stripped_line = next(prev_pf.split("\n")).strip()
            else:
                output.append(line)
        
        else:
            output.append(line)
    
    return "\n".join(output)

def apply_edit(prev_df, prev_pf, edit_json_df, edit_json_pf):
    update_df, update_pf = prev_df, prev_pf  # Default to original if no changes are made
    if edit_json_df == {}:
        update_df = apply_edit_domain(prev_df, edit_json_df)
    if edit_json_pf == {}:
        update_pf = apply_edit_problem(prev_pf, edit_json_pf)
    return update_df, update_pf

def llm_to_pddl_check_delta(obs, taken_action, prev_df="", prev_pf=""):

    prompt_edit = """
        Please provide the edit output in JSON format, including the edit suggestions for a domain file as 'df' and the edit suggestions for a problem file as 'pf'. 
        The output format should be: {{"df": "...", "pf": "..."}}
        You will modify the following df and pf using add, delete, and replace operations (in a JSON format). 
        You SHOULD NOT provide a domain file and a problem file directly.
        If you think the current observation is correct with your previous generated files, then provide empty JSON: 
        {{"df": "{}", "pf": "{}"}}
        This is the structure for df edit file if you think this observation is different from previous generated version, remember to add bracket:
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

    prompt_obs_action = f"""
        Background: You are in an environment that you explore step by step. You must build and update PDDL files of the environment based on only your observations. 
        Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
        Your task is always to keep exploration and go to a location you have not visited yet.

        Here is your last action {taken_action} and the observation after taking that action: {summarize_obs(obs)}
    """ 

    prompt_prev_files = f"""
        This is previous domain file: {prev_df}
        This is previous problem file: {prev_pf}
    """
        
    prompt = prompt_edit + prompt_obs_action + prompt_prev_files

    if "I'm not sure what you mean." in summarize_obs(obs) and "open door" in taken_action:
        print('\n There is no door here or there is nothing in this direction.') # how to utilize this? previous obs. how to extract locations
        prompt += 'Additionally notes: You are trying to open a door but there is no door here or there is nothing in this direction.'
    elif "open door" in taken_action:
        prompt += "\n Additionally notes: You opened a door and revealing the above place. \
            Is this what you are expecting based on your previous generated problem file? \
            If yes, you should generate the empty edit json file! \
            If not, do you need to edit the previous file, mainly problem file? Provide the edit json! Thank you!"

    edit_json_df, edit_json_pf = run_llm_model(prompt) # , model_name

    print(edit_json_df, edit_json_pf)

    edit_json_df = json.loads(edit_json_df)
    edit_json_pf = json.loads(edit_json_pf)
    
    zero_edit = {
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
        "goal": []
        }
    if edit_json_pf == zero_edit or edit_json_pf == {}:
        return True, 0, 0
    
    # print("Edit json:",edit_json_df, edit_json_pf)
    print(edit_json_df, edit_json_pf, type(edit_json_pf), edit_json_pf=={})
    df, pf = apply_edit(prev_df, prev_pf, edit_json_df, edit_json_pf)

    return False, df, pf


### ========= Alfworld =========
# choose a problem to solve
problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)

problems = [p for p in problems if "movable_recep" not in p]
if len(problems) == 0:
    raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
problem_id = 5
problem = os.path.dirname(problems[problem_id])
problem_type_dic = {0: 'clean', 1: 'basic', 2: 'basic', 3:'slice & heat', 4: 'heat',\
     5:'use', 6:'clean', 7: 'use', 8: 'basic', 9:'cool'}
game_type = problem_type_dic[problem_id] # set game_type here!

game_dictionary = {
    "basic&use": [1,8,23,21,30,5,7,27,20,22],
    "cool": [9,13,18,25,42,44,46,50,58,68],
    "heat": [4,60,69,71,93,104,119,151,153,154],
    "clean": [0,6,10,37,38,52,61,79,80,83],
    "slice+": [29,39,56,3,17,34,49,73,19,64]
}

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
brief_obs = "Action: look around\n" + summarize_obs(init_obs)+'\n'
print("Initial observation:", init_obs)
print(goal)
print("Valid actions:", valid_actions)


def llm_to_pddl(model_name, brief_obs, prev_df="", prev_pf="", prev_err="", prev_err_2=None, have_error=False, have_duplicate=False, edit=False, overall_memory=None, large_loop_error_message=None, goal_type='detailed', goal=goal):
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

        In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.
        
        Note: 
        1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
        2. Your initial goal should always be to go to a new location instead of put something into somewhere.
        3. Do not enter stage 2 when not finishing stage 1.

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


    if edit:
        edit_json_df, edit_json_pf = run_llm_model(prompt, model_name)
        # print("Edit json:",edit_json_df, edit_json_pf)
        df, pf = apply_edit(prev_df, prev_pf, edit_json_df, edit_json_pf)
        # print("New df and pf:", df, pf)
    else:
        df, pf = run_llm_model(prompt, model_name)

    err = None #error_message(df, pf)
    return df, pf, err, prompt


def llm_to_actions_baseline(model_name, brief_obs, valid_actions, overall_memory=None, large_loop_error_message=None, goal_type="detailed", goal=goal):
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
    if goal_type == 'detailed':
        prompt = prompt_detailed
    # elif goal_type == 'subgoal':
    #     prompt = prompt_subgoal
    # else:
    #     prompt = prompt_general

    actions = run_gpt_for_actions_baseline(prompt, model_name)
    return actions, prompt



# ===== Main functions here =====
def run_iterative_model(model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", start_trial = 0, end_trial = 11, folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed"):
    # trial_record = 
    # structured_info_record = "output/summary"
    for trial in range(start_trial, end_trial):
        retry = 0
        while retry < 2:  # allow up to 2 attempts per trial
            try:
                succeed = False
                today = "2025-04-19" #date.today()
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
                problem = os.path.dirname(problems[problem_id])
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
                    f.write(f"Trial {trial} (Attempt {retry+1}) model ({model_name}) failed: {str(e)}\n")
                retry += 1

def run_iterative_model_50(model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed", trials_to_run=None):
    # trial_record = 
    # structured_info_record = "output/summary"
    # for trial in range(start_trial, end_trial):
    # if model_name in ["o3-mini-2025-01-31", "deepseek"]:
    #     trial = 50
    # else:
    trial = 0
    for game_type, game_lst in game_dictionary.items():
        # if model_name in ["o3-mini-2025-01-31", "deepseek"]:
        #     game_lst_sep = game_lst
        # else:
        game_lst_sep = game_lst*2
        for problem_id in game_lst_sep: # extra indent
            trial += 1

            if trials_to_run and trial not in trials_to_run: # skip trials not in the list
                continue

            retry = 0
            while retry < 2:  # allow up to 2 attempts per trial
                try:
                    succeed = False
                    today = "2025-05-04" #date.today()
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
                    problem = os.path.dirname(problems[problem_id])
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
                        f.write(f"Trial {trial} (Attempt {retry+1}) model ({model_name}) failed: {str(e)}\n")
                    retry += 1

def run_baseline_alfworld(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", start_trial=0, end_trial=5, folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed"):
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
                problem = os.path.dirname(problems[problem_id])
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
                            for act in successful_actions:
                                obs, _, done, infos = env.step(act)

                        actions, prompt = llm_to_actions_baseline(
                            model_name,
                            brief_obs,
                            valid_actions,
                            overall_memory,
                            large_loop_error_message,
                            goal_type=goal_type,
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
                    data_row = [today, model_name, model_type, game_type, goal_type, trial, succeed, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    writer.writerow(data_row)

                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    f.write(f"Trial {trial} (Attempt {retry+1}) failed: {str(e)}\n")
                retry += 1

def run_baseline_alfworld_50(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed"):
    if model_name in ["o3-mini-2025-01-31", "deepseek"]:
        trial = 50
    else:
        trial = 0
    for game_type, game_lst in game_dictionary.items():
        if model_name in ["o3-mini-2025-01-31", "deepseek"]:
            game_lst_sep = game_lst
        else:
            game_lst_sep = game_lst*2
        for problem_id in game_lst_sep:
        # for problem_id in game_lst: # extra indent
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
                    problem = os.path.dirname(problems[problem_id])
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
                                goal_type=goal_type,
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
                        data_row = [today, model_name, model_type, game_type, goal_type, trial, succeed, len(trial_record)-1,trial_record[-1][-1], trial_record]
                        writer.writerow(data_row)

                    break

                except Exception as e:
                    error_log_path = f"output/{folder_name}/errors.txt"
                    with open(error_log_path, "a") as f:
                        f.write(f"Trial {trial} (Attempt {retry+1}) failed: {str(e)}\n")
                    retry += 1




i = 0
num_trials = 10
folder_name = "7_0503_Alfworld_tem_subgoal"
result_name = folder_name

## Run baseline models
# run_baseline_alfworld("gpt-4o-2024-05-13", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_baseline_alfworld("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_baseline_alfworld("gpt-4.1-2025-04-14", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_baseline_alfworld("o4-mini-2025-04-16", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_baseline_alfworld("deepseek", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")


## Run PDDL generation models
# run_iterative_model("gpt-4o-2024-05-13", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("gpt-4.1-2025-04-14", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("o4-mini-2025-04-16", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("deepseek", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")

# run_iterative_model_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_50("gpt-4o-2024-05-13", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model_50("deepseek", folder_name=folder_name, result_name=result_name, goal_type="detailed")

run_iterative_model_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="subgoal")
run_iterative_model_50("gpt-4o-2024-05-13", folder_name=folder_name, result_name=result_name, goal_type="subgoal")
run_iterative_model_50("deepseek", folder_name=folder_name, result_name=result_name, goal_type="subgoal")

# run_baseline_alfworld_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_baseline_alfworld_50("deepseek", folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_baseline_alfworld_50("gpt-4o-2024-05-13", folder_name=folder_name, result_name=result_name, goal_type="detailed")


# run_iterative_model("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model("gpt-4o-2024-05-13", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model("deepseek", 7, 10, folder_name=folder_name, result_name=result_name, goal_type="subgoal")