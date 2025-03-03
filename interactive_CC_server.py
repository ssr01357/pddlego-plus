import time
from datetime import date
import csv
import json
import asyncio
import re

from kani import Kani
from kani.engines.huggingface import HuggingEngine

import subprocess
import requests

from textworld_express import TextWorldExpressEnv

from openai import OpenAI
import os

client = OpenAI()
# os.environ["OPENAI_API_KEY"] = ""

# Solver set up
def run_solver(domain_file, problem_file, solver):
    # domain_file = open(f'domain.pddl').read()
    # problem_file = open(f'problem.pddl').read()

    req_body = {"domain" : domain_file, "problem" : problem_file}

    # Send job request to solve endpoint
    solve_request_url=requests.post(f"https://solver.planning.domains:5001/package/{solver}/solve", json=req_body).json()

    # Query the result in the job
    celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

    while celery_result.json().get("status","")== 'PENDING':
        # Query the result every 0.5 seconds while the job is executing
        celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])
        time.sleep(0.5)

    result = celery_result.json()['result']
    return result

def get_action_from_pddl(df, pf):
    # run_fast_downward(path_to_df, path_to_pf)
    result = run_solver(df, pf, "dual-bfws-ffparser")
    action = result['output']['plan']
    err_2 = result['stderr']
    return map_actions(action), err_2

# LLM set up
close_source_model_lists = ['o3-mini', 'gpt-4o', 'gpt-4o-mini-2024-07-18', 'o3-mini-2025-01-31']
def run_llm_model(prompt, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"):

    if model_name in close_source_model_lists: # closed source LLMs
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
        # print(response_content)
        response_content = extract_json_block(response_content)
        # print(response_content)

        # # If your model returns a JSON block wrapped in triple backticks, strip them
        

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

def llm_to_actions_baseline(model_name, brief_obs, valid_actions, overall_memory=None, large_loop_error_message=None):
    prompt = f"""
        You are in an environment that you explore step by step. Based on your observations, generate a series of valid actions to progress in the environment.
        Here are your current observations: {brief_obs}
        Here are some valid actions you can take: {valid_actions}
        Your goal is to explore new locations and interact with the environment effectively. Ensure actions are logical and do not repeat unnecessarily.

        Additional context:
        {overall_memory if overall_memory else "No additional memory available."}

        If there are errors or obstacles, here is the message:
        {large_loop_error_message if large_loop_error_message else "No errors or obstacles mentioned."}

        Provide the output in strict JSON format like this:
        {{
            "actions": ["action1", "action2", ...]
        }}
    """
    actions = run_gpt_for_actions_baseline(prompt, model_name)
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

def map_actions(action):
    actions = action.lower().replace("(", "").replace(")", "").split('\n')
    action_lst = []
    for act in actions:
        if "open" in act and "door" in act:
            direction = act.split(' ')[-1]
            action_lst.append(f'open door to {direction}')
        elif "move" in act:
            action_lst.append(f"move {act.split(' ')[-1]}")
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




# Prompt modify and main function
env = TextWorldExpressEnv(envStepLimit=100)
NUM_LOCATIONS = 11
env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
valid_actions = sorted(infos['validActions'])
valid_actions.remove('look around')
valid_actions.remove('inventory')
# brief_obs = "Action: look around\n" + summarize_obs(obs)+'\n'

def llm_to_pddl(model_name, brief_obs, prev_df="", prev_pf="", prev_err="", prev_err_2=None, have_error=False, have_duplicate=False, edit=False, overall_memory=None, large_loop_error_message = None):
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

    prompt_obs_action = f"""
        You are in an environment that you explore step by step. You must build and update PDDL files of the environment based on only your observations. 
        Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
        Do not assume that there will be a door connecting rooms.
        Your task is always to keep exploration and go to a location you have not visited yet.
        In other words, your goal should go to other not visited location.
        If you enter a room, make sure you put everything you observed such as the direction in the problem file.
        Here are your current observations: {brief_obs}
        Here are some valid actions you can take: {valid_actions}
        You should generate df and pf strictly follow this valid actions. There are in total 2 actions, that should exactly be the following two:
        1. :action open-door
            :parameters (?loc1 - location ?loc2 - location ?dir - direction)
        2. :action move
            :parameters (?from - location ?to - location ?dir - direction)
        You should have a goal in the problem file like this: 
        (:goal 
            (at ?location)
        ) where location should be somewhere not visited
        Note: in problem file's init, you shouldn't have "not ()" but only the single status
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
    prompt += prompt_obs_action

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
    # check err and its df & pf here:
    # ....
    return df, pf, err, prompt

# ==== Merging method helper functions ====
# Merging method: Without previous problem file but only feeding observations
def generate_problem_file_from_observation(observation, model_name="o3-mini", domain_file="", err="", err_2=""):
    # prompt = f"""
    #     You are in an environment that you explore step by step. You must build and update PDDL files of the environment based on only your observations. 
    #     Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
    #     Do not assume that there will be a door connecting rooms.
    #     Your task is always to keep exploration and go to a location you have not visited yet.
    #     In other words, your goal should go to other not visited location.
    #     If you enter a room, make sure you put everything you observed such as the direction in the problem file.
        
    #     Here are some valid actions you can take: {valid_actions}
    #     You should generate df and pf strictly follow this valid actions. There are in total 2 actions, that should exactly be the following two:
    #     1. :action open-door
    #         :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    #     2. :action move
    #         :parameters (?from - location ?to - location ?dir - direction)
    #     You should have a goal in the problem file like this: 
    #     (:goal 
    #         (at ?location)
    #     ) where location should be somewhere not visited
    #     Note: in problem file's init, you shouldn't have "not ()" but only the single status
        

    #     The domain file is here: {domain_file}
    #     If it is empty, remember to build a PDDL domain file from scratch first. 
    #     If you have already built one domain file, i.e. it is not empty, you need to check and revise according to your understanding or the following error messages: {err} and {err_2}.

    #     Now, you are building a PDDL problem file from scratch, based ONLY on this observation:
    #     {observation}
    #     You do not have access to any previous problem file.

    #     Your goal:
    #     - Include only the relevant objects, initial states, and goals that reflect this single observation.
    #     - The problem file must be self-contained (i.e., must have (:objects ...), (:init ...), and (:goal ...)).
    #     - Use the domain's recognized actions or objects if needed, but do not assume external info not in the observation.

    #     Output strictly in JSON:
    #     {{
    #         "df": "CONTENT_OF_THE_DOMAIN_FILE",
    #         "pf": "CONTENT_OF_THE_PROBLEM_FILE"
    #     }}
    # """
    prompt = f"""You are in an environment that you explore step by step. Your job is to build and update the PDDL files (both the domain file and the problem file) strictly based on your observations—do not add information that has not been observed. In particular:
        - Do not invent objects or rooms that are not mentioned in the observations.
        - If you see a closed door, assume there may be a room behind it—but only include it if supported by the observation.
        - Do not assume a door connects two rooms unless it is clearly observed.
        - Your overall goal is to keep exploring and reach a location that has not yet been visited.

        You have the following valid actions available (and you must follow these exactly):
        1. **:action open-door**  
        :parameters (?loc1 - location ?loc2 - location ?dir - direction)
        2. **:action move**  
        :parameters (?from - location ?to - location ?dir - direction)

        Additionally, your problem file must include a goal section exactly in this format:
        (:goal (at ?location) )

        where “?location” is replaced with a location that has not been visited.

        The current domain file is provided here: {domain_file}
        - If this domain file is empty, first build a PDDL domain file from scratch.
        - If the domain file is not empty, check and, if necessary, revise it according to your understanding or the following error messages: {err} and {err_2}.

        Now, you are tasked with building a PDDL **problem file** from scratch based solely on the following observation:
        {observation}
        *Note: You do not have access to any previous problem file.*

        Your goals for generating the problem file are:
        - Include only the relevant objects, initial states, and goals that directly reflect this observation.
        - Ensure the problem file is self-contained (i.e., it must include the sections (:objects ...), (:init ...), and (:goal ...)).
        - Use the objects or actions defined in the domain file if necessary, but do not assume any external information.

        Output strictly in JSON format with the following structure:
        {{
            "df": "CONTENT_OF_THE_DOMAIN_FILE",
            "pf": "CONTENT_OF_THE_PROBLEM_FILE"
        }}
    """
    df, pf = run_llm_model(prompt, model_name=model_name)
    return df, pf, None, prompt

def merge_problem_files_llm(old_problem_file, new_problem_file, model_name="o3-mini"):
    prompt = f"""
        You have two problem files (PDDL). The old problem file:
        <<<
        {old_problem_file}
        >>>

        The new problem file based on the new observation:
        <<<
        {new_problem_file}
        >>>

        Merge them so that:
        - You include any new objects or init facts from the new PF if they are truly new.
        - You do not lose any critical old PF objects or init facts that are still valid.
        - The final goal should reflect both ensuring the new place is discovered or consistent with your exploration aim.

        Output strictly in JSON:
        {{
        "pf": "YOUR_FINAL_MERGED_PROBLEM_FILE"
        }}
    """

    _df, merged_pf = run_llm_model(prompt, model_name=model_name)
    return merged_pf

def merge_problem_files_code(old_pf: str, new_pf: str) -> str:
    # 1. Extract objects, init, goal from old_pf
    old_objects = []
    old_init = []
    old_goal = []
    # (Write your parser code for old_pf below or call an existing parser.)

    # 2. Extract objects, init, goal from new_pf
    new_objects = []
    new_init = []
    new_goal = []
    # (Write your parser code for new_pf below.)

    # 3. Merge objects
    # For example, unify sets:
    final_objects_set = set(old_objects) | set(new_objects)

    # 4. Merge init
    # For instance, unify sets again. Remove duplicates.
    final_init_set = set(old_init) | set(new_init)

    # 5. Choose or unify the goal
    # If your approach is "always last known goal," or "union of old + new," do it here.
    # For example, choose new PF's goal if you want the "latest location" to be the goal:
    final_goal = new_goal if new_goal else old_goal

    # 6. Rebuild the PF with the merged sections:
    final_pf = f"""
        (define (problem MergedProblem)
        (:domain MyDomain)

        (:objects
            {' '.join(sorted(list(final_objects_set)))}
        )

        (:init
            {"\n    ".join(sorted(list(final_init_set)))}
        )

        (:goal
            {" ".join(final_goal)}
        )
        )
    """

    # Return the merged PF string
    return final_pf
# ==== Merging method helper functions (end) ====



def run_iterative_model(model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", start_trial = 0, end_trial = 11):
    # trial_record = 
    # structured_info_record = "output/summary"
    for trial in range(start_trial, end_trial):
        coin_found = False
        today = date.today()
        file_name = f"output/05_020625_100trials/{today}_{model_name.replace("/","_")}_{trial}.txt"
        trial_record = []
        
        env = TextWorldExpressEnv(envStepLimit=100)
        NUM_LOCATIONS = 11
        env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
        obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
        with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
            f.write(f"Observations: {obs} \n") 
            f.write(f"Gold path: {env.getGoldActionSequence()} \n")
            f.write(f"Valid Actions: {infos['validActions']} \n")
            f.write(f"taskDescription: {infos['taskDescription']} \n")

        # task_description = infos['taskDescription']
        valid_actions = sorted(infos['validActions'])
        valid_actions.remove('look around')
        valid_actions.remove('inventory')

        MAX_STEPS = 20

        brief_obs = "Action: look around\n" + summarize_obs(obs)+'\n' # initial definition
        with open(file_name, "a") as f:
            f.write(f"brief_obs: {brief_obs} \n") 
        # print(brief_obs)

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

                if within_step_tries > 1: # second or third ... time in the larger loop
                    # reset env by refilling successful actions (stupid but useful)
                    env = TextWorldExpressEnv(envStepLimit=100)
                    NUM_LOCATIONS = 11
                    env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
                    obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
                    for successful_action in successful_actions:
                        obs, reward, done, infos = env.step(successful_action)

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
                            df, pf, err, prompt = llm_to_pddl(model_name, brief_obs) # error 1 here
                            action, err_2 = get_action_from_pddl(df, pf) # error 2 here
                            with open(file_name, "a") as f:
                                f.write(f"--Small Loop--: {num_tries} \n")
                                f.write(f"Error: {err} \n")
                                f.write(f"Prompt: {prompt} \n") 
                                f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                f.write(f"Actions from solver(df, pf): {action} \n")

                            while not action and num_tries < 5:
                                df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, df, pf, err, err_2, True, False, edit)
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
                            df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, df, pf, err, None, False, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message) # need to add new error message
                            action, err_2 = get_action_from_pddl(df, pf)

                            with open(file_name, "a") as f:
                                f.write(f"--Small Loop--: {num_tries} \n")
                                f.write(f"Error: {err} \n")
                                f.write(f"Prompt: {prompt} \n") 
                                f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                f.write(f"Actions from solver(df, pf): {action} \n")

                            while not action and num_tries < 5:
                                df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, df, pf, err, err_2, True, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message)
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
                    # Feedback from plan-environment interaction
                    # err_validate = validate_pddl(df, pf, taken_action)
                    # print(err_validate)

                    obs, reward, done, infos = env.step(taken_action)

                    # Directly end the game if found coin
                    if "coin" in obs:
                        taken_action = "take coin"
                        obs, reward, done, infos = env.step(taken_action)
                        end_game = True
                        with open(file_name, "a") as f:
                            f.write('Coin found!')
                            coin_found = True
                        break
                    
                    action_text = "Action: " + taken_action + "\n"
                    obs_text = summarize_obs(obs) + "\n"

                    brief_obs = action_text + obs_text

                    obs_queue.append(brief_obs)
                    with open(file_name, "a") as f:
                        f.write(f"> {taken_action} \n {brief_obs} \n")

                    # Define action passed
                    if "You can't move there, the door is closed." in brief_obs:
                        large_loop_error_message = f"This is the action you take: {taken_action}. \
                            The door that you are moving to is closed. \
                            You should first open door to that direction then move there!"
                        break
                    elif "That is already open." in brief_obs:
                        large_loop_error_message = f"This is the action you take: {taken_action}. \
                            You try to open a door that is already open. You already visited here. Make sure the status of door is correct."
                        break
                    elif "I'm not sure what you mean." in brief_obs:
                        action_passed = False
                        if "open door" in taken_action:
                            large_loop_error_message = f'This is the action you take: {taken_action}. \
                                When you try to open door, there is no door here or there is nothing in this direction.\
                                If there is no door, you can directly move to that direction.\n'
                        elif "move" in taken_action:
                            large_loop_error_message = f'This is the action you take: {taken_action}. \
                                You cannot move to that direction. Review the predicate of your actions and the problem files to check the status.'
                        else:
                            large_loop_error_message = f'This is the action you take: {taken_action}. \
                                You got the environment error!'
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
        
        with open("output/results.csv", "a", newline="") as csvfile:
            # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
            data_row = [today, model_name, trial, coin_found, len(trial_record)-1,trial_record[-1][-1], trial_record]
            writer = csv.writer(csvfile)
            writer.writerow(data_row)


def run_baseline_model(model_name, start_trials, end_trials):
    for trial in range(start_trials, end_trials):
        coin_found = False
        today = date.today()
        file_name = f"output/06_021425_baseline/{today}_{model_name.replace('/','_')}_{trial}.txt"
        trial_record = []  # This will be a list of steps; each step is a list of large-loop iteration numbers

        env = TextWorldExpressEnv(envStepLimit=100)
        NUM_LOCATIONS = 11
        env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
        obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
        with open(file_name, "a") as f:
            f.write(f"Observations: {obs} \n")
            f.write(f"Gold path: {env.getGoldActionSequence()} \n")
            f.write(f"Valid Actions: {infos['validActions']} \n")
            f.write(f"taskDescription: {infos['taskDescription']} \n")

        valid_actions = sorted(infos['validActions'])
        if 'look around' in valid_actions:
            valid_actions.remove('look around')
        if 'inventory' in valid_actions:
            valid_actions.remove('inventory')

        MAX_STEPS = 20
        brief_obs = "Action: look around\n" + summarize_obs(obs) + "\n"
        with open(file_name, "a") as f:
            f.write(f"brief_obs: {brief_obs} \n")

        action_queue = []
        obs_queue = []
        all_actions = []
        successful_actions = []
        overall_memory = brief_obs
        overall_memory_dic = []  # For recording detailed memory if needed
        end_game = False

        for step_id in range(MAX_STEPS):
            with open(file_name, "a") as f:
                f.write(f"\n\n====Step {step_id}==== \n")
            trial_step_record = []  # This will record each large-loop try for the current step
            within_step_tries = 0
            action_passed = False
            large_loop_error_message = ""

            # Under each step, try up to 5 large-loop iterations until actions pass.
            while within_step_tries < 5 and not action_passed:
                with open(file_name, "a") as f:
                    f.write(f"\n----Larger Loop No. {within_step_tries}---- \n")
                    f.write(f"successful_actions: {successful_actions} \n")
                within_step_tries += 1

                if within_step_tries > 1:  # For subsequent tries, reset the environment
                    env = TextWorldExpressEnv(envStepLimit=100)
                    NUM_LOCATIONS = 11
                    env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
                    obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
                    for act in successful_actions:
                        obs, reward, done, infos = env.step(act)

                # Reset action queues and temporary memory for this large-loop iteration.
                action_queue = []
                tem_action_queue = []
                tem_memory = ""

                start_checkpoint = True
                while start_checkpoint or action_queue:
                    with open(file_name, "a") as f:
                        f.write(f"Small Loop, action_queue: {action_queue} \n")
                    start_checkpoint = False

                    if not action_queue:
                        if obs_queue:
                            brief_obs = "\n".join(obs_queue)
                            obs_queue = []
                        # Generate actions using the baseline LLM function.
                        actions = llm_to_actions_baseline(model_name, brief_obs, valid_actions, overall_memory, large_loop_error_message)
                        with open(file_name, "a") as f:
                            f.write(f"Generated actions: {actions} \n")

                        if actions:
                            action_queue.extend(actions)
                            tem_action_queue.extend(actions)
                            all_actions.extend(actions)
                        else:
                            end_game = True
                            break

                    with open(file_name, "a") as f:
                        f.write(f"Current action_queue: {action_queue} \n")
                    taken_action = action_queue.pop(0)
                    obs, reward, done, infos = env.step(taken_action)

                    # Immediately end the game if coin is found.
                    if "coin" in obs:
                        taken_action = "take coin"
                        obs, reward, done, infos = env.step(taken_action)
                        end_game = True
                        with open(file_name, "a") as f:
                            f.write("Coin found!\n")
                        coin_found = True
                        break

                    action_text = "Action: " + taken_action + "\n"
                    obs_text = summarize_obs(obs) + "\n"
                    brief_obs = action_text + obs_text
                    obs_queue.append(brief_obs)
                    with open(file_name, "a") as f:
                        f.write(f"> {taken_action} \n {brief_obs} \n")

                    # Check for common errors in the observation and update the error message.
                    if "You can't move there, the door is closed." in brief_obs:
                        large_loop_error_message = (
                            f"This is the action you take: {taken_action}. "
                            "The door that you are moving to is closed. "
                            "You should first open the door to that direction then move there!"
                        )
                        break
                    elif "That is already open." in brief_obs:
                        large_loop_error_message = (
                            f"This is the action you take: {taken_action}. "
                            "You try to open a door that is already open. You already visited here. "
                            "Make sure the status of door is correct."
                        )
                        break
                    elif "I'm not sure what you mean." in brief_obs:
                        if "open door" in taken_action:
                            large_loop_error_message = (
                                f"This is the action you take: {taken_action}. "
                                "When you try to open door, there is no door here or nothing in that direction. "
                                "If there is no door, you can directly move to that direction.\n"
                            )
                        elif "move" in taken_action:
                            large_loop_error_message = (
                                f"This is the action you take: {taken_action}. "
                                "You cannot move to that direction. Review your action predicates and the problem files to check the status."
                            )
                        else:
                            large_loop_error_message = (
                                f"This is the action you take: {taken_action}. "
                                "You got the environment error!"
                            )
                        break

                    tem_memory += brief_obs
                    overall_memory_dic.append({"type": "action", "content": taken_action})
                    overall_memory_dic.append({"type": "observation", "content": summarize_obs(obs)})

                    if not action_queue:
                        action_passed = True
                        successful_actions.extend(tem_action_queue)
                        overall_memory += tem_memory

                # Record this large-loop iteration result for the step.
                trial_step_record.append(within_step_tries)
                if (within_step_tries == 5 and not action_passed) or end_game:
                    end_game = True
                    break

            trial_record.append(trial_step_record)
            if end_game:
                break

        with open("output/baseline_results.csv", "a", newline="") as csvfile:
            # Write out: date, model_name, trial, coin_found, last step index, last large-loop iteration, and the full trial record.
            data_row = [today, model_name, trial, coin_found, len(trial_record)-1, trial_record[-1] if trial_record else None, trial_record]
            writer = csv.writer(csvfile)
            writer.writerow(data_row)


# Merging method: only observation generate problem files
def run_merging_pf_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", start_trial=0, end_trial=11, merging_method="llm"):
    """
    An alternative version of your iterative loop approach that:
      1) Only generates a new problem file from each new observation (without referencing the old PF).
      2) Fixes that newly created PF in a large/small loop manner (like your existing approach).
      3) Merges the newly fixed PF with the previously merged/old PF either by LLM or code, depending on merging_method.

    Args:
        model_name (str): e.g. "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        start_trial (int): which trial to start from
        end_trial (int): which trial to end at (exclusive)
        merging_method (str): "llm" or "code" to specify how we merge the old PF with the new PF.

    Note:
        This function imitates the logic of run_iterative_model(), but the domain file df is kept stable/assumed
        or can be (re)generated if you wish. The key difference is how PF is produced: we always use a "generate from observation"
        approach and then fix that new PF, finally merging it with the old PF.
    """

    # If you have a stable domain file somewhere, you could define it here (or load from disk).
    # Otherwise, you can keep it empty and let your LLM create or fix it as well.
    # For demonstration, let's just say df is empty or set some minimal domain for the "open-door" & "move" actions:
    # df = ""  # or a minimal domain if you want
    # old_pf = ""  # We'll merge new PF into this each step

    for trial in range(start_trial, end_trial):
        coin_found = False
        today = date.today()
        file_name = f"output/07_022825_merging/merging_{today}_{model_name.replace('/','_')}_{trial}.txt"
        trial_record = []

        # Initialize environment
        env = TextWorldExpressEnv(envStepLimit=100)
        NUM_LOCATIONS = 11
        env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
        obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)

        with open(file_name, "a") as f:
            f.write(f"Observations: {obs} \n")
            f.write(f"Gold path: {env.getGoldActionSequence()} \n")
            f.write(f"Valid Actions: {infos['validActions']} \n")
            f.write(f"taskDescription: {infos['taskDescription']} \n")

        # We'll remove these for clarity
        valid_actions = sorted(infos['validActions'])
        if 'look around' in valid_actions:
            valid_actions.remove('look around')
        if 'inventory' in valid_actions:
            valid_actions.remove('inventory')

        MAX_STEPS = 20

        # Start observation
        brief_obs = "Action: look around\n" + summarize_obs(obs) + "\n"
        with open(file_name, "a") as f:
            f.write(f"brief_obs: {brief_obs} \n")

        # Some tracking variables
        action_queue = []
        obs_queue = []
        df = ""
        pf_to_merge = ""
        all_actions = []
        successful_actions = []
        overall_memory = brief_obs

        end_game = False

        for step_id in range(MAX_STEPS):
            with open(file_name, "a") as f:
                f.write(f"\n\n====Step {step_id}==== \n")

            trial_step_record = []
            within_step_tries = 0
            action_passed = False
            large_loop_error_message = ""

            # We'll attempt up to 5 large-loop tries for each step
            while within_step_tries < 5 and not action_passed:
                with open(file_name, "a") as f:
                    f.write(f"\n----Larger Loop No. {within_step_tries}---- \n")
                    f.write(f"successful_actions: {successful_actions} \n")

                within_step_tries += 1

                # If not the first iteration in the large loop, re-simulate successful actions so far
                if within_step_tries > 1:
                    env = TextWorldExpressEnv(envStepLimit=100)
                    env.load(gameName="coin", gameParams=f"numLocations={NUM_LOCATIONS},numDistractorItems=0,includeDoors=1,limitInventorySize=0")
                    obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
                    for successful_action in successful_actions:
                        obs, reward, done, infos = env.step(successful_action)

                action_queue = []
                tem_action_queue = []
                tem_memory = ""
                start_checkpoint = True

                # ---- Small loop starts ----
                while start_checkpoint or action_queue:
                    with open(file_name, "a") as f:
                        f.write(f'Small Loop, action_queue: {action_queue} \n')

                    start_checkpoint = False

                    if not action_queue:
                        # Combine any queued observations
                        if obs_queue:
                            brief_obs = "\n".join(obs_queue)
                            obs_queue = []

                        num_tries = 0

                        # Generate new PF from the observation
                        df, pf, err, prompt = generate_problem_file_from_observation(
                            observation=brief_obs,
                            model_name=model_name,
                            domain_file=df  # Initially, df is an empty string
                        )

                        action, err_2 = get_action_from_pddl(df, pf) # error 2 here

                        with open(file_name, "a") as f:
                            f.write(f"--Small Loop--: {num_tries} \n")
                            f.write(f"Error: {err} & {err_2} \n")
                            f.write(f"Prompt: {prompt} \n") 
                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                            f.write(f"Actions from solver(df, pf): {action} \n")

                        while not action and num_tries < 5:
                            df, pf, err, prompt = generate_problem_file_from_observation(
                                observation=brief_obs,
                                model_name=model_name,
                                domain_file=df,
                                err=err, 
                                err_2=err_2
                            )
                            action, err_2 = get_action_from_pddl(df, pf)
                            num_tries += 1

                            with open(file_name, "a") as f:
                                f.write(f"--Small Loop--: {num_tries} \n")
                                f.write(f"Error: {err} & {err_2} \n")
                                f.write(f"Prompt: {prompt} \n") 
                                f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                f.write(f"Actions from solver(df, pf): {action} \n")
                        
                        trial_step_record.append([within_step_tries, num_tries])

                        if not action:
                            end_game = True
                            break
                        else:
                            action_queue.extend(action)
                            all_actions.extend(action)
                            tem_action_queue.extend(action)

                    with open(file_name, "a") as f:
                        f.write(f"Current action_queue: {action_queue} \n")

                    if not action_queue:
                        break

                    # Pop the first action
                    taken_action = action_queue.pop(0)
                    obs, reward, done, infos = env.step(taken_action)

                    if "coin" in obs:
                        # If we see 'coin' in obs, let's pick it up and end the game
                        taken_action2 = "take coin"
                        obs, reward, done, infos = env.step(taken_action2)
                        coin_found = True
                        end_game = True
                        with open(file_name, "a") as f:
                            f.write("Coin found!\n")
                        break

                    action_text = "Action: " + taken_action + "\n"
                    obs_text = summarize_obs(obs) + "\n"
                    brief_obs = action_text + obs_text
                    obs_queue.append(brief_obs)

                    with open(file_name, "a") as f:
                        f.write(f"> {taken_action} \n {brief_obs} \n")

                    # Basic checks for environment error messages
                    if "You can't move there, the door is closed." in brief_obs:
                        large_loop_error_message = f"This is the action you take: {taken_action}. The door is closed."
                        break
                    elif "That is already open." in brief_obs:
                        large_loop_error_message = f"This is the action you take: {taken_action}. The door was already open."
                        break
                    elif "I'm not sure what you mean." in brief_obs:
                        action_passed = False
                        large_loop_error_message = f"This is the action you take: {taken_action}. 'I'm not sure what you mean.'"
                        break

                    tem_memory += brief_obs

                    # If we exhausted the queue successfully, mark success
                    if not action_queue:
                        action_passed = True
                        overall_memory += tem_memory
                        successful_actions.extend(tem_action_queue)
                # ---- End small loop ----


                # End condition if we used up too many tries or found coin
                if (within_step_tries == 5 and not action_passed) or end_game:
                    end_game = True
                    break
            # ---- End Large loop ----

            # After BOTH the small loop and large loop finish successfully, merge the new PF.
            if action_passed:
                if pf_to_merge and pf:
                    print('**** merging old with new generated')
                    if merging_method == "llm":
                        merged_pf = merge_problem_files_llm(old_problem_file=pf_to_merge, new_problem_file=pf, model_name=model_name)
                    else:
                        merged_pf = merge_problem_files_code(pf_to_merge, pf)
                else:
                    print('**** Directly using the first pf')
                    merged_pf = pf
                pf_to_merge = merged_pf

            trial_record.append(trial_step_record)
            if end_game:
                break

        # Log results to CSV at the end of each trial
        with open("output/merging_results.csv", "a", newline="") as csvfile:
            data_row = [
                today,
                model_name,
                trial,
                coin_found,
                len(trial_record)-1 if trial_record else 0,
                trial_record[-1] if trial_record else "No steps",
                trial_record
            ]
            writer = csv.writer(csvfile)
            writer.writerow(data_row)


# Run baseline models
# run_baseline_model("gpt-4o-mini-2024-07-18", 0, 2)
# run_baseline_model("o3-mini-2025-01-31", 0, 10)
# run_baseline_model("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 3, 10) # models--google--gemma-2-27b-it
# run_baseline_model("google/gemma-2-27b-it", 0, 10)


# Run PDDL generation models
# run_iterative_model("o3-mini-2025-01-31", 0, 2) # gpt-4o; o3-mini
# run_iterative_model("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 10, 10) # models--google--gemma-2-27b-it
# run_iterative_model("google/gemma-2-27b-it", 6, 10)



run_merging_pf_model("o3-mini-2025-01-31", 0, 3, merging_method="llm")
