import time
import csv
import json
import asyncio

from kani import Kani
from kani.engines.huggingface import HuggingEngine

import subprocess
import requests

from textworld_express import TextWorldExpressEnv

from openai import OpenAI
import os

client = OpenAI()
os.environ["OPENAI_API_KEY"] = ""


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
def run_llm_model(prompt, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"):

    if "gpt" in model_name: # closed source LLMs
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

        df = result.get("df", None)
        pf = result.get("pf", None)

        if df is None or pf is None:
            raise ValueError("Missing 'df' or 'pf' in the response. Check the prompt or the model output.")

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
        response_content = response_content[response_content.find('</think>')+10:]

        # If your model returns a JSON block wrapped in triple backticks, strip them
        if response_content.startswith("```json"):
            response_content = (
                response_content
                .lstrip("```json")
                .rstrip("```")
                .strip()
            )

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
        print('You have duplicated error message!!')
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

def run_iterative_model(model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", trials = 2):
    # trial_record = 
    # structured_info_record = "output/summary"
    for trial in trials:
        # date = ... # today's date
        file_name = f"output/05_020625_100trials/{date}_{model_name}_{trial}.txt"
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

        # print("Observations: "+obs)
        # print("Gold path: " + str(env.getGoldActionSequence()))
        # print("Valid Actions: " + str(infos['validActions']))
        # print("taskDescription: " + str(infos['taskDescription']))

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
            # print(f"\n====Step {step_id}====")

            trial_step_record = []

            within_step_tries = 0
            action_passed = False

            large_loop_error_message = ""

            # Under step#, it should repeat until run all actions and found no error
            while within_step_tries < 5 and not action_passed:
                with open(file_name, "a") as f:
                    f.write(f'\n----Larger Loop No. {within_step_tries}---- \n') 
                    f.write(f'successful_actions: {successful_actions} \n')
                # print(f'----Larger Loop No. {within_step_tries}----')
                # print(f'successful_actions: {successful_actions}')
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
                    # print(f'Small Loop, action_queue: {action_queue}')
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
                            # print("\n--Small Loop--",num_tries)
                            # print("Error:", err)
                            # print("Prompt:", prompt)
                            # print("df and pf:", df, pf)
                            # print("Actions from solver(df, pf)", action)

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

                                # print("\n--Small Loop--",num_tries)
                                # print("Error:", err)
                                # print("Prompt:", prompt)
                                # print("df and pf:", df, pf)
                                # print("Actions from solver(df, pf)", action)
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
                            # print("\n--Small Loop--",num_tries)
                            # print("Error:", err)
                            # print("Prompt:", prompt)
                            # print("df and pf:", df, pf)
                            # print("Actions from solver(df, pf)", action)

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
                                # print("\n--Small Loop--",num_tries)
                                # print("Error:", err)
                                # print("Prompt:", prompt)
                                # print("df and pf:", df, pf)
                                # print("Actions from solver(df, pf)", action)

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
                    # print("Current action_queue:", action_queue)
                    
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
                        # print('Coin found!')
                        break
                    
                    action_text = "Action: " + taken_action + "\n"
                    obs_text = summarize_obs(obs) + "\n"

                    brief_obs = action_text + obs_text

                    obs_queue.append(brief_obs)
                    with open(file_name, "a") as f:
                        f.write(f"> {taken_action} \n {brief_obs} \n") 
                    # print(">", taken_action)
                    # print(brief_obs)
                    # =====
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
                        # print('===error message here!', obs_text)
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

                # break here

                if (within_step_tries == 5 and not action_passed) or end_game:
                    end_game = True
                    break

            trial_record.append(trial_step_record)

            if end_game:
                break
        
        with open("output/results.csv", "a", newline="") as csvfile:
            # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
            data_row = [date, model_name, trial, len(trial_record)-1,trial_record[-1][-1], trial_record]
            writer = csv.writer(csvfile)
            writer.writerow(data_row)


run_iterative_model("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 2)
# run_iterative_model("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 2)