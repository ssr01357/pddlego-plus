import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import time
from datetime import date
import csv
import json
import asyncio
import re

from dotenv import load_dotenv
load_dotenv()

from kani import Kani
from kani.engines.huggingface import HuggingEngine

import subprocess
import requests

from textworld_express import TextWorldExpressEnv

from openai import OpenAI
from utils import repair_json

# from solver import run_solver

import torch
import gc

_hf_engine_cache: dict[str, HuggingEngine] = {}
_kani_cache: dict[str, Kani] = {}

OPENAI_MODELS_LIST = ['gpt-4o','o3-mini',"gpt-4.1","o4-mini", "gpt-5-nano"]
ENV_PARAMS = {"gameName": "coin", "gameParams": "numLocations=11,numDistractorItems=0,includeDoors=1,limitInventorySize=0"}
MAX_STEPS = 20

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
    err = result['stderr'] + result['stdout']
    return map_actions(action), err, result # 액션의 리스트( = 플랜)을 반환함

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
    
    else: # Open source LLMs
        async def _ask_model(model_name, user_prompt):
            if model_name not in _hf_engine_cache:
                engine = HuggingEngine(
                    model_id=model_name,
                    use_auth_token=True,
                    model_load_kwargs={"device_map": "auto"} # , "trust_remote_code": True
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
    return result


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



def llm_to_pddl(model_name, brief_obs, valid_actions, prev_df="", prev_pf="", prev_err=None, have_error=False, have_duplicate=False, edit=False, overall_memory=None, large_loop_error_message = None, goal_type='detailed'):
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

    prompt_obs_action_subgoal = f"""
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
        Note: in problem file's init, you shouldn't have "not ()" but only the single status
    """ 

    prompt_obs_action_detailed = f"""
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
        You made some mistakes when generating those files. Here is the error message: {prev_err}
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
    if goal_type == 'detailed':
        prompt += prompt_obs_action_detailed
    elif goal_type == 'subgoal':
        prompt += prompt_obs_action_subgoal

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


    result = run_llm(prompt, model_name)
    df = result.get("df")
    pf = result.get("pf")

    return df, pf, prompt


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

        Provide the output in strict JSON format like this, while you should only generate one action at a time:
        {{
            "actions": ["action1"]
        }}
    """
    #  "while you should only generate one action at a time" "actions": ["action1"]
    #  "actions": ["action1", "action2", ...]
    result = run_llm(prompt, model_name)
    actions = result.get("actions", None)
    return actions



# Main functions here:
def run_iterative_model(model_name, start_trial = 0, end_trial = 11, folder_name="3_0421_CC", result_name="CC_results", goal_type="detailed"):
    for trial in range(start_trial, end_trial): #이건 정답 찾을때까지
        retry = 0
        while retry < 2:  # 이건 에러처리용 다시시도
            try:
                coin_found = False
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
                
                env = TextWorldExpressEnv(envStepLimit=100)
                env.load(**ENV_PARAMS)
                obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
                with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
                    f.write(f"Observations: {obs} \n") 
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {infos['validActions']} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")

                

                # task_description = infos['taskDescription']
                valid_actions = sorted(infos['validActions'])
                valid_actions.remove('look around') # 이건 초반에 항상 초기값
                valid_actions.remove('inventory') # 이건 필요없으니까?

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
                            # reset env by refilling successful actions (stupid but useful)<- 왜 이렇게 하지?
                            env = TextWorldExpressEnv(envStepLimit=100)
                            env.load(**ENV_PARAMS)
                            obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True) # 여기서 나온 obs, infos는 쓰지도 않음
                            for successful_action in successful_actions: # 초기화 다음, 어떤 행동을 취하기 시작한 후부터의 아웃풋에 주목
                                obs, reward, done, infos = env.step(successful_action) # <-아... 앞단계에서 검증된거 한꺼번에 넣어주고 다음 action찾으려고

                        action_queue = [] # reset action_queue ()
                        tem_action_queue = []
                        tem_memory = ""

                        start_checkpoint = True
                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f'Small Loop, action_queue: {action_queue} \n')
                            start_checkpoint = False

                            if not action_queue: # when start_checkpoint = True before. 이 시점에는 무조건 false긴 하지만 그래도 이 while문 들어올 때
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                action = "" # 스몰루프를 시작할 때마다, action을 초기화한다...그리고 obs_queue 리스트도 초기화한다....이게 llm먹이는 기본 단위인가?
                                
                                if not df and not pf: # First step no need duplicates detection
                                    num_tries = 0
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions) # error 1 here
                                    action, err, raw_result = get_action_from_pddl(df, pf) # error 2 here
                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                    
                                    # if not err and not action:
                                    # if True:
                                    #     with open(file_name, "a") as f:
                                    #         f.write(f"Parser found no error but no actions generated, likely due to unsolvable problem. \n")
                                    #         f.write(f"Raw result from parser: {raw_result} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, True, False, edit)
                                        action, err, raw_result = get_action_from_pddl(df, pf)
                                        num_tries += 1
                                        
                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")

                                        # if not err and not action:
                                        # if True:
                                        #     with open(file_name, "a") as f:
                                        #         f.write(f"Parser found no error but no actions generated, likely due to unsolvable problem. \n")
                                        #         f.write(f"Raw result from parser: {raw_result} \n")
                                else:
                                    num_tries = 0
                                    # Every time read new error message from larger loop
                                    # In llm_to_pddl, detect if new large loop error message exists
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, None, False, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message) # need to add new error message
                                    action, err, raw_result = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")

                                    # if not err and not action:
                                    # if True:
                                    #     with open(file_name, "a") as f:
                                    #         f.write(f"Parser found no error but no actions generated, likely due to unsolvable problem. \n")
                                    #         f.write(f"Raw result from parser: {raw_result} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, err, True, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message)
                                        action, err, raw_result = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")

                                        # if not err and not action:
                                        # if True:
                                        #     with open(file_name, "a") as f:
                                        #         f.write(f"Parser found no error but no actions generated, likely due to unsolvable problem. \n")
                                        #         f.write(f"Raw result from parser: {raw_result} \n")

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
                                f.write(f"> {taken_action} \n {obs} \n")

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
                
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
                    model_type = "PDDL"
                    data_row = [today, model_name, model_type, goal_type, trial, coin_found, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    # [today, model_name, trial, coin_found, len(trial_record)-1,trial_record[-1][-1], trial_record]
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


def run_baseline_model(model_name, start_trials, end_trials, folder_name="08_031825_alfworld", result_name="alfworld_results"):
    for trial in range(start_trials, end_trials):
        retry = 0
        while retry < 2:  # allow up to 2 attempts per trial
            try:
                coin_found = False
                today = date.today()

                fixed_model_name = model_name.replace("/","_")

                folder_path = f"output/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_name = f"{folder_path}/{today}_{fixed_model_name}_baseline_detailed_{trial}.txt"
                if os.path.exists(file_name): # retry == 1 and 
                    open(file_name, 'w').close()  # empty file
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                trial_record = []

                env = TextWorldExpressEnv(envStepLimit=100)
                env.load(**ENV_PARAMS)
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
                            env.load(**ENV_PARAMS)
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
                                f.write(f"> {taken_action} \n {obs} \n")

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

                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    # Write out: date, model_name, trial, coin_found, last step index, last large-loop iteration, and the full trial record.
                    # data_row = [today, model_name, trial, coin_found, len(trial_record)-1, trial_record[-1] if trial_record else None, trial_record]
                    model_type = 'baseline' # PDDL
                    goal_type = 'detailed' # detailed or subgoal
                    data_row = [today, model_name, model_type, goal_type, trial, coin_found, len(trial_record)-1,trial_record[-1][-1], trial_record]

                    writer = csv.writer(csvfile)
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




i = 0
num_trials = 2
# folder_name = "CC_o4_mini_high"
folder_name = "yewon_coin_0831" 
result_name = folder_name
model_id1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_id2 = "Qwen/Qwen3-32B"
model_id3 = "meta-llama/Llama-3.1-8B-Instruct"
model_id4 = "meta-llama/Llama-3.1-70B-Instruct"
model_id5 = "meta-llama/Llama-3.3-70B-Instruct"
model_id6 = "mistralai/Mixtral-8x7B-Instruct-v0.1"

## Run PlanGen models
# run_baseline_model(model_id1, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# clear_cuda_memory(model_id1)
run_baseline_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name)
clear_cuda_memory(model_id2)
# run_baseline_model(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# clear_cuda_memory(model_id3)
# run_baseline_model("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_baseline_model("gpt-4.1-2025-04-14", i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_baseline_model("o4-mini-2025-04-16", i, i+num_trials, folder_name=folder_name, result_name=result_name)
# run_baseline_model("deepseek", i, i+num_trials, folder_name=folder_name, result_name=result_name)

# run_baseline_model_50("gpt-4o-2024-05-13", folder_name=folder_name, result_name=result_name)
# run_baseline_model_50("o3-mini-2025-01-31", folder_name=folder_name, result_name=result_name)
# run_baseline_model_50("deepseek", folder_name=folder_name, result_name=result_name)
# run_baseline_model_50("gpt-4.1-2025-04-14", folder_name=folder_name, result_name=result_name)
# run_baseline_model_50("o4-mini-2025-04-16", folder_name=folder_name, result_name=result_name)


## Run PDDLego+ models
# run_iterative_model(model_id1, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id1)
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id2)
# run_iterative_model(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id3)
# run_iterative_model(model_id4, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id4)
# run_iterative_model(model_id5, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id5)
# run_iterative_model(model_id6, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id6)
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# clear_cuda_memory(model_id2)
# run_iterative_model(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# clear_cuda_memory(model_id3)
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("gpt-4.1-2025-04-14", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("o4-mini-2025-04-16", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("deepseek", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")

