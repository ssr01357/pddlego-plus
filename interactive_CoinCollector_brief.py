import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
import torch
import gc
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
from prompts_CoinCollector_final import *
from solver import run_solver

_hf_engine_cache: dict[str, HuggingEngine] = {}

OPENAI_MODELS_LIST = ['gpt-4o','o3-mini',"gpt-4.1","o4-mini", "gpt-5-nano-2025-08-07", "gpt-5-nano"]
ENV_PARAMS = {"gameName": "coin", "gameParams": "numLocations=11,numDistractorItems=0,includeDoors=1,limitInventorySize=0"}
MAX_STEPS = 20

def _sanitize_valid_actions(infos):
    # vals = sorted(infos['validActions'])
    # return [v for v in vals if v not in ("look around", "inventory") and not v.startswith("close door to")]
    return ['move east', 'move north', 'move south', 'move west', 'open door to east', 'open door to north', 'open door to south', 'open door to west']

def clear_cuda_memory(model_name):
    global _hf_engine_cache

    if model_name in _hf_engine_cache:
        del _hf_engine_cache[model_name]
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_action_from_pddl(df, pf):
    result = run_solver(df, pf, "dual-bfws-ffparser", validate_with_val=True)
    plan_text = result["output"].get("plan") or ""
    # If there is no plan, return (None, logs) so the outer loop behaves as before.
    mapped = map_actions(plan_text) if plan_text else None
    return mapped, result["stderr"], plan_text

def run_llm_json(prompt: str, model_name: str, system_prompt=None) -> dict:
    """
    Send `prompt` to `model_name`, parse JSON (with repair), and
    return the full response dict. Callers pick out the fields they need.
    """
    if any(model_name.startswith(base) for base in OPENAI_MODELS_LIST):
        client = OpenAI()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {"model": model_name, "messages": messages} # don't set temperature to 0
        if re.match(r"^o\d+", model_name):
            params["reasoning_effort"] = "high"

        response = client.chat.completions.create(**params)
        response_content = response.choices[0].message.content

    elif model_name == "deepseek":
        deepseekAPI = os.getenv("deepseek_API")
        client = OpenAI(api_key=deepseekAPI, base_url="https://api.deepseek.com")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=False,
        )
        response_content = response.choices[0].message.content

    # Open-source LLMs via HuggingFace
    else:
        async def _ask_model(model_name, user_prompt, sys_prompt):
            if model_name not in _hf_engine_cache:
                engine = HuggingEngine(
                    model_id=model_name,
                    use_auth_token=True,
                    model_load_kwargs={"device_map": "auto", "trust_remote_code": True},
                )
                _hf_engine_cache[model_name] = engine
            engine = _hf_engine_cache[model_name]
            
            ai = Kani(engine, system_prompt=sys_prompt)
            return await ai.chat_round_str(user_prompt)

        response_content = asyncio.run(_ask_model(model_name, prompt, system_prompt))
           
     

    # Single JSON parsing + repair path
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

def map_env_feedback_to_large_loop_error(brief_obs: str, taken_action: str):
    # Door is closed -> tell the planner to open it before moving
    if "You can't move there, the door is closed." in brief_obs:
        msg = (
            f"This is the action you take: {taken_action}. "
            "The door that you are moving to is closed. "
            "You should first open door to that direction then move there!"
        )
        return msg, "door_closed"

    # Door already open -> avoid redundant open
    if "That is already open." in brief_obs:
        msg = (
            f"This is the action you take: {taken_action}. "
            "You try to open a door that is already open. You already visited here. "
            "Make sure the status of door is correct."
        )
        return msg, "already_open"

    # Generic parser/command failure; tailor by attempted action
    if "I'm not sure what you mean." in brief_obs:
        if "open door" in taken_action:
            msg = (
                f"This is the action you take: {taken_action}. "
                "When you try to open door, there is no door here or there is nothing in this direction. "
                "If there is no door, you can directly move to that direction.\n"
            )
            return msg, "invalid_open"
        elif "move" in taken_action:
            msg = (
                f"This is the action you take: {taken_action}. "
                "You cannot move to that direction."
            )
            return msg, "invalid_move"
        else:
            msg = (
                f"This is the action you take: {taken_action}. "
                "You got the environment error!"
            )
            return msg, "env_error"

    return None, None


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

def llm_to_pddl(model_name, brief_obs, valid_actions, prev_df="", prev_pf="", prev_err="", have_duplicate=False, overall_memory=None, large_loop_error_message = None, goal_type='detailed'):
    prompt = prompt_format.format()

    # all prompts should have observations and actions
    if goal_type == 'detailed':
        prompt += prompt_obs_action_detailed.format(brief_obs=brief_obs, valid_actions=valid_actions)
    elif goal_type == 'subgoal':
        prompt += prompt_obs_action_subgoal.format(brief_obs=brief_obs, valid_actions=valid_actions)

    if prev_df and prev_pf:
        prompt += prompt_prev_files.format(prev_df=prev_df, prev_pf=prev_pf, overall_memory=overall_memory)

        if large_loop_error_message:
            prompt += prompt_simulation_error.format(large_loop_error_message=large_loop_error_message)
        if prev_err:
            prompt += prompt_error_parser.format(prev_err=prev_err)
        if have_duplicate:
            prompt += prompt_duplicate_note.format()

        prompt += "\nNow rewrite both the domain and problem files with the minimal fix.\n"

    resp = run_llm_json(prompt, model_name, system_prompt=SYS_PROMPT_PDDL)
    df = resp.get("df", None)
    pf = resp.get("pf", None)
    return df, pf, prompt


def llm_to_actions_baseline(model_name, brief_obs, valid_actions, overall_memory=None, large_loop_error_message=None):
    prompt = prompt_baseline.format(
        brief_obs=brief_obs,
        valid_actions=valid_actions,
        overall_memory=overall_memory or "N/A",
        large_loop_error_message=large_loop_error_message or "N/A",
    )

    actions = run_llm_json(prompt, model_name, system_prompt=SYS_PROMPT_PLAN).get("actions", None)
    return actions


# Main functions here:
def run_iterative_model(model_name, start_trial = 0, end_trial = 11, folder_name="3_0421_CC", result_name="CC_results", goal_type="detailed"):

    # for trial in range(start_trial, end_trial): 
    adjusted_end = end_trial
    trial = start_trial
    while trial < adjusted_end:
        start_time = time.time()
        retry = 0
        while retry < 2: 
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
                obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                valid_actions = _sanitize_valid_actions(infos)
        
                # If coin is visible in the very first observation, skip this seed and
                # extend the window so we still complete the requested number of meaningful trials.
                if "coin" in obs.lower():
                    print(f"[Seed {trial}] Coin found at the beginning — skipping this seed and extending the trial window.")
                    adjusted_end += 1
                    break  # break out of the retry loop; we'll advance to the next seed below
 
                with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
                    f.write(f"Observations: {obs} \n") 
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {valid_actions} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")


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
                end_game = False

                overall_memory = brief_obs

                for step_id in range(MAX_STEPS):
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
                            obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True) # 여기서 나온 obs, infos는 쓰지도 않음
                            for successful_action in successful_actions: # 초기화 다음, 어떤 행동을 취하기 시작한 후부터의 아웃풋에 주목
                                obs, reward, done, infos = env.step(successful_action) # <-아... 앞단계에서 검증된거 한꺼번에 넣어주고 다음 action찾으려고
                            valid_actions = _sanitize_valid_actions(infos) # 이거 오리지널스크립트에서는 업데이트 안함.이유를 알았음. 어차피 안변함 -> no, 틀렸음. 바뀌네. 

                        action_queue = [] # reset action_queue ()
                        tem_action_queue = []
                        tem_memory = ""

                        start_checkpoint = True

                        executed_actions = [] 
                        executed_memory = ""

                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f'Small Loop, action_queue: {action_queue} \n')
                            start_checkpoint = False

                            if not action_queue: # when start_checkpoint = True before. 이 시점에는 무조건 false긴 하지만 그래도 이 while문 들어올 때
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                action = ""
                                
                                if not df and not pf: # First step no need duplicates detection
                                    num_tries = 0
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions) # error 1 here
                                    action, err, plan_text = get_action_from_pddl(df, pf) # error 2 here
                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, False)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1
                                        
                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")
                                else:
                                    num_tries = 0
                                    # Every time read new error message from larger loop
                                    # In llm_to_pddl, detect if new large loop error message exists
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, "", detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message) # need to add new error message
                                    action, err, plan_text = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")

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
                            valid_actions = _sanitize_valid_actions(infos)

                            executed_actions.append(taken_action)
                            brief_obs = "Action: " + taken_action + "\n" + summarize_obs(obs)+'\n'
                            executed_memory += brief_obs 

                            # Directly end the game if found coin
                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                valid_actions = _sanitize_valid_actions(infos)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write('Coin found!\n')
                                    f.write(f"Final obs: {obs} \n")
                                    coin_found = True
                                break
                            
                            # action_text = "Action: " + taken_action + "\n"
                            # obs_text = summarize_obs(obs) + "\n"

                            # brief_obs = action_text + obs_text

                            obs_queue.append(brief_obs)
                            with open(file_name, "a") as f:
                                f.write(f"> {taken_action} \n {obs} \n")

                            large_loop_error_message, _code = map_env_feedback_to_large_loop_error(brief_obs, taken_action)
                            # If you want to go one step further later, you can use the returned _code
                            if _code:
                                if executed_actions:
                                    successful_actions.extend(executed_actions[:-1]) # except the last one that caused error
                                    overall_memory += executed_memory[:-len(brief_obs)] # except the last one that caused error
                                action_passed = False
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
                duration = time.time() - start_time
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
                    model_type = "PDDL"
                    data_row = [today, model_name, model_type, goal_type, trial, coin_found, duration, len(trial_record)-1,trial_record[-1][-1], trial_record]
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
        trial += 1

# =========================

def run_iterative_model2(model_name, start_trial = 0, end_trial = 11, folder_name="3_0421_CC", result_name="CC_results", goal_type="detailed"):
   
    # for trial in range(start_trial, end_trial):
    adjusted_end = end_trial
    trial = start_trial
    while trial < adjusted_end:
        start_time = time.time() 
        retry = 0
        while retry < 2: 
            try:
                coin_found = False
                today = date.today()

                fixed_model_name = model_name.replace("/","_")

                folder_path = f"output/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_name = f"{folder_path}/{today}_{fixed_model_name}_PDDL_Multi_Step_{goal_type}_{trial}.txt"
                
                if os.path.exists(file_name): # retry == 1 and 
                    open(file_name, 'w').close()  # empty file
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                trial_record = []
                
                env = TextWorldExpressEnv(envStepLimit=100)
                env.load(**ENV_PARAMS)
                obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                valid_actions = _sanitize_valid_actions(infos)
 
                # If coin is visible in the very first observation, skip this seed and
                # extend the window so we still complete the requested number of meaningful trials.
                if "coin" in obs.lower():
                    print(f"[Seed {trial}] Coin found at the beginning — skipping this seed and extending the trial window.")
                    adjusted_end += 1
                    break  # break out of the retry loop; we'll advance to the next seed below
                with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
                    f.write(f"Observations: {obs} \n") 
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {valid_actions} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")


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
                end_game = False

                overall_memory = brief_obs

                for step_id in range(MAX_STEPS):
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
                            env = TextWorldExpressEnv(envStepLimit=100)
                            env.load(**ENV_PARAMS)
                            obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True) # 여기서 나온 obs, infos는 쓰지도 않음
                            for successful_action in successful_actions: # 초기화 다음, 어떤 행동을 취하기 시작한 후부터의 아웃풋에 주목
                                obs, reward, done, infos = env.step(successful_action) # <-아... 앞단계에서 검증된거 한꺼번에 넣어주고 다음 action찾으려고
                            valid_actions = _sanitize_valid_actions(infos) # 이거 오리지널스크립트에서는 업데이트 안함.이유를 알았음. 어차피 안변함

                        action_queue = [] # reset action_queue ()
                        tem_action_queue = []
                        tem_memory = ""

                        executed_actions = []
                        executed_memory = ""

                        start_checkpoint = True
                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f'Small Loop, action_queue: {action_queue} \n')
                            start_checkpoint = False

                            if not action_queue: # when start_checkpoint = True before. 이 시점에는 무조건 false긴 하지만 그래도 이 while문 들어올 때
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                action = ""
                                
                                if not df and not pf: # First step no need duplicates detection
                                    num_tries = 0

                                    first_generation_tries = 0

                                    while ((not df) or (not pf)) and first_generation_tries < 3: # if one of them is empty, regenerate both
                                        prompt_df = prompt_format_df.format() + prompt_df_generation.format(brief_obs=brief_obs, valid_actions=valid_actions)
                                        resp1 = run_llm_json(prompt_df, model_name, system_prompt=SYS_PROMPT_PDDL)
                                        df = resp1.get("df", None)
                                        # print(f"First generation try {first_generation_tries}, raw response: {resp1}, df: {df}")

                                        prompt_pf_init = prompt_format_pf_init.format() + prompt_pf_init_generation.format(brief_obs=brief_obs, valid_actions=valid_actions, df=df)
                                        resp2 = run_llm_json(prompt_pf_init, model_name, system_prompt=SYS_PROMPT_PDDL)
                                        pf_init = resp2.get("pf_objects_and_init", None)
                                        # print(f"First generation try {first_generation_tries}, raw response: {resp2}, pf_init: {pf_init}")

                                        prompt_pf_complete = prompt_format_pf_complete.format() + prompt_pf_complete_generation.format(brief_obs=brief_obs, valid_actions=valid_actions, df=df, pf_init=pf_init)
                                        resp3 = run_llm_json(prompt_pf_complete, model_name, system_prompt=SYS_PROMPT_PDDL)
                                        pf = resp3.get("pf", None)
                                        # print(f"First generation try {first_generation_tries}, raw response: {resp3}, pf: {pf}")

                                        with open(file_name, "a") as f:
                                            f.write(f"--First Generation Try--: {first_generation_tries} \n")
                                            f.write(f"Prompt DF: {prompt_df} \n") 
                                            f.write(f"Generated df: \n {df} \n") 
                                            f.write(f"Prompt PF init: {prompt_pf_init} \n") 
                                            f.write(f"Generated pf_init: \n {pf_init} \n") 
                                            f.write(f"Prompt PF complete: {prompt_pf_complete} \n") 
                                            f.write(f"Generated pf_complete: \n {pf} \n")

                                        first_generation_tries += 1   

                                    action, err, plan_text = get_action_from_pddl(df, pf) # error 2 here

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, False)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1
                                        
                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")
                                else:
                                    num_tries = 0
                                    # Every time read new error message from larger loop
                                    # In llm_to_pddl, detect if new large loop error message exists
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, "", detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message) # need to add new error message
                                    action, err, plan_text = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")

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
                            valid_actions = _sanitize_valid_actions(infos)

                            executed_actions.append(taken_action)
                            brief_obs = "Action: " + taken_action + "\n" + summarize_obs(obs)+'\n'
                            executed_memory += brief_obs 

                            # Directly end the game if found coin
                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                valid_actions = _sanitize_valid_actions(infos)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write('Coin found!\n')
                                    f.write(f"Final obs: {obs} \n")
                                    coin_found = True
                                break
                            
                            # action_text = "Action: " + taken_action + "\n"
                            # obs_text = summarize_obs(obs) + "\n"

                            # brief_obs = action_text + obs_text

                            obs_queue.append(brief_obs)
                            with open(file_name, "a") as f:
                                f.write(f"> {taken_action} \n {obs} \n")

                            large_loop_error_message, _code = map_env_feedback_to_large_loop_error(brief_obs, taken_action)
                            # If you want to go one step further later, you can use the returned _code
                            if _code:
                                # action_passed = False
                                if executed_actions:
                                    successful_actions.extend(executed_actions[:-1]) # except the last one that caused error
                                    overall_memory += executed_memory[:-len(brief_obs)] # except the last one that caused error
                                action_passed = False
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
                duration = time.time() - start_time
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
                    model_type = "PDDL_Multi_Step"
                    data_row = [today, model_name, model_type, goal_type, trial, coin_found, duration, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    # [today, model_name, trial, coin_found, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    writer = csv.writer(csvfile)
                    writer.writerow(data_row)
                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    log_message = (
                        f"[PDDL_Multi_Step] Trial {trial} (Attempt {retry+1}) | "
                        f"Model: {model_name} | Goal Type: {goal_type} | "
                        f"Failed: {str(e)}\n"
                    )
                    f.write(log_message)
                retry += 1
        trial += 1        

# =========================

def run_baseline_model(model_name, start_trials, end_trials, folder_name="08_031825_alfworld", result_name="alfworld_results"):
    # for trial in range(start_trials, end_trials):
    adjusted_end = end_trials
    trial = start_trials
    while trial < adjusted_end:
        start_time = time.time()
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
                obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                valid_actions = _sanitize_valid_actions(infos)
           
                # If coin is visible in the very first observation, skip this seed and
                # extend the window so we still complete the requested number of meaningful trials.
                if "coin" in obs.lower():
                    print(f"[Seed {trial}] Coin found at the beginning — skipping this seed and extending the trial window.")
                    adjusted_end += 1
                    break  # break out of the retry loop; we'll advance to the next seed below
                with open(file_name, "a") as f:
                    f.write(f"Observations: {obs} \n")
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {valid_actions} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")

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
                            obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                            for act in successful_actions:
                                obs, reward, done, infos = env.step(act)
                            valid_actions = _sanitize_valid_actions(infos)

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
                            valid_actions = _sanitize_valid_actions(infos)

                            # Immediately end the game if coin is found.
                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                valid_actions = _sanitize_valid_actions(infos)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write("Coin found!\n")
                                    f.write(f"Final obs: {obs} \n")
                                coin_found = True
                                break

                            action_text = "Action: " + taken_action + "\n"
                            obs_text = summarize_obs(obs) + "\n"
                            brief_obs = action_text + obs_text
                            obs_queue.append(brief_obs)
                            with open(file_name, "a") as f:
                                f.write(f"> {taken_action} \n {obs} \n")
                            
                            msg, _code = map_env_feedback_to_large_loop_error(brief_obs, taken_action)
                            if msg:
                                large_loop_error_message = msg
                                # large_loop_error_message = obs
                                with open(file_name, "a") as f:
                                    f.write(f"Large loop error message: {large_loop_error_message} \n")
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

                duration = time.time() - start_time
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    # Write out: date, model_name, trial, coin_found, last step index, last large-loop iteration, and the full trial record.
                    # data_row = [today, model_name, trial, coin_found, len(trial_record)-1, trial_record[-1] if trial_record else None, trial_record]
                    model_type = 'baseline' # PDDL
                    goal_type = 'detailed' # detailed or subgoal
                    data_row = [today, model_name, model_type, goal_type, trial, coin_found, duration, len(trial_record)-1,trial_record[-1][-1], trial_record]

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
        trial += 1


# =========================




i = 0
num_trials = 20
# folder_name = "yewon_coin_1006_finalPrompt2"  # when _code detected, action_passed = False comment out
folder_name = "yewon_coin_1008_finalPrompt3"  # executed actions and memory added when _code detected


result_name = folder_name
model_id1 = "meta-llama/Meta-Llama-3-70B-Instruct"
model_id2 = "Qwen/Qwen3-32B"
model_id3 = "Qwen/Qwen3-8B"
openai_model = "gpt-5-nano-2025-08-07" # most cost-efficient



run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
run_iterative_model2(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
run_baseline_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name)

clear_cuda_memory(model_id2)
