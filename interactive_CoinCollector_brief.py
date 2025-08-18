import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
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
from prompts_CoinCollector import *
from solver import run_solver

_hf_engine_cache: dict[str, HuggingEngine] = {}

OPENAI_MODELS_LIST = ['gpt-4o','o3-mini',"gpt-4.1","o4-mini", "gpt-5-nano-2025-08-07"]
ENV_PARAMS = {"gameName": "coin", "gameParams": "numLocations=11,numDistractorItems=0,includeDoors=1,limitInventorySize=0"}

def _tail(s, n=4000):
    return s if not s or len(s) <= n else s[-n:]


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

# def map_actions(plan_text):
#     lines = [l.strip().lower() for l in plan_text.splitlines() if l.strip()]
#     action_lst = []
#     dirs = {"north","south","east","west","n","s","e","w"}
#     def norm(d):
#         return {"n":"north","s":"south","e":"east","w":"west"}.get(d,d)

#     for l in lines:
#         # strip parens and split
#         toks = re.findall(r"[a-z0-9_-]+", l)
#         if not toks: 
#             continue
#         name = toks[0]
#         # prefer the enforced names
#         if name in ("open-door", "open_door", "open"):
#             # pick last token that looks like a direction
#             dir_tok = next((norm(t) for t in reversed(toks) if t in dirs), None)
#             if dir_tok:
#                 action_lst.append(f"open door to {dir_tok}")
#         elif name in ("move","go","walk"):
#             dir_tok = next((norm(t) for t in reversed(toks) if t in dirs), None)
#             if dir_tok:
#                 action_lst.append(f"move {dir_tok}")
#     return action_lst or None


def map_env_feedback_to_large_loop_error(brief_obs: str, taken_action: str):
    """
    Map environment feedback in `brief_obs` (and the `taken_action`) to a
    single large_loop_error_message. Returns (message, code) or (None, None)
    if no actionable error was detected.
    """
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
                "You cannot move to that direction. Review the predicate of your actions and the problem files to check the status."
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



    # df, pf = run_llm(prompt, model_name)
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
                if "coin" in obs.lower():
                    print("Coin found at the beginning!")
                    coin_found = True
                    exit(0)
                with open(file_name, "a") as f:  # "w" creates a new file or overwrites an existing file
                    f.write(f"Observations: {obs} \n") 
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {infos['validActions']} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")
                # 이건 첫번째 tries(=larger loop에서 쓰임)

                # task_description = infos['taskDescription']
                valid_actions = sorted(infos['validActions'])
                valid_actions.remove('look around') # 이건 초반에 항상 초기값
                valid_actions.remove('inventory') # 이건 필요없으니까?

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

                            # Directly end the game if found coin
                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write('Coin found!\n')
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
                            # If you want to go one step further later, you can use the returned _code
                            if msg:
                                large_loop_error_message = msg
                                # large_loop_error_message = obs
                                with open(file_name, "a") as f:
                                    f.write(f"Large loop error message: {large_loop_error_message} \n")
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
                if "coin" in obs.lower():
                    print("Coin found at the beginning!")
                    coin_found = True
                    exit(0)
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



def llm_to_pyir(model_name, brief_obs, valid_actions, history = None):
    prompt = prompt_pyir.format(pypddl_instruction=pypddl_instruction, brief_obs=brief_obs, valid_actions=valid_actions)

    if history:
        prompt += f"""
        ### Previous IR and PDDL history:
        {history}
        ### End of history.

        Now, based on the above history and your current observation, update the Python representation of PDDL files. 
        """

    resp = run_llm_json(prompt, model_name, system_prompt=SYS_PROMPT_PYIR)
    py_domain = resp.get("py_domain", "")
    py_problem = resp.get("py_problem", "")
    return py_domain, py_problem, prompt




def llm_pyir_to_pddl(model_name, py_domain, py_problem, brief_obs, valid_actions,
                     prev_df="", prev_pf="", prev_err="", large_loop_error_message=""):

    prev_pddl = ""
    if prev_df and prev_pf:
        prev_pddl = f"[Domain file]\n{prev_df}\n\n[Problem file]\n{prev_pf}"

    prompt = prompt_pyir2pddl.format(
        py_domain=py_domain,
        py_problem=py_problem,
        brief_obs=brief_obs,
        valid_actions=valid_actions,
        prev_pddl=prev_pddl or "N/A",
        prev_err=prev_err or "N/A",
        env_err=large_loop_error_message or "N/A"
    )

    resp = run_llm_json(prompt, model_name, system_prompt=SYS_PROMPT_PDDL)
    df = resp.get("df", None)
    pf = resp.get("pf", None)
    return df, pf, prompt

# =========================
# [3] Python IR → PDDL: runner
# =========================

def run_pyir_model(model_name, start_trial=0, end_trial=11, folder_name="3_0421_CC", result_name="CC_results", goal_type="detailed"):
    for trial in range(start_trial, end_trial):
        retry = 0
        while retry < 2:
            try:
                coin_found = False
                today = date.today()
                fixed_model_name = model_name.replace("/", "_")

                folder_path = f"output/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_name = f"{folder_path}/{today}_{fixed_model_name}_PyIR_{goal_type}_{trial}.txt"
                if os.path.exists(file_name):
                    open(file_name, 'w').close()
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                trial_record = []

                env = TextWorldExpressEnv(envStepLimit=100)
                env.load(**ENV_PARAMS)
                obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
                if "coin" in obs.lower():
                    print("Coin found at the beginning!")
                    coin_found = True
                    exit(0)
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
                df = ""
                pf = ""
                py_domain = ""
                py_problem = ""
                all_actions = []
                successful_actions = []
                end_game = False
                overall_memory = brief_obs

                for step_id in range(0, MAX_STEPS):
                    with open(file_name, "a") as f:
                        f.write(f"\n\n====Step {step_id}==== \n")

                    trial_step_record = []
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""

                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a") as f:
                            f.write(f"\n----Larger Loop No. {within_step_tries}---- \n")
                            f.write(f"successful_actions: {successful_actions} \n")

                        within_step_tries += 1

                        if within_step_tries > 1:
                            env = TextWorldExpressEnv(envStepLimit=100)
                            env.load(**ENV_PARAMS)
                            obs, infos = env.reset(seed=1, gameFold="train", generateGoldPath=True)
                            for act in successful_actions:
                                obs, reward, done, infos = env.step(act)

                        action_queue = []
                        tem_action_queue = []
                        tem_memory = ""

                        start_checkpoint = True
                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f"Small Loop, action_queue: {action_queue} \n")
                            start_checkpoint = False

                            if not action_queue: # when start_checkpoint = True
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                
                                action = None
                                num_tries = 0
                                pddl_err = "" # To store the planner's stderr

                                while not action and num_tries < 5:
                                    # 1. generate or refresh Python IR
                                    # refresh it on the first try, or if PDDL generation fails multiple times (tries 3 and 5).
                                    if num_tries in [0, 2, 4]:
                                        history_blob = ""
                                        if py_domain or py_problem or df or pf:
                                            history_blob = (
                                                "### Previous Python IR (edit minimally if possible)\n"
                                                f"[py_domain]\n{py_domain}\n\n[py_problem]\n{py_problem}\n\n"
                                                "### Previous PDDL (generated from the IR)\n"
                                                f"[df]\n{df}\n\n[pf]\n{pf}\n"
                                                "### Planner Error from previous PDDL (if any)\n"
                                                f"{pddl_err or 'N/A'}\n"
                                                "### Environment feedback from previous action (if any)\n"
                                                f"{large_loop_error_message or 'N/A'}\n"
                                            )
       
                                        py_domain, py_problem, pyir_prompt = llm_to_pyir(
                                            model_name, brief_obs, valid_actions, history=history_blob
                                        )
                                        with open(file_name, "a") as f:
                                            f.write(f"[PyIR Prompt] {pyir_prompt}\n")
                                            f.write(f"Generated py_domain:\n{py_domain}\n")
                                            f.write(f"Generated py_problem:\n{py_problem}\n")

                                    # 2. convert the latest Python IR to PDDL
                                    df, pf, conv_prompt = llm_pyir_to_pddl(model_name, py_domain, py_problem, brief_obs, valid_actions,
                                                                             prev_df=df, prev_pf=pf, prev_err=pddl_err, large_loop_error_message=large_loop_error_message
                                                                             )
                                    # 3. get actions from the PDDL files
                                    action, pddl_err, plan_text = get_action_from_pddl(df, pf)
                                    with open(file_name, "a") as f:
                                        f.write(f"\n--- Attempting to Plan (Small Loop Try #{num_tries + 1}) ---\n")
                                        f.write(f"[IR→PDDL Prompt] {conv_prompt} \n")
                                        f.write(f"[PDDL df]\n{df}\n\n[PDDL pf]\n{pf}\n")
                                        f.write(f"Actions from solver: {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")
                                        if pddl_err:
                                            f.write(f"Solver stderr:\n{pddl_err}\n")
                                    num_tries += 1
                            

                                trial_step_record.append([within_step_tries, num_tries])

                                if action:
                                    action_queue.extend(action)
                                    tem_action_queue.extend(action)
                                    all_actions.extend(action)
                                else:
                                    end_game = True
                                    break

                            # Execute one action
                            with open(file_name, "a") as f:
                                f.write(f"Current action_queue: {action_queue} \n")

                            taken_action = action_queue.pop(0)
                            obs, reward, done, infos = env.step(taken_action)

                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write('Coin found!')
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
                                with open(file_name, "a") as f:
                                    f.write(f"Large loop error message: {large_loop_error_message} \n")
                                break

                            tem_memory += brief_obs

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
                    model_type = "PyIR"
                    data_row = [today, model_name, model_type, goal_type, trial,
                                coin_found, len(trial_record)-1, trial_record[-1][-1], trial_record]
                    writer = csv.writer(csvfile)
                    writer.writerow(data_row)
                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    log_message = (
                        f"[PyIR] Trial {trial} (Attempt {retry+1}) | "
                        f"Model: {model_name} | Goal Type: {goal_type} | "
                        f"Failed: {str(e)}\n"
                    )
                    f.write(log_message)
                retry += 1


i = 0
num_trials = 5
# folder_name = "CC_o4_mini_high"
folder_name = "yewon_coin_0818_easier"
# 1

result_name = folder_name
model_id1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_id2 = "Qwen/Qwen3-32B"
model_id3 = "meta-llama/Llama-3.3-70B-Instruct"
openai_model = "gpt-4.1"
## Run PlanGen models
# run_baseline_model(model_id1, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# clear_cuda_memory(model_id1)
# run_baseline_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# clear_cuda_memory(model_id2)
# run_baseline_model(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name)
# clear_cuda_memory(model_id3)
# run_baseline_model(openai_model, i, i+num_trials, folder_name=folder_name, result_name=result_name)

## Run PDDLego+ models
# run_iterative_model(model_id1, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id1)
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id2)
# run_iterative_model(model_id3, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# clear_cuda_memory(model_id3)
# run_iterative_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# clear_cuda_memory(model_id2)
# run_iterative_model("o3-mini-2025-01-31", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model(openai_model, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
# run_iterative_model("gpt-4o", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="subgoal")
# run_iterative_model("o4-mini-2025-04-16", i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")

run_pyir_model(openai_model, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")
run_pyir_model(model_id2, i, i+num_trials, folder_name=folder_name, result_name=result_name, goal_type="detailed")