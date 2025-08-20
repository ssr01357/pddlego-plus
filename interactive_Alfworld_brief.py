import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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
from utils import repair_json
from prompts_Alfworld import *
from solver import run_solver

import torch
import gc

_hf_engine_cache: dict[str, HuggingEngine] = {}

OPENAI_MODELS_LIST = ['gpt-4o','o3-mini',"gpt-4.1","o4-mini", "gpt-5-nano-2025-08-07"]


# ===== Shared config =====
MAX_EPISODE_STEPS = 1_000_000

PROBLEM_TYPE = {  
    0: 'clean', 1: 'basic', 2: 'basic', 3: 'slice & heat', 4: 'heat',
    5: 'use', 6: 'clean', 7: 'use', 8: 'basic', 9: 'cool'
}

# Scan problems once (you already had this)
PROBLEMS = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
PROBLEMS = [p for p in PROBLEMS if "movable_recep" not in p]
if len(PROBLEMS) == 0:
    raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")

# Read base logic once and reuse
DOMAIN_PATH = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
GRAMMAR_PATH = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
with open(DOMAIN_PATH, "r") as _f:
    BASE_DOMAIN = _f.read()
with open(GRAMMAR_PATH, "r") as _f:
    BASE_GRAMMAR = _f.read()

# Request infos can be reused across envs
REQUEST_INFOS = textworld.EnvInfos(
    won=True,
    admissible_commands=True,
    score=True,
    max_score=True,
    intermediate_reward=True,
    extras=["expert_plan"]
)
def prepare_problem(problem_id: int):
    """
    Build game for a given problem_id.
    Returns:
        env, expert, meta dict, and initial (init_obs, goal, valid_actions)
    """
    problem_dir = os.path.dirname(PROBLEMS[problem_id])
    pddl_file = os.path.join(problem_dir, "initial_state.pddl")
    json_file = os.path.join(problem_dir, "traj_data.json")

    with open(json_file, "r") as f:
        traj_data = json.load(f)

    # grammar is per-problem (task injected); do not cache this mutation
    grammar = add_task_to_grammar(BASE_GRAMMAR, traj_data)

    # write the combined .tw-pddl (same pattern you use everywhere)
    gamefile = os.path.join(problem_dir, "game.tw-pddl")
    gamedata = {
        "pddl_domain": BASE_DOMAIN,
        "grammar": grammar,
        "pddl_problem": open(pddl_file).read()
    }
    json.dump(gamedata, open(gamefile, "w"))

    # New expert per env is safest
    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

    env_id = textworld.gym.register_game(
        gamefile,
        REQUEST_INFOS,
        max_episode_steps=MAX_EPISODE_STEPS,
        wrappers=[AlfredDemangler(), expert]
    )
    env = textworld.gym.make(env_id)

    # Initial state
    obs, infos = env.reset()
    init_obs = obs.split('\n')[2]
    goal = obs.split('\n')[-1]
    valid_actions = filter_valid_actions(infos["admissible_commands"])

    meta = {
        "problem_id": problem_id,
        "problem_dir": problem_dir,
        "gamefile": gamefile,
        "game_type": PROBLEM_TYPE.get(problem_id, "unknown")
    }
    return env, expert, meta, init_obs, goal, valid_actions

def filter_valid_actions(cmds):
    """Remove meta commands you never execute."""
    return sorted(set(cmds) - {"look", "inventory", "help"})

def reset_env_with_prefix(gamefile: str, successful_actions: list[str], expert: AlfredExpert):
    """Recreate env, reset, then replay successful prefix actions."""
    env_id = textworld.gym.register_game(
        gamefile,
        REQUEST_INFOS,
        max_episode_steps=MAX_EPISODE_STEPS,
        wrappers=[AlfredDemangler(), expert]
    )
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()
    for act in successful_actions:
        obs, _, _, infos = env.step(act)
    return env, obs, infos

def make_output_file(folder_name, today, model_name, tag, trial, goal_type=None, game_type=None):
    """Uniform file naming for all runs."""
    fixed_model_name = model_name.replace("/", "_")
    folder_path = f"output/{folder_name}"
    os.makedirs(folder_path, exist_ok=True)
    parts = [str(today), fixed_model_name, tag]
    if goal_type: parts.append(goal_type)
    if game_type: parts.append(game_type)
    parts.append(str(trial))
    return f"{folder_path}/{'_'.join(parts)}.txt"


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


def llm_to_pddl(model_name, brief_obs, valid_actions, goal, prev_df="", prev_pf="", prev_err="", have_duplicate=False, overall_memory=None, large_loop_error_message=None, goal_type='detailed'):
    prompt = prompt_format.format()

    # all prompts should have observations and actions
    # select which prompt version to use: prompt_obs_action_detailed or prompt_obs_action_general_goal or prompt_obs_action_subgoal
    # default is detailed
    if goal_type == 'detailed':
        prompt += prompt_obs_action_detailed.format(goal=goal, brief_obs=brief_obs, valid_actions=valid_actions)
    elif goal_type == 'subgoal':
        prompt += prompt_obs_action_subgoal.format(goal=goal, brief_obs=brief_obs, valid_actions=valid_actions)
    elif goal_type == 'without_hint':
        prompt += prompt_obs_action_without_hint.format(goal=goal, brief_obs=brief_obs, valid_actions=valid_actions)
    elif goal_type == 'without_detailed_goal':
        prompt += prompt_obs_action_without_detailed_goal.format(goal=goal, brief_obs=brief_obs, valid_actions=valid_actions)
    else:
        prompt += prompt_obs_action_general_goal.format(goal=goal, brief_obs=brief_obs, valid_actions=valid_actions)

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


def llm_to_actions_baseline(model_name, brief_obs, valid_actions, goal, overall_memory=None, large_loop_error_message=None):
    prompt = prompt_baseline.format(
        goal=goal or "Explore and interact meaningfully based on available observations.",
        brief_obs=brief_obs,
        valid_actions=valid_actions,
        overall_memory=overall_memory or "N/A",
        large_loop_error_message=large_loop_error_message or "N/A",
    )

    actions = run_llm_json(prompt, model_name, system_prompt=SYS_PROMPT_PLAN).get("actions", None)
    return actions




# ===== Main functions here =====
def run_iterative_model(model_name, start_trial = 0, end_trial = 11, folder_name="08_031825_alfworld", result_name="alfworld_results", goal_type="detailed"):
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
                env, expert, meta, init_obs, goal, valid_actions = prepare_problem(problem_id)
                game_type = meta["game_type"]

                today = date.today()
                file_name = make_output_file(
                    folder_name, today, model_name,
                    tag=f"PDDL_{goal_type}", trial=trial, goal_type=goal_type
                )

                with open(file_name, "a") as f:
                    f.write(f"Playing {problem_id}: {meta['problem_dir']}\n")
                    f.write(f"Observations: {init_obs}\n")
                    f.write(f"Valid Actions: {valid_actions}\n")
                    f.write(f"taskDescription: {goal}\n")

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
                            env, obs, infos = reset_env_with_prefix(meta["gamefile"], successful_actions, expert)
                            goal = obs.split('\n')[-1]

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
                                    # df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type) # error 1 here
                                    current_valid_actions = filter_valid_actions(infos["admissible_commands"])
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, current_valid_actions, goal, goal_type=goal_type)
                                    action, err_2, plan_text = get_action_from_pddl(df, pf) # error 2 here
                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        # df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type, df, pf, err, err_2, True, False, edit)
                                        current_valid_actions = filter_valid_actions(infos["admissible_commands"])
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, current_valid_actions, goal, prev_df=df, prev_pf=pf, prev_err=err_2, have_duplicate=False, overall_memory=overall_memory, large_loop_error_message=large_loop_error_message, goal_type=goal_type)
                                        action, err_2, plan_text = get_action_from_pddl(df, pf)
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
                                    # df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type, df, pf, err, None, False, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message) # need to add new error message
                                    current_valid_actions = filter_valid_actions(infos["admissible_commands"])
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, current_valid_actions, goal, prev_df=df, prev_pf=pf, prev_err="", have_duplicate=detect_duplicates(all_actions, 3), overall_memory=overall_memory, large_loop_error_message=large_loop_error_message, goal_type=goal_type)
                                    action, err_2, plan_text = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        # df, pf, err, prompt = llm_to_pddl(model_name, brief_obs, goal, goal_type, df, pf, err, err_2, True, detect_duplicates(all_actions, 3), edit, overall_memory, large_loop_error_message)
                                        current_valid_actions = filter_valid_actions(infos["admissible_commands"])
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, current_valid_actions, goal, prev_df=df, prev_pf=pf, prev_err=err_2, have_duplicate=False, overall_memory=overall_memory, large_loop_error_message=large_loop_error_message, goal_type=goal_type)
                                        action, err_2, plan_text = get_action_from_pddl(df, pf)
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
                                f.write(f"After taking action '{taken_action}', valid actions: {filter_valid_actions(infos['admissible_commands'])} \n")


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
                env, expert, meta, init_obs, goal, valid_actions = prepare_problem(problem_id)
                game_type = meta["game_type"]

                today = date.today()
                file_name = make_output_file(folder_name, today, model_name, tag="baseline", trial=trial)

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
                            env, obs, infos = reset_env_with_prefix(meta["gamefile"], successful_actions, expert)
                            goal = obs.split('\n')[-1]
                            valid_actions = filter_valid_actions(infos["admissible_commands"])

                        current_valid_actions = filter_valid_actions(infos["admissible_commands"])
                        actions, prompt = llm_to_actions_baseline(
                            model_name,
                            brief_obs,
                            current_valid_actions,
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

                        while action_queue:
                            act = action_queue.pop(0)
                            obs, reward, done, infos = env.step(act)
                            action_text = "Action: " + act + "\n"
                            obs_text = summarize_obs(obs) + "\n"
                            brief_obs = action_text + obs_text
                            tem_memory += brief_obs
                            all_actions.append(act)

                            with open(file_name, "a") as f:
                                f.write(f"> {act}\n{brief_obs}\n")
                                f.write(f"After action '{act}', valid actions: {filter_valid_actions(infos['admissible_commands'])}\n")
                                

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
