SYS_PROMPT_PLAN = """
You will be given a naturalistic domain description and problem description.
Your task is to generate a plan (a series of actions).
"""
SYS_PROMPT_PDDL = """
You will be given a naturalistic domain description and problem description. 
Your task is to generate domain file and problem file in Planning Domain Definition Language (PDDL) with appropriate tags.
"""
SYS_PROMPT_PYIR = (
    "Generate a class-based Python IR for PDDL. "
    "Output strict JSON with keys py_domain and py_problem only."
)
##############################################
prompt_format = """
    Please provide the output in strict JSON format, without any additional text or explanation, including a PDDL domain file as 'df' and a PDDL problem file as 'pf'. 
    The format should strictly be:
        {{
        "df": "...",
        "pf": "..."
        }}

"""

prompt_obs_action_subgoal = """
""" 

prompt_obs_action_detailed = """
You are in a partially observable environment that you explore step by step. You must build and update PDDL files of the environment based on only your observations. 
Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
Do not assume that there will be a door connecting rooms.
Your task is always to keep exploration and go to a location you have not visited yet.
In other words, your goal should go to other not visited location.
If you enter a room, make sure you put everything you observed such as the direction in the problem file.
Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}
You should generate df and pf that strictly follow these action schemas. There are in total 2 actions, that should exactly be the following two:
1. :action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
2. :action move
    :parameters (?from - location ?to - location ?dir - direction)
You should have a goal in the problem file like this: 
(:goal 
    (at ?location)
) 


Note: 
- In problem file's init, you shouldn't have "not ()" but only the single status.
- In problem file's goal, ?location must be grounded, no variables or quantifiers.

""" 
#####################################################

prompt_prev_files = """
This is all the memory you have in this game including each action and its corresponding observations: 
{overall_memory}

You have already generated df and pf files according to the observations.
This is previous domain file: 
{prev_df}

This is previous problem file: 
{prev_pf}

"""

prompt_new_obs = """
Now modify those domain and problem files according to the new observations and notes. Fix any errors you made in the previous setting according to the new observation.
Generate updated files based on your new observation.

"""


prompt_error_parser = """
You made some mistakes when generating those files. Here is the error message: 
{prev_err}

"""

# error from simulation environment
prompt_simulation_error = """
Based on the df and pf that you generated, the external solver could generate a plan but after simulating in the game environment, it caused those errors: 
{large_loop_error_message}

"""

prompt_duplicate_note = """
You are repeating the same sequence of actions for at least three times. You may stuck in one location or have the wrong goal.
You should revise your problem file to avoid the repeat.
Remember your goal is always to keep exploration: this means prioritizing opening closed doors and moving to unvisited known locations. Ensure your goal reflects this intent.

"""

#####################################################

prompt_baseline = """
You are in an environment that you explore step by step. Based on your observations, generate a series of valid actions to progress in the environment.
Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}
Your goal is to explore new locations and interact with the environment effectively. Ensure actions are logical and do not repeat unnecessarily.

Additional context:
{overall_memory}

If there are errors or obstacles, here is the message:
{large_loop_error_message}

Provide the output in strict JSON format like this, while you should only generate one action at a time:
{{
    "actions": ["action1"]
}}

"""


# =========================
# [2] Multi-step PF generation
# =========================
prompt_format_df = """
Please provide the output in strict JSON format, without any additional text or explanation. 
The format should strictly be:
{{
  "df": "..."
}}

"""

prompt_df_generation = """
You are in a partially observable environment that you explore step by step. Your task is to generate a PDDL domain file ('df') based on the observations and valid actions.
Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
Do not assume that there will be a door connecting rooms.

Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

You should generate a df that strictly follows these valid actions. There are in total 2 actions, that should exactly be the following two:
1. :action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
2. :action move
    :parameters (?from - location ?to - location ?dir - direction)

"""

prompt_format_pf_init = """
Please provide the output in a strict JSON format, without any additional text or explanation.
The format should strictly be:
{{
  "pf_objects_and_init": "..."
}}

"""

prompt_pf_init_generation = """
You are in a partially observable environment that you explore step by step. Your task is to define the objects and the initial state for a PDDL problem file ('pf') based on the provided domain file ('df') and your current observations from the environment.
**DO NOT** generate the `(:goal ...)` section in this step. 

Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
Do not assume that there will be a door connecting rooms.

If you enter a room, make sure you put everything you observed such as the direction in the problem file.

This is the domain file: {df}
    
Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

Note: 
- In problem file's init, you shouldn't have "not ()" but only the single status.

"""

prompt_format_pf_complete = """
Please provide the output in strict JSON format, without any additional text or explanation. 
The format should strictly be:
{{
  "pf": "..."
}}

"""

prompt_pf_complete_generation = """
You are in a partially observable environment that you explore step by step. Your output must be one single, complete PDDL problem file. To create it, add a `(:goal ...)` section to the provided objects and initial state, then wrap everything in the standard `(define (problem ...))` structure.

Your task is always to keep exploration and go to a location you have not visited yet.
In other words, your goal should be to go to another not visited location.

This is the domain file:
{df}

Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

This is the objects and initial state of the problem file:
{pf_init}

You should have a goal in the problem file like this: 
(:goal 
    (at ?location)
)


Note: 
- In problem file's goal, ?location must be grounded, no variables or quantifiers.

"""



