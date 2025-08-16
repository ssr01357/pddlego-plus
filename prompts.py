prompt_format = """
Please provide the output in strict JSON format, without any additional text or explanation, including a PDDL domain file as 'df' and a PDDL problem file as 'pf'. 
The format should strictly be:
    {{
    "df": "...",
    "pf": "..."
    }}

"""

#####################################################

prompt_edit = """
    Please provide the output in JSON format, including the edit suggestions for a domain file as 'df' and the edit suggestions for a problem file as 'pf'. 
    The output format should be: {{"df": "...", "pf": "..."}}
    You will modify the following df and pf using add, delate, and replace operations (in a JSON format). 
    You SHOULD NOT provide a domain file and a problem file directly.
    This is the structure for df edit file, remember to add bracket:
    {{
    "predicates": {{
        "add": ["(predicates to add)"],
        "replace": {{"(old)": "(new)"}},
        "delete": ["(predicates to delete)"]
        }},
    "action": {{
        "open-door": {{
            "precondition": ["(entire full new precondition for open-door)"], # directly replace the whole precondition
            "effect": ["(entire full new effect for open-door)"] # so as effect
            }},
        "move": {{
            "precondition": []
            "effect": []
            }}
        }}
    }}
    This is the structure for pf edit file:
    {{
    "objects": {{
        "add": [],
        "replace": {{}},
        "delete": []
        }},
    "init": {{
        "add": [],
        "replace": {{}},
        "delete": []
        }},
    "goal": ["(entire full new goal)"]
    }}
"""

prompt_obs_action_subgoal = """
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

prompt_obs_action_detailed = """
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
#####################################################

prompt_prev_files = """
This is all the memory you have in this game including each action and its corresponding observations: 
{overall_memory}

You have already generate df and pf files according to the observations.
This is previous domain file: 
{prev_df}

This is previous problem file: 
{prev_pf}

"""

#####################################################
prompt_new_obs = """
Now modify those two files according to the new observations and notes. Fix any errors you made in the previous setting according to the new observation.
Generate updated files based on your new observation.

"""

# error from Parser(df, pf)
prompt_error_parser = """
You made some mistakes when generating those files. Here is the error message: 
{prev_err}
"""

#####################################################
# error from simulation environment
prompt_simulation_error = """
Based on the df and pf that you generated, the external solver could generate actions but after simulating in the game environment, it caused those errors: 
{large_loop_error_message} 
"""

#####################################################
prompt_duplicate_note = """
You are repeating the same sequence of actions for at least three times. You may stuck in one location or have the wrong goal.
You should revise your problem file to avoid the repeat.
Remember your goal is always to keep exploration and go to a location you have not visited yet, i.e. your goal should be go to other not visited location but shouldn't be at one fixed location.
"""

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