# ====================================================================================
# 변경 없는 프롬프트들
# ====================================================================================

SYS_PROMPT_PLAN = """
You will be given a naturalistic domain description and problem description.
Your task is to generate a plan (a series of actions).
"""
SYS_PROMPT_PDDL = """
You will be given a naturalistic domain description and problem description. 
Your task is to generate domain file and problem file in Planning Domain Definition Language (PDDL) with appropriate tags.
"""
# SYS_PROMPT_PYIR은 사용하지 않는다면 제거해도 무방합니다.
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

# error from Parser(df, pf)
prompt_error_parser = """
You made some mistakes when generating those files. Here is the error message: 
{prev_err}
"""
#####################################################
# error from simulation environment
prompt_simulation_error = """
Based on the df and pf that you generated, the external solver could generate a plan but after simulating in the game environment, it caused those errors: 
{large_loop_error_message} 
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

# ====================================================================================
# 수정된 프롬프트들 (A안 반영)
# ====================================================================================

# [변경됨] 연결 정보 업데이트 강조
prompt_new_obs = """
Now modify those two files according to the new observations and notes. Fix any errors you made in the previous setting according to the new observation.
Pay close attention if the observation reveals a new location (e.g., "revealing the...") and ensure (connected ...) predicates are correctly added/updated in the problem file based on the discovery.
Generate updated files based on your new observation.
"""

# [변경됨] 목표 설정 전략 재강조
prompt_duplicate_note = """
You are repeating the same sequence of actions for at least three times. You may stuck in one location or have the wrong goal.
You should revise your problem file to avoid the repeat.
Remember your goal is always to keep exploration: this means prioritizing opening closed doors (information gathering goal) or moving to unvisited known locations (movement goal). Ensure your goal reflects this intent.
"""


# [1] PDDL Generation (Single-step)

# [변경됨] 액션 정의 변경, 부분 관찰 가능성 및 Predicate 사용 규칙, 목표 설정 전략 추가
prompt_obs_action_subgoal = """
    You are in a partially observable environment that you explore step by step. You must build and update PDDL files based ONLY on your observations. 
    Do not invent information. Crucially, you do NOT know where a closed door leads until you open it.

    Here are your current observations: {brief_obs}
    Here are some valid actions you can take: {valid_actions}

    You must generate the DF and PF strictly following these action schemas:
    1. :action open-door
        :parameters (?loc - location ?dir - direction)
        ;; Used for discovery. Does NOT require knowing the destination.
    2. :action move
        :parameters (?from - location ?to - location ?dir - direction)
        ;; Requires known connection and open path.

    Key Predicates (must be defined in DF and used in PF):
    - (at ?loc)
    - (has-door ?loc ?dir)
    - (door-is-closed ?loc ?dir) / (door-is-open ?loc ?dir)
    - (connected ?loc1 ?loc2 ?dir): Connection knowledge discovered AFTER observation.

    Exploration Strategy & State Representation:
    Your task is always to explore.
    1. If you see a closed door, use (has-door) and (door-is-closed). Do NOT use (connected).
    2. When an action reveals a new location (e.g., "revealing the laundry room"), you MUST update the PF to include the new location object AND the (connected ...) facts for both directions.
    
    Goal Setting:
    - Prioritize information gathering: open an unexplored door (e.g., (:goal (door-is-open ?current_loc ?dir))).
    - If paths are open and connections known, move to an unvisited location (e.g., (:goal (at ?new_loc))).

    Note: in problem file's init, you shouldn't have "not ()" but only the single status.
""" 

# [변경됨] 액션 정의 변경, 부분 관찰 가능성 및 Predicate 사용 규칙, 상세한 목표 설정 전략 추가
prompt_obs_action_detailed = """
    You are in a partially observable environment that you explore step by step. You must build and update PDDL files based ONLY on your observations. 
    Do not invent information. Crucially, you do NOT know where a closed door leads until you open it.

    Here are your current observations: {brief_obs}
    Here are some valid actions you can take: {valid_actions}

    You must generate the DF and PF strictly following these action schemas:
    1. :action open-door
        :parameters (?loc - location ?dir - direction)
        ;; Used for discovery. Does NOT require knowing the destination.
    2. :action move
        :parameters (?from - location ?to - location ?dir - direction)
        ;; Requires known connection and open path.

    Key Predicates (must be defined in DF and used in PF):
    - (at ?loc)
    - (has-door ?loc ?dir): A door exists at ?loc in ?dir.
    - (door-is-closed ?loc ?dir) / (door-is-open ?loc ?dir): State of the door.
    - (connected ?loc1 ?loc2 ?dir): Connection knowledge discovered AFTER opening a door or observing an open passage.

    Exploration Strategy & State Representation:
    1. If you see a closed door, use (has-door) and (door-is-closed). Do NOT use (connected).
    2. When an action reveals a new location (e.g., "revealing the laundry room"), you MUST update the PF to include the new location object AND the (connected ...) facts for both directions.
    
    Goal Setting (Must be grounded, e.g., (at room_x), no variables or quantifiers):
    - Priority 1: Information Gathering. If there are closed doors at the current location, the goal should be to open one:
      (:goal (door-is-open ?current_loc ?unexplored_dir))
    - Priority 2: Movement. If paths are open and connections known, the goal should be to move to an unvisited location:
      (:goal (at ?unvisited_location))

    Note: in problem file's init, you shouldn't have "not ()" but only the single status.
""" 


# =========================
# [2] Multi-step PF generation (PDDL2 방식)
# =========================
prompt_format_df = """
Please provide the output in strict JSON format, without any additional text or explanation. 
The format should strictly be:
{{
  "df": "..."
}}
"""

# [변경됨] 액션 정의 변경, 필수 Predicate 및 액션 조건 정의 가이드라인 추가
prompt_df_generation = """
You are in a partially observable environment. Your task is to generate a PDDL domain file ('df') based on the observations and valid actions.

Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

You must generate a DF that strictly follows these action schemas. These are the only two actions allowed:
1. :action open-door
    :parameters (?loc - location ?dir - direction)
    ;; Description: Opens a door for discovery. Does not require knowing the destination.
2. :action move
    :parameters (?from - location ?to - location ?dir - direction)
    ;; Description: Moves between known connected locations.

The domain file MUST define the necessary predicates to support these actions, including:
- (at ?loc)
- (has-door ?loc ?dir)
- (door-is-closed ?loc ?dir)
- (door-is-open ?loc ?dir)
- (connected ?loc1 ?loc2 ?dir)

The 'move' action precondition must ensure the connection is known (connected) AND the path is clear (the door is open, or no door exists).
"""

prompt_format_pf_init = """
Please provide the output in a strict JSON format, without any additional text or explanation.
The format should strictly be:
{{
  "pf_objects_and_init": "..."
}}
"""

# [변경됨] 연결 정보(connected) 사용 규칙 및 관찰 해석 가이드 상세화
prompt_pf_init_generation = """
You are in a partially observable environment. Your task is to define the objects and the initial state for a PDDL problem file ('pf') based on the provided domain file ('df') and your current observations.
**DO NOT** generate the `(:goal ...)` section in this step. 

This is the domain file: {df}
    
Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

Instructions for Initial State Definition:
Base the state ONLY on observations. Do not assume connections.

1. Handling Closed Doors: If you observe "a closed [type] door" to the [direction].
   -> Add (has-door ?loc ?dir) and (door-is-closed ?loc ?dir).
   -> CRITICAL: Do NOT add (connected ...) as the destination is unknown.
2. Handling Open Doors: If you observe "Through an open [type] door, to the [direction] you see the [New Room]".
   -> Add (has-door ?loc ?dir), (door-is-open ?loc ?dir).
   -> Add (connected ?loc ?new_room ?dir) AND the reverse connection.
3. Handling Open Passages: If you observe "To the [direction] you see the [New Room]" (no door mentioned).
   -> Add (connected ?loc ?new_room ?dir) and the reverse connection.
4. Handling Reveals: If the observation is a result of an action (e.g., "You open the door, revealing the [New Room]"), ensure the state reflects the *result* (door is now open) and the newly discovered connection.

Note: in problem file's init, you shouldn't have "not ()" but only the single status.
"""

prompt_format_pf_complete = """
Please provide the output in strict JSON format, without any additional text or explanation. 
The format should strictly be:
{{
  "pf": "..."
}}
"""

# [변경됨] 목표 설정 전략(정보 획득 vs 이동) 명확화
prompt_pf_complete_generation = """
You are in a partially observable environment. Your output must be one single, complete PDDL problem file. To create it, add a `(:goal ...)` section to the provided objects and initial state, then wrap everything in the standard `(define (problem ...))` structure.

This is the domain file:
{df}

Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

This is the objects and initial state of the problem file:
{pf_init}

Goal Setting Strategy (Exploration):
Your task is always to explore. The goal must be grounded (no variables).

1. Priority 1: Information Gathering. If the initial state indicates there are closed doors at the agent's current location ((door-is-closed ?current_loc ?dir)), the goal MUST be to open one of them.
   Example: (:goal (door-is-open ?current_loc ?unexplored_dir))
   
2. Priority 2: Movement to Unvisited. If all paths from the current location are open and connections are known, the goal should be to move to an adjacent, unvisited location.
   Example: (:goal (at ?unvisited_location))
"""