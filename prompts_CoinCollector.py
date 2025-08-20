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
Return output in STRICT JSON with two keys only, no extra commentary:
{{
  "df": "<PDDL domain file text>",
  "pf": "<PDDL problem file text>"
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
You should generate df and pf strictly follow these valid actions. There are in total 2 actions, that should exactly be the following two:
1. :action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
2. :action move
    :parameters (?from - location ?to - location ?dir - direction)

Note:
In problem file's init, you shouldn't have "not ()" but only the single status.
The goal must be ground (e.g., (at loc_2)), never (at ?l) or quantifiers.

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
You should generate df and pf strictly follow these valid actions. There are in total 2 actions, that should exactly be the following two:
1. :action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
2. :action move
    :parameters (?from - location ?to - location ?dir - direction)
You should have a goal in the problem file like this: 
(:goal 
    (at loc_2)  ; example of a grounded goal
) where the location should be somewhere not visited

Note:
In problem file's init, you shouldn't have "not ()" but only the single status.
The goal must be ground (e.g., (at loc_2)), never (at ?l) or quantifiers.

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



# =========================
# [3] Python IR → PDDL: prompts
# =========================



pypddl_instruction = '''Python representation of PDDL domain file contains domain name, requirements, types of objects in the domain, predicates, and actions.
Based on the natural language domain description, identify the actions that are possible.
Identify action sematics i.e. understand the preconditions under which that action could be done and the effects of the action.
Then identify appropriate predicates that could enable action semantics i.e. preconditions and effects.
Python representation of PDDL domain file has a definitive syntax that must be followed for any domain. An abstract example is given below:

In the following Python domain file, the example class DomainName has been created. Its structure is similar to how a PDDL domain should be defined.

Name of the domain is the name of the Python class (DomainName).
Types are defined as class variables at the top (Type1, Type2).
Predicates are defined as instance methods decorated with @predicate.
Actions are defined as instance methods decorated with @action

The positional arguments of @predicate and @action decorators are the types of the respective arguments.
Methods decorated with @predicate should have empty bodies.
Methods decorated with @action return a tuple of two lists

<domain_file>
# imports stays exactly same for all domain files
# this python code should be executed without any python syntax errors
from py2pddl import Domain, create_type
from py2pddl import predicate, action

class DomainName(Domain):
	# let's think step by step and correctly define all the aspects of a domain file
	# making sure that the list of types, predicates, and actions comprehensively define the give domain
    Type1 = create_type("Type1")
    Type2 = create_type("Type2")

	# making sure that all the predicates needed are defined below
	# predicates have all the arguments needed along with the types
    # let's not have two predicates with the same name
    # let's make sure two arguments of a predicate function doesn't have same names. That would be a python syntax error.
    @predicate(Type1, Type2)
    def predicate1(self, arg1, arg2):
        """Complete the method signature and specify
        the respective types in the decorator"""

    @predicate(Type1)
    def predicate2(self, arg1):
        """Complete the method signature and specify
        the respective types in the decorator"""

	# let's define a list of actions that comprehensively define the given domain
	# the names for actions are always given precisely in the domain description, using only those actions
	# making sure that each action is defined with :parameters where every arguments needed for preconditions and effects are specified

	# writing the definition of action1 which is one of the action given in domain description
	# first defining all the parameters needed for predicates in preconditions and effects
	# making sure preconditions is logically correct and aligns with the pre-conditions where action1 could be performed
	# making sure effects is logically correct and aligns with the post-conditions or results of action being performed
    @action(Type1, Type2, Type2)
    def action1(self, arg1, arg2, arg3):
        precond = [self.predicate1(arg1, arg3), self.predicate2(arg1)]
        effect = [~self.predicate1(arg1, arg2), self.predicate2(arg3)]
        return precond, effect

	# writing the definition of action1 which is one of the action given in domain description
	# first defining all the parameters needed for predicates in preconditions and effects
	# making sure :precondition is logically correct and aligns with the pre-conditions where action1 could be performed
	# making sure :effect is logically correct and aligns with the post-conditions or results of action being performed
    @action(Type1)
    def action2(self, arg1):
        precond = [self.predicate2(arg1)]
        effect = [~self.predicate2(arg1)]
        return precond, effect
</domain_file>

Notes for generating domain file: 
- the above example file is only for understanding the syntax
- type1 & type2 are only representative and should be replaced with appropriate types. There could be any number of types.
- predicate1 & predicate2 are only representative and should be replaced with appropriate predicates. There could be any number of predicates.
- action1 & action2 are only representative and should be replaced with appropriate actions. There could be any number of actions.
- arg1 & arg2 are only representative and should be replaced with appropriate arguments for predicates and in preconditions and effects.
- Use predicates with arguments of the right type as declared in domain file
- All the arguments to any :precondition or :effect of an action should be declared in :parameters as input arguments
- Verify and reason about correctness by reiterating the reasoning behind different predicates and actions




Python representation of PDDL problem file contains problem name, domain name, objects in this problem instance, init state of objects, and goal state of objects.
Based on the natural language problem description, identify the relevant objects for this problems with their names and types.
Represent the initial state with the appropriate predicates and object arguments. Represent the goal state with the appropriate predicates and object arguments.
Python representation of PDDL problem file has a definitive syntax that must be followed for any problem. An abstract example is given below.

<problem_file>
# imports stays the same for all problem files
# Assume DomainName is declared just before the problem file
# this python code should be executed correctly without any syntax errors
from py2pddl import goal, init

# let's define all the aspects of a problem file below starting with domain, objects, init state, and goal state

class ProblemName(DomainName):
	# making sure of defining all the objects that are required to define init state and goal state
    # defining the objects with a list of strings will generate a dict and the object should be retrieve as a dict as in the example below in goal function
    def __init__(self):
        super().__init__()
        self.type1Objs = DomainName.Type1.create_objs([1, 2], prefix="type1Obj")
        self.type2Objs = DomainName.Type2.create_objs(["type2Obj1", "type2Obj2"])

	# reasoning about the validity of init state that captures the properties of objects in the initial state
	# defining the correct init state that represent the problem file below
    # let's index correctly for lists and dicts
    @init
    def init(self):
        at = [self.predicate1(self.type1Objs[1], self.type2Objs["type2Obj1"]),
              self.predicate2(self.type1Objs[1]),]
        return at

	# reasoning about the validity of init state that captures the properties of objects in the initial state
	# defining the correct init state that represent the problem file below
    # let's index correctly for lists and dicts
    @goal
    def goal(self):
        return [self.predicate1(self.type1Objs[1], self.type2Objs["type2Obj2"]),
                self.predicate2(self.type1Objs[2])]
</problem_file>

Notes for generating problem file:
- No need to import DomainName for python problem file. Python Domain file and python problem file would be merged and executed.
- type1Objs, type2Oobjs, ... are only representative and should be replaced with appropriate objects. There could be any number of obects with their types.
- init state with predicate1 & predicate2 is only representative and should be replaced with appropriate predicates that define init state
- goal state with predicate1 & predicate2 is only representative and should be replaced with appropriate predicates that define goal state
- Use predicates with arguments of the right type as declared in domain file
- All the objects that would be arguments of predicates in init and goal states should be declared in __init__
'''




# later: Use clear, conventional Python that a downstream tool could translate to PDDL.
prompt_pyir = """
You will be given a natural language domain description and problem description.
Your task is to generate a Python representation of Planning Domain Definition Language (PDDL) — one domain class and one problem class — using a simple, class-based intermediate representation (IR). 
{pypddl_instruction}

Here is the natural language description of the domain and problem:

You are in an environment that you explore step by step. You must build and update the Python Representation of PDDL files of the environment based on only your observations.
Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
Do not assume that there will be a door connecting rooms.
Your task is always to keep exploration and go to a location you have not visited yet.
In other words, your goal should go to other not visited location.
Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

Return output in strict JSON with two fields only, no extra text:
{{
  "py_domain": "<python code for the domain class that represents the PDDL domain>",
  "py_problem": "<python code for the problem class that represents the PDDL problem>"
}}
"""

# =========================

# [3] Python IR → PDDL: helpers

# =========================

# prompt_pyir2pddl = """
# You will be given a Python representation of Planning Domain Definition Language (PDDL) — one domain class and one problem class, along with the natrural language description of the domain and problem.
# Your task is to generate syntactically correct domain file and problem file in Planning Domain Definition Language (PDDL), based on the provided information.

# # Natural language description of the domain and problem
# You are in an environment that you explore step by step. You must build and update the Python Representation of PDDL files of the environment based on only your observations.
# Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
# Do not assume that there will be a door connecting rooms.
# Your task is always to keep exploration and go to a location you have not visited yet.
# In other words, your goal should go to other not visited location.
# Here are your current observations: {brief_obs}
# Here are some valid actions you can take: {valid_actions}

# # Python representation of PDDL domain file
# {py_domain}

# # Python representation of PDDL problem file
# {py_problem}


# Return output in STRICT JSON with two keys only, no extra commentary:
# {{
#   "df": "<PDDL domain file text>",
#   "pf": "<PDDL problem file text>"
# }}
# """
# Replace prompt_pyir2pddl with:
prompt_pyir2pddl = """
You will be given a class-based Python IR for a PDDL domain (one domain class) and a PDDL problem instance (one problem class),
plus the current observations and valid actions. Convert that IR into syntactically correct and mutually consistent PDDL.

### Hard constraints (must follow exactly)
- Declare EXACTLY these two actions with these names and parameter orders:
  1) :action open-door
     :parameters (?loc1 - location ?loc2 - location ?dir - direction)
  2) :action move
     :parameters (?from - location ?to - location ?dir - direction)


- Use typed PDDL (declare `location` and `direction` in :types).
- In problem file's init, you shouldn't have "not ()" but only the single status.
- The goal must be ground (e.g., (at loc_2)), never (at ?l) or quantifiers.
- Keep edits minimal if previous PDDL is provided.

### Current observations
{brief_obs}

### Valid actions
{valid_actions}

### (Optional) Previous PDDL (use as base with minimal fixes)
{prev_pddl}

### (Optional) Planner/validator feedback to fix
{prev_err}

### (Optional) Environment feedback to fix
{env_err}

### Python IR: domain class
{py_domain}

### Python IR: problem class
{py_problem}

Return output in STRICT JSON with two keys only, no extra commentary:
{{
  "df": "<PDDL domain file text>",
  "pf": "<PDDL problem file text>"
}}
"""