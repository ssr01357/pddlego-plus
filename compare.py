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

prompt_pyir2pddl = """
You will be given a Python representation of Planning Domain Definition Language (PDDL) — one domain class and one problem class, along with the natrural language description of the domain and problem.
Your task is to generate syntactically correct domain file and problem file in Planning Domain Definition Language (PDDL), based on the provided information.

# Natural language description of the domain and problem
You are in an environment that you explore step by step. You must build and update the Python Representation of PDDL files of the environment based on only your observations.
Do not create something not appeared in the observations and also do not miss any observations e.g. through closed doors you may assume a room behind.
Do not assume that there will be a door connecting rooms.
Your task is always to keep exploration and go to a location you have not visited yet.
In other words, your goal should go to other not visited location.
Here are your current observations: {brief_obs}
Here are some valid actions you can take: {valid_actions}

# Python representation of PDDL domain file
{py_domain}

# Python representation of PDDL problem file
{py_problem}


Return output in STRICT JSON with two keys only, no extra commentary:
{{
  "df": "<PDDL domain file text>",
  "pf": "<PDDL problem file text>"
}}
"""


def llm_to_pyir(model_name, brief_obs, valid_actions, history = None):
    prompt = prompt_pyir.format(pypddl_instruction=pypddl_instruction, brief_obs=brief_obs, valid_actions=valid_actions)

    if history:
        prompt += f"""
        ### Previous IR and PDDL history:
        {history}
        ### End of history.

        Now, based on the above history and your current observation, update the Python representation of PDDL files. 
        """

    resp = run_llm_json(prompt, model_name)
    py_domain = resp.get("py_domain", "")
    py_problem = resp.get("py_problem", "")
    return py_domain, py_problem, prompt




def llm_pyir_to_pddl(model_name, py_domain, py_problem, brief_obs, valid_actions):
    prompt = prompt_pyir2pddl.format(
        py_domain=py_domain,
        py_problem=py_problem,
        brief_obs=brief_obs,
        valid_actions=valid_actions
    )
    resp = run_llm_json(prompt, model_name)
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
                                action = ""

                                # (A) Ensure/Update Python IR
                                ir_tries = 0
                                while not py_domain or not py_problem:
                                    ir_tries += 1
                                    if ir_tries > 3:
                                        raise ValueError("Failed to generate valid Python IR after multiple attempts.")
                                    py_domain, py_problem, pyir_prompt = llm_to_pyir(model_name, brief_obs, valid_actions) # now, history is not implemented
                                    with open(file_name, "a") as f:
                                        f.write(f"[PyIR Prompt #{ir_tries}] {pyir_prompt}\n")
                                        f.write(f"Generated py_domain:\n{py_domain}\n")
                                        f.write(f"Generated py_problem:\n{py_problem}\n")
                    

                                # (B) Convert IR → PDDL and ask planner
                                if not df and not pf:
                                    num_tries = 0
                                    df, pf, conv_prompt = llm_pyir_to_pddl(model_name, py_domain, py_problem, brief_obs, valid_actions)
                                    action, err = get_action_from_pddl(df, pf)
                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"[IR→PDDL Prompt] {conv_prompt} \n")
                                        f.write(f"[PDDL df]\n{df}\n\n[PDDL pf]\n{pf}\n")
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        # if err:
                                        #     f.write(f"Solver stderr:\n{err}\n")

                                    # PDDL repair loop
                                    while not action and num_tries < 5:
                                        df, pf, conv_prompt = llm_pyir_to_pddl(model_name, py_domain, py_problem, brief_obs, valid_actions)
                                        action, err = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"[IR→PDDL Prompt Retry] {conv_prompt} \n")
                                            f.write(f"[PDDL df]\n{df}\n\n[PDDL pf]\n{pf}\n")
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            # if err:
                                            #     f.write(f"Solver stderr:\n{err}\n")

                                    # # Occasionally refresh IR if still failing
                                    # if (not action) and num_tries in (5, 6):
                                    #     py_domain, py_problem, pyir_prompt = llm_to_pyir(model_name, brief_obs, valid_actions, history=f"{py_domain}\n{py_problem}\n{df}\n{pf}")
                                    #     df, pf, conv_prompt = llm_pyir_to_pddl(model_name, py_domain, py_problem, brief_obs, valid_actions)
                                    #     action, err = get_action_from_pddl(df, pf)
                                    #     with open(file_name, "a") as f:
                                    #         f.write(f"[PyIR Re-synthesis Prompt #{num_tries}] {pyir_prompt}\n")
                                    #         f.write(f"Re-synthesized py_domain:\n{py_domain}\n")
                                    #         f.write(f"Re-synthesized py_problem:\n{py_problem}\n")
                                else: # 처음 들어왔는데 (action_queue가 비어있고 df, pf가 있는 경우)
                                    num_tries = 0
                                    # Every time read new error message from larger loop
                                    # In llm_to_pddl, detect if new large loop error message exists
                                    df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, "", detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message) # need to add new error message
                                    action, err = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Prompt: {prompt} \n") 
                                        f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                        f.write(f"Actions from solver(df, pf): {action} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message)
                                        action, err = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"Prompt: {prompt} \n") 
                                            f.write(f"Generated df and pf: \n {df} \n {pf} \n") 
                                            f.write(f"Actions from solver(df, pf): {action} \n")

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
                                large_loop_error_message = obs
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
