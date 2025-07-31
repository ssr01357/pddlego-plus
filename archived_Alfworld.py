import subprocess

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

    edit_json_df, edit_json_pf = run_llm(prompt) # , model_name

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
