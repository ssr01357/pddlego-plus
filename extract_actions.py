import os
import re

def count_successful_steps(filepath):
    with open(filepath, 'r') as f:
        log = f.read()

    # Extract all `successful_actions` blocks
    successful_actions_blocks = re.findall(r"successful_actions:\s*\[(.*?)\]", log, re.DOTALL)
    all_successful_actions = [re.findall(r"'(.*?)'", block) for block in successful_actions_blocks]

    if not all_successful_actions:
        return 0  # No successful steps at all

    total_successful = len(all_successful_actions[-1])  # Last block only

    # Extract last solver actions
    solver_actions_blocks = re.findall(r"Actions from solver\(df, pf\):\s*(\[.*?\]|None)", log)
    last_solver_block = solver_actions_blocks[-1] if solver_actions_blocks else None

    # Check if trial succeeded
    lines = log.strip().splitlines()
    trial_succeeded = lines[-1].strip() in {"Coin found!", "Done!"}

    # Add last solver step if success and valid
    if trial_succeeded and last_solver_block and last_solver_block != "None":
        last_actions = re.findall(r"'(.*?)'", last_solver_block)
        total_successful += len(last_actions)

    return total_successful

def average_successful_steps_in_folder(folder_path):
    total_steps = 0
    file_count = 0

    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder_path, fname)
            steps = count_successful_steps(fpath)
            total_steps += steps
            file_count += 1
            # print(f"{fname}: {steps} steps")

    if file_count == 0:
        print("No .txt files found.")
        return 0

    avg = total_steps / file_count
    print(f"\nAverage successful steps across {file_count} files: {avg:.2f}")
    return avg

# Example usage
folder = "output/4_Alfworld/AlfW_PDDLego+_deepseek"
average_successful_steps_in_folder(folder)



# Example usage
# print("Successful steps:", count_successful_steps("output/4_Alfworld/AlfW_PDDLego+_o3-mini/2025-04-24_o3-mini-2025-01-31_PDDL_detailed_34.txt"))
# print("Successful steps:", count_successful_steps("output/6_CoinCollector/CC_PDDLego+_deepseek/2025-05-02_deepseek_PDDL_detailed_9_76.txt"))
