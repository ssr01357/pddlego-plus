import pandas as pd
import ast

# 1. Read your results file
file_name = "6_0430_CC"  # 4_0424_Alfworld_50_fixed, 6_0430_CC
df = pd.read_csv(
    f"output/{file_name}.csv", 
    header=None,
    names=[
        "date",
        "model",
        "model_type",
        "game_type",
        "goal_type",
        "game_id",
        "succeed",
        "final_step_index",
        "last_attempt_str",
        "steps_str",
    ],
)

# 2. Normalize the 'succeed' column to boolean
df["succeed"] = df["succeed"].astype(str).map({"True": True, "False": False})

# 3. Define helper functions

def count_pddl_first_success(steps_str: str) -> int:
    """
    Count consecutive steps that are exactly [[*, 0]] (first-time correct)
    Stops counting at the first non-first-time step.
    """
    steps = ast.literal_eval(steps_str)
    count = 0
    for step in steps:
        if isinstance(step, list) and len(step) == 1 and isinstance(step[0], list) and step[0][1] == 0:
            count += 1
        else:
            break
    return count

def pddl_pure_success(steps_str: str, succeed: bool) -> bool:
    """
    True if all steps are first-time correct and the trial succeeded.
    """
    steps = ast.literal_eval(steps_str)
    return succeed and (count_pddl_first_success(steps_str) == len(steps))

def count_baseline_first_success(steps_str: str) -> int:
    """
    Count consecutive steps that are exactly [1], stopping at first deviation.
    """
    steps = ast.literal_eval(steps_str)
    count = 0
    for step in steps:
        if step == [1]:
            count += 1
        else:
            break
    return count

def baseline_pure_success(steps_str: str, succeed: bool) -> bool:
    """
    True if all steps are [1] and the trial succeeded.
    """
    steps = ast.literal_eval(steps_str)
    return succeed and (count_baseline_first_success(steps_str) == len(steps))

# 4. Apply logic by model_type
df["first_time_success_count"] = df.apply(
    lambda row: count_pddl_first_success(row["steps_str"])
    if row["model_type"] == "PDDL"
    else count_baseline_first_success(row["steps_str"]),
    axis=1,
)

df["pure_first_time_success"] = df.apply(
    lambda row: pddl_pure_success(row["steps_str"], row["succeed"])
    if row["model_type"] == "PDDL"
    else baseline_pure_success(row["steps_str"], row["succeed"]),
    axis=1,
)

# 5. Save to a new CSV
output_path = f"output/{file_name}_no_error_fixing.csv"
df.to_csv(output_path, index=False)

print(f"Saved first-time success summary to {output_path}")
