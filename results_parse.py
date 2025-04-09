import pandas as pd
import ast

# Read CSV data.
# Updated Columns (in order):
# date, model, game_type, prompt_type, game_id, succeed, final_step_index, last_attempt_str, steps_str
df = pd.read_csv("output/alfworld_detailed_results.csv", header=None, 
                 names=["date", "model", "game_type", "prompt_type", "game_id", "succeed", 
                        "final_step_index", "last_attempt_str", "steps_str"])

# Convert succeed column to Boolean.
df["succeed"] = df["succeed"].astype(str).map({"True": True, "False": False})

def analyze_game(row):
    """
    Analyze one game (row) and return metrics, based on solver/simulation errors and abort types.
    """
    try:
        steps_list = ast.literal_eval(row["steps_str"])
    except Exception as e:
        print(f"Error parsing steps for row {row.name}: {e}")
        return None
    
    num_steps = len(steps_list)
    
    solver_errors = 0
    solver_fixed = 0
    for i, step in enumerate(steps_list):
        for j, attempt in enumerate(step):
            large_loop, small_loop = attempt
            if small_loop != 0:
                solver_errors += 1
                if not (i == num_steps - 1 and j == len(step) - 1):
                    solver_fixed += 1
                elif small_loop != 5:
                    solver_fixed += 1

    simulation_errors = 0
    simulation_fixed = 0
    for i, step in enumerate(steps_list):
        if len(step) > 1:
            simulation_errors += 1
            if i != num_steps - 1 or (i == num_steps - 1 and row["succeed"]):
                simulation_fixed += 1

    final_step = steps_list[-1]
    final_attempt = final_step[-1]
    _, final_small = final_attempt

    abort_solver = 1 if (final_small != 0 and final_small == 5) else 0
    abort_simulation = 1 if (len(final_step) > 1 and not row["succeed"]) else 0

    return {
        "succeed": 1 if row["succeed"] else 0,
        "steps": num_steps,
        "solver_errors": solver_errors,
        "solver_fixed": solver_fixed,
        "simulation_errors": simulation_errors,
        "simulation_fixed": simulation_fixed,
        "abort_solver": abort_solver,
        "abort_simulation": abort_simulation,
    }

# Apply per-game analysis.
metrics = df.apply(analyze_game, axis=1)
metrics_df = pd.DataFrame(metrics.tolist())

# Combine metrics with identifiers
df_metrics = pd.concat([df[["model", "game_type", "prompt_type"]], metrics_df], axis=1)

# === Optional grouping ===
group_keys = ["model", "prompt_type"]  # You can add "game_type" if needed
# group_keys = ["model", "game_type", "prompt_type"]  # <-- Uncomment for 3-level group

df_success = df_metrics[df_metrics["succeed"] == 1]
df_failure = df_metrics[df_metrics["succeed"] == 0]

avg_steps_success = df_success.groupby(group_keys)["steps"].mean().rename("avg_steps_success")
avg_steps_failure = df_failure.groupby(group_keys)["steps"].mean().rename("avg_steps_failure")

grouped = df_metrics.groupby(group_keys).agg(
    succeed_count=("succeed", "sum"),
    total_solver_errors=("solver_errors", "sum"),
    total_solver_fixed=("solver_fixed", "sum"),
    total_simulation_errors=("simulation_errors", "sum"),
    total_simulation_fixed=("simulation_fixed", "sum"),
    total_abort_solver=("abort_solver", "sum"),
    total_abort_simulation=("abort_simulation", "sum")
)

result = grouped.join(avg_steps_success, how="left").join(avg_steps_failure, how="left")
result = result.reset_index()

print(result)
