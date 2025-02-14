import pandas as pd
import ast

# Read CSV data.
# Columns (in order):
# date, model, game_id, coin_found, final_step_index, last_attempt_str, steps_str
df = pd.read_csv("output/results.csv", header=None, 
                 names=["date", "model", "game_id", "coin_found", "final_step_index", "last_attempt_str", "steps_str"])

# Convert coin_found column to Boolean.
df["coin_found"] = df["coin_found"].astype(str).map({"True": True, "False": False})

def analyze_game(row):
    """
    Analyze one game (row) and return metrics.
    
    - For each attempt in every step:
        * If the attempt's small_loop (second number) is nonzero, count one solver error.
        * That solver error is considered fixed if:
              - It is not the final attempt of the whole game, OR
              - It is the final attempt but its second number is not 5.
    - A simulation error is counted once for any step that has >1 attempt.
      It is fixed if either the step is not the final step or (if final) the coin was found.
    
    Additionally, we determine abort counts:
      - abort_solver: 1 if the very final attempt is an unfixed solver error.
      - abort_simulation: 1 if the final step had a simulation error and it wasn’t fixed.
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
            # Each attempt is [large_loop, small_loop].
            large_loop, small_loop = attempt
            if small_loop != 0:
                solver_errors += 1
                # Check if this is the final attempt of the whole game.
                if not (i == num_steps - 1 and j == len(step) - 1):
                    # Not final attempt: count as fixed.
                    solver_fixed += 1
                else:
                    # Final attempt: fixed only if second number is not 5.
                    if small_loop != 5:
                        solver_fixed += 1

    simulation_errors = 0
    simulation_fixed = 0
    for i, step in enumerate(steps_list):
        if len(step) > 1:
            simulation_errors += 1
            # A simulation error is fixed if:
            # - The step is not the final step, OR
            # - It is the final step and coin_found is True.
            if i != num_steps - 1 or (i == num_steps - 1 and row["coin_found"]):
                simulation_fixed += 1

    # Determine abort counts from the final step.
    final_step = steps_list[-1]
    final_attempt = final_step[-1]
    final_large, final_small = final_attempt
    
    # Abort at solver error: final attempt is a solver error and not fixed.
    abort_solver = 1 if (final_small != 0 and final_small == 5) else 0
    # Abort at simulation error: final step had >1 attempt and was not fixed (coin not found).
    abort_simulation = 1 if (len(final_step) > 1 and not row["coin_found"]) else 0

    return {
        "coin_found": 1 if row["coin_found"] else 0,
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

# Combine metrics with model information.
df_metrics = pd.concat([df[["model"]], metrics_df], axis=1)

# Calculate average steps for successful and failed games.
df_success = df_metrics[df_metrics["coin_found"] == 1]
df_failure = df_metrics[df_metrics["coin_found"] == 0]

avg_steps_success = df_success.groupby("model")["steps"].mean().rename("avg_steps_success")
avg_steps_failure = df_failure.groupby("model")["steps"].mean().rename("avg_steps_failure")

# Group other metrics by model.
grouped = df_metrics.groupby("model").agg(
    coin_found_count=("coin_found", "sum"),
    total_solver_errors=("solver_errors", "sum"),
    total_solver_fixed=("solver_fixed", "sum"),
    total_simulation_errors=("simulation_errors", "sum"),
    total_simulation_fixed=("simulation_fixed", "sum"),
    total_abort_solver=("abort_solver", "sum"),
    total_abort_simulation=("abort_simulation", "sum")
)

# Join average steps data.
result = grouped.join(avg_steps_success, how="left").join(avg_steps_failure, how="left")
result = result.reset_index()

print(result)
