import pandas as pd
import numpy as np
import ast

# 0. File setup
file_name = "6_0430_CC_50_fixed_non_error_fixing_first_time_success" # 4_0424_Alfworld_50_fixed_non_error_fixing, 6_0430_CC_50_fixed_non_error_fixing_first_time_success
input_path = f"output/{file_name}.csv"
output_path = f"output/{file_name}_summary.csv"

# 1. Read the trial‐level data (with the two extra columns)
df = pd.read_csv(input_path, dtype={"steps_str": str})

# 2. Validate presence of those extra columns
expected = {
    "date","model","model_type","game_type","goal_type",
    "game_id","succeed","final_step_index","last_attempt_str",
    "steps_str","first_time_success_count","pure_first_time_success"
}
missing = expected - set(df.columns)
if missing:
    raise KeyError(f"Missing columns: {missing}")

#==== previous code ====
df["succeed"] = df["succeed"].astype(str).map({"True": True, "False": False})


def analyze_game(row):
    try:
        steps_list = ast.literal_eval(row["steps_str"])
    except Exception as e:
        print(f"Error parsing steps for row {row.name}: {e}")
        return None

    num_steps = len(steps_list)
    is_baseline = row["model_type"] == "baseline"

    if is_baseline:
        simulation_errors = 0
        simulation_fixed = 0

        for i, step in enumerate(steps_list):
            # Step contains simulation error if any value ≠ 1
            has_error = any(val != 1 for val in step)
            if has_error:
                simulation_errors += 1
                if i < len(steps_list) - 1:
                    simulation_fixed += 1

        return {
            "succeed": 1 if row["succeed"] else 0,
            "steps": num_steps,
            "solver_errors": np.nan,
            "solver_fixed": np.nan,
            "simulation_errors": simulation_errors,
            "simulation_fixed": simulation_fixed,
            "abort_solver": np.nan,
            "abort_simulation": simulation_errors - simulation_fixed,
        }

    # Full model logic (unchanged)
    solver_errors = 0
    solver_fixed = 0

    for i, step in enumerate(steps_list):
        for j, attempt in enumerate(step):
            if not isinstance(attempt, (list, tuple)) or len(attempt) != 2:
                print(f"Unexpected format at row {row.name}, step {i}, attempt {j}: {attempt}")
                continue
            _, small_loop = attempt
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
    if isinstance(final_attempt, (list, tuple)) and len(final_attempt) == 2:
        _, final_small = final_attempt
    else:
        final_small = 0

    abort_solver = 1 if (final_small != 0 and final_small == 5) else 0
    abort_simulation = simulation_errors - simulation_fixed

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

# Apply per-game analysis
metrics = df.apply(analyze_game, axis=1)
metrics_df = pd.DataFrame(metrics.tolist())

# Combine metrics with identifiers (include model_type now)
df_metrics = pd.concat([df[["model_type", "model", "game_type", "goal_type"]], metrics_df], axis=1)

# === Flexible group key setup ===
# You can modify this as needed:
# group_keys = ["model_type", "model", "goal_type"]
# group_keys = ["model_type", "model", "game_type", "goal_type"]
group_keys = ["model_type", "model", "goal_type", "game_type"]
# group_keys = ["model_type","game_type"]

# Filter successes and failures
df_success = df_metrics[df_metrics["succeed"] == 1]
df_failure = df_metrics[df_metrics["succeed"] == 0]

# Compute average steps
avg_steps_success = df_success.groupby(group_keys)["steps"].mean().rename("avg_steps_success")
avg_steps_failure = df_failure.groupby(group_keys)["steps"].mean().rename("avg_steps_failure")

# Aggregate results
grouped = df_metrics.groupby(group_keys).agg(
    succeed_count=("succeed", "sum"),
    total_solver_errors=("solver_errors", "sum"),
    total_solver_fixed=("solver_fixed", "sum"),
    total_simulation_errors=("simulation_errors", "sum"),
    total_simulation_fixed=("simulation_fixed", "sum"),
    total_abort_solver=("abort_solver", "sum"),
    total_abort_simulation=("abort_simulation", "sum")
)

# Merge in the average step metrics
result = grouped.join(avg_steps_success, how="left").join(avg_steps_failure, how="left").reset_index()

# Add trial count
trial_count = df_metrics.groupby(group_keys).size().rename("trial_count")

# Merge into result
result = result.join(trial_count, on=group_keys)

# Set solver-related columns to NaN for baseline models
baseline_mask = result["model_type"] == "baseline"
result.loc[baseline_mask, ["total_solver_errors", "total_solver_fixed", "total_abort_solver"]] = np.nan

# Convert selected count columns to integer type (preserving NaNs)
int_cols = [
    "succeed_count", "total_simulation_errors", "total_simulation_fixed",
    "total_abort_simulation", "trial_count"
]
result[int_cols] = result[int_cols].astype("Int64")
summary = result.copy()

# 3. Convert types
# df["succeed"] = (
#     df["succeed"].astype(str)
#                  .map({"True": True, "False": False})
#                  .fillna(False)
# )
df["first_time_success_count"] = (
    pd.to_numeric(df["first_time_success_count"], errors="coerce")
      .fillna(0)
      .astype(int)
)
df["pure_first_time_success"] = (
    df["pure_first_time_success"].astype(str)
                                  .map({"True": True, "False": False})
                                  .fillna(False)
)

# 4. Parse steps_str into lists and count total steps
def safe_parse(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

df["steps_list"] = df["steps_str"].apply(safe_parse)
df["steps"] = df["steps_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)


# 6. Build the “non_error_fixing” summary, but only for PDDL trials
pddl_df = df[df["model_type"] == "PDDL"]
nef_rows = []
nef_group_keys = ["model", "goal_type", "game_type"]

for (model, goal, game), grp in pddl_df.groupby(nef_group_keys):
    nef_rows.append({
        "model_type": "non_error_fixing",
        "model": model,
        "goal_type": goal,
        "game_type": game,
        "succeed_count": int(grp["pure_first_time_success"].sum()),
        "trial_count": len(grp),
        "avg_steps_success": grp.loc[grp["pure_first_time_success"], "steps"].mean(),
        "avg_steps_failure": grp.loc[~grp["pure_first_time_success"], "steps"].mean(),
        # solver/simulation columns are NaN here
        "total_solver_errors": np.nan,
        "total_solver_fixed": np.nan,
        "total_simulation_errors": np.nan,
        "total_simulation_fixed": np.nan,
        "total_abort_solver": np.nan,
        "total_abort_simulation": np.nan,
    })

nef = pd.DataFrame(nef_rows)

# 7. Align columns and concatenate
for col in summary.columns:
    if col not in nef.columns:
        nef[col] = np.nan
nef = nef[summary.columns]

combined = pd.concat([summary, nef], ignore_index=True)

# 8. Write out
combined.to_csv(output_path, index=False)
print(f"Wrote combined summary (including non_error_fixing for PDDL only) to {output_path}")
