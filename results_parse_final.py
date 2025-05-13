import pandas as pd
import numpy as np
import ast

# === CONFIGURATION ===
file_name = "6_CoinCollector"  # 4_Alfworld, 6_CoinCollector, CC_o4_mini_high
input_csv = f"output/{file_name}.csv"
output_csv = f"output/{file_name}_summary.csv"

# === STEP 1: Load and Normalize Data ===
df = pd.read_csv(
    input_csv,
    header=None,
    names=[
        "date", "model", "model_type", "game_type", "goal_type",
        "game_id", "succeed", "final_step_index", "last_attempt_str", "steps_str"
    ]
)
df["succeed"] = df["succeed"].astype(str).map({"True": True, "False": False})


# === STEP 2: Define Step Parsing Logic ===

def safe_parse_steps(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

def count_pddl_first_success(steps_str):
    steps = safe_parse_steps(steps_str)
    return sum(
        1 for step in steps
        if isinstance(step, list) and len(step) == 1 and isinstance(step[0], list) and step[0][1] == 0
    )

def pddl_pure_success(steps_str, succeed):
    steps = safe_parse_steps(steps_str)
    return succeed and count_pddl_first_success(steps_str) == len(steps)

def count_baseline_first_success(steps_str):
    steps = safe_parse_steps(steps_str)
    return sum(1 for step in steps if step == [1])

def baseline_pure_success(steps_str, succeed):
    steps = safe_parse_steps(steps_str)
    return succeed and all(step == [1] for step in steps)


# === STEP 3: Annotate First-Time Success Columns ===
df["first_time_success_count"] = df.apply(
    lambda row: count_pddl_first_success(row["steps_str"]) if row["model_type"] == "PDDL"
    else count_baseline_first_success(row["steps_str"]),
    axis=1
)

df["pure_first_time_success"] = df.apply(
    lambda row: pddl_pure_success(row["steps_str"], row["succeed"]) if row["model_type"] == "PDDL"
    else baseline_pure_success(row["steps_str"], row["succeed"]),
    axis=1
)


# === STEP 4: Analyze Per-Trial Step-Level Errors ===

def analyze_game(row):
    steps_list = safe_parse_steps(row["steps_str"])
    num_steps = len(steps_list)
    is_baseline = row["model_type"] == "baseline"

    if is_baseline:
        errors = [any(val != 1 for val in step) for step in steps_list]
        sim_err = sum(errors)
        sim_fix = sum(errors[:-1])
        return {
            "succeed": int(row["succeed"]), "steps": num_steps,
            "solver_errors": np.nan, "solver_fixed": np.nan,
            "simulation_errors": sim_err, "simulation_fixed": sim_fix,
            "abort_solver": np.nan, "abort_simulation": sim_err - sim_fix
        }

    # For PDDL models
    solver_errors = solver_fixed = sim_errors = sim_fixed = 0
    for i, step in enumerate(steps_list):
        for j, attempt in enumerate(step):
            if isinstance(attempt, (list, tuple)) and len(attempt) == 2:
                _, small_loop = attempt
                if small_loop != 0:
                    solver_errors += 1
                    if not (i == len(steps_list) - 1 and j == len(step) - 1):
                        solver_fixed += 1
                    elif small_loop != 5:
                        solver_fixed += 1

        if len(step) > 1:
            sim_errors += 1
            if i != len(steps_list) - 1 or row["succeed"]:
                sim_fixed += 1

    final_small = steps_list[-1][-1][1] if isinstance(steps_list[-1][-1], (list, tuple)) else 0
    abort_solver = 1 if final_small == 5 else 0

    return {
        "succeed": int(row["succeed"]), "steps": num_steps,
        "solver_errors": solver_errors, "solver_fixed": solver_fixed,
        "simulation_errors": sim_errors, "simulation_fixed": sim_fixed,
        "abort_solver": abort_solver, "abort_simulation": sim_errors - sim_fixed
    }


metrics_df = pd.DataFrame(df.apply(analyze_game, axis=1).tolist())
df_metrics = pd.concat([df[["model_type", "model", "game_type", "goal_type"]], metrics_df], axis=1)


# === STEP 5: Group-Level Summary Metrics ===
group_keys = ["model_type", "model", "goal_type", "game_type"]

df_success = df_metrics[df_metrics["succeed"] == 1]
df_failure = df_metrics[df_metrics["succeed"] == 0]

summary = df_metrics.groupby(group_keys).agg(
    succeed_count=("succeed", "sum"),
    total_solver_errors=("solver_errors", "sum"),
    total_solver_fixed=("solver_fixed", "sum"),
    total_simulation_errors=("simulation_errors", "sum"),
    total_simulation_fixed=("simulation_fixed", "sum"),
    total_abort_solver=("abort_solver", "sum"),
    total_abort_simulation=("abort_simulation", "sum")
)

summary["avg_steps_success"] = df_success.groupby(group_keys)["steps"].mean()
summary["avg_steps_failure"] = df_failure.groupby(group_keys)["steps"].mean()
summary["trial_count"] = df_metrics.groupby(group_keys).size()
summary.reset_index(inplace=True)

# Clean up for baseline
baseline_mask = summary["model_type"] == "baseline"
summary.loc[baseline_mask, ["total_solver_errors", "total_solver_fixed", "total_abort_solver"]] = np.nan


# === STEP 6: Non-Error-Fixing (First-Time Pure Success Only) Summary ===
df["steps_list"] = df["steps_str"].apply(safe_parse_steps)
df["steps"] = df["steps_list"].apply(len)

nef_rows = []
for (model, goal, game), grp in df[df["model_type"] == "PDDL"].groupby(["model", "goal_type", "game_type"]):
    success_grp = grp[grp["pure_first_time_success"]]
    failure_grp = grp[~grp["pure_first_time_success"]]
    nef_rows.append({
        "model_type": "non_error_fixing", "model": model, "goal_type": goal, "game_type": game,
        "succeed_count": len(success_grp), "trial_count": len(grp),
        "avg_steps_success": success_grp["steps"].mean(),
        "avg_steps_failure": failure_grp["steps"].mean(),
        **{col: np.nan for col in [
            "total_solver_errors", "total_solver_fixed",
            "total_simulation_errors", "total_simulation_fixed",
            "total_abort_solver", "total_abort_simulation"
        ]}
    })

nef_df = pd.DataFrame(nef_rows)
for col in summary.columns:
    if col not in nef_df.columns:
        nef_df[col] = np.nan
nef_df = nef_df[summary.columns]

# === STEP 7: Combine and Save ===
combined = pd.concat([summary, nef_df], ignore_index=True)
combined.to_csv(output_csv, index=False)
print(f"Wrote combined summary to {output_csv}")
