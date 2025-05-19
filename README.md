# PDDLego+: Zero-Shot Iterative Formalization and Planning in Partially Observable Environments

This repository contains the implementation for our paper: "Zero-Shot Iterative Formalization and Planning in Partially Observable Environments".

## Overview

PDDLego+ is a framework that iteratively formalizes, plans, grows, and refines PDDL (Planning Domain Definition Language) representations in a zero-shot manner for partially observable environments. Unlike previous approaches that focus on fully observable environments, PDDLego+ tackles the more realistic and challenging scenario where complete information is not available, eliminating the need for access to existing trajectories.

Our framework demonstrates:
- Superior performance in textual simulated environments
- Robustness against increasing problem complexity
- Interpretable domain knowledge that benefits future tasks

## Environments

We evaluate PDDLego+ on two textual simulated environments:

1. **CoinCollector**: A grid-world environment where an agent must collect coins while avoiding obstacles
2. **ALFWorld**: A household environment with complex object interactions

## Models

We evaluate the performance of various language models with our framework:
- DeepSeek-R1-671B
- GPT-4.1-2025-04-14
- o3-mini-2025-01-31
- o4-mini-2025-04-16
- gpt-4o-2024-05-13
- DeepSeek-R1-Distill-Llama-70B
- DeepSeek-R1-Distill-Qwen-32B

## Repository Structure

```
PDDLEGO-PLUS/
├── df_cache/                    # Cache for fixed domain file operations
├── Figures/                     # Generated figures in paper
├── output/                      # Trial output log
│   ├── Alfworld/               # ALFWorld trial logs
│   ├── CoinCollector/          # CoinCollector trial logs
│   ├── Alfworld.csv            # Complete ALFWorld trial data
│   ├── Alfworld_summary.csv    # Summarized ALFWorld metrics
│   ├── CoinCollector.csv       # Complete CoinCollector trial data
│   └── CoinCollector_summary.csv # Summarized CoinCollector metrics
├── test_files/                  # Test files for development
├── .gitignore                  
├── domain.pddl                  # Example PDDL domain file
├── extract_actions.py           # Script to extract actions count from trial results
├── interactive_Alfworld.py      # Main ALFWorld environment interaction code
├── interactive_CoinCollector.py # Main CoinCollector environment interaction code
├── LICENSE                      
├── plan.txt                     # Example generated plan
├── problem.pddl                 # Example PDDL problem file
├── requirement.txt              # Dependencies and requirements
├── results_parse_final.py       # Script to parse trial data into metrics
└── visualize_pddlego+.ipynb     # Jupyter notebook for result visualization
```

## Experiment Configuration

Our experiments test different configurations:

- **PlanGen**: Direct action generation without formal planning
- **PDDLego+**: Our iterative PDDL formalization and planning approach
- **Goal Types**:
  - `detailed`: Full goal description (simple + hint + goal description)
  - `subgoal`: simple prompt
  - `without_hint`: simple + goal description
  - `without_detailed_goal`: simple + hint
- **Fixed Domain File**: Experiments controlling domain files from df_cache folder

## Installation and Setup

Our implementation uses Python scripts including `interactive_Alfworld.py` and `interactive_CoinCollector.py`. Results are analyzed using Python scripts including `results_parse_final.py` and `extract_actions.py`. The visualizations are done in Jupyter Notebook `visualize_pddlego+.ipynb`.

```bash
# Create a virtual environment (optional but recommended)
conda create -n pddlego-plus python=3.10
conda activate pddlego-plus

# Install dependencies
pip install -r requirement.txt

# Enter your OpenAI API key and deepseek API key
export OPENAI_API_KEY="your_openai_api_key_here"
export deepseek_API="your_deepseek_api_key_here"
```

## Usage

Run `interactive_CoinCollector.py` or `interactive_Alfworld.py` to launch experiments.
Each script supports:
- single-trial runs
- fixed 100-trial batches (*_50)
- fixed-df experiments that reuse a cached domain file

All helper functions are already defined inside the two scripts; just call them when the file is executed as main.

Before running, set
- folder_name: sub-folder under output/ where raw logs will be written
- result_name: CSV file name logging the summary of each trial

Simply uncomment the blocks you need.

### Available Models

- `deepseek`: representing `DeepSeek-R1-671B`
- `o3-mini-2025-01-31`
- `o4-mini-2025-04-16`
- `gpt-4.1-2025-04-14`
- `gpt-4o-2024-05-13`
- `DeepSeek-R1-Distill-Llama-70B`
- `DeepSeek-R1-Distill-Qwen-32B`
  
Note: 
1. Remember to uncomment the server block at the beginning of the python file to run models on server.
2. If you want to run other OpenAI models, remember to add model names in the `close_source_model_lists`.

### Available Goal Types

- `detailed`: Full detailed goal specification
- `subgoal`: Simplified goal specification
- `without_hint`: Goal without hints (ALFWorld only)
- `without_detailed_goal`: Simplified goal without details (ALFWorld only)

### Running CoinCollector Experiments

Run the `interactive_CoinCollector.py` script with the following functions:

```python
# Configuration variables
model = "your_chosen_model"
folder_name = "results_folder"
result_name = "results_file"
i = 0  # starting trial index
num_trials = 10  # number of trials for single trial runs

# PDDLego+ with detailed goals (single trial batch)
run_iterative_model(model, i, i+num_trials, folder_name, result_name, goal_type="detailed")

# PDDLego+ with detailed goals (fixed 100 trials)
run_iterative_model_50(model, folder_name, result_name, goal_type="detailed")

# PDDLego+ with simple prompt (fixed 100 trials)
run_iterative_model_50(model, folder_name, result_name, goal_type="subgoal")

# PlanGen baseline (single trial batch)
run_baseline_model(model, i, i+num_trials, folder_name, result_name)

# PlanGen baseline (fixed 100 trials)
run_baseline_model_50(model, folder_name, result_name)

# PDDLego+ with fixed domain file (100 trials)
run_iterative_model_fixed_df(model, folder_name, result_name, goal_type="detailed")
```

### Running ALFWorld Experiments

Run the `interactive_Alfworld.py` script with the following functions:

```python
# Configuration variables
model = "your_chosen_model"
folder_name = "results_folder"
result_name = "results_file"
i = 0  # starting trial index
num_trials = 10  # number of trials for single trial runs

# PDDLego+ with detailed goals (single trial batch)
run_iterative_model(model, i, i+num_trials, folder_name, result_name, goal_type="detailed")

# PDDLego+ with detailed goals (fixed 100 trials)
run_iterative_model_50(model, folder_name, result_name, goal_type="detailed")

# PDDLego+ with simple prompt (fixed 100 trials)
run_iterative_model_50(model, folder_name, result_name, goal_type="subgoal")

# PDDLego+ with simple+hint prompt (fixed 100 trials)
run_iterative_model_50(model, folder_name, result_name, goal_type="without_detailed_goal")

# PDDLego+ with simple+goal prompt (fixed 100 trials)
run_iterative_model_50(model, folder_name, result_name, goal_type="without_hint")

# PlanGen baseline (single trial batch)
run_baseline_alfworld(model, i, i+num_trials, folder_name, result_name)

# PlanGen baseline (fixed 100 trials)
run_baseline_alfworld_50(model, folder_name, result_name)

# PDDLego+ with fixed domain file (100 trials)
run_iterative_model_fixed_df(model, folder_name, result_name, goal_type="detailed")
```

### Analyzing Results

#### Parsing Results

Modify the `file_name` variable in `results_parse_final.py` to generate summary statistics:

```bash
python results_parse_final.py
```

#### Extracting Action Data

Update the folder path in `extract_actions.py` to analyze action counts:

```bash
python extract_actions.py
```

#### Visualizing Results

Use the Jupyter notebook `visualize_pddlego+.ipynb` to generate visualizations of experimental results.
