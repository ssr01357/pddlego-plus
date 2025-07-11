# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDDLego+ is a zero-shot iterative formalization and planning framework for partially observable environments. It dynamically generates and refines PDDL (Planning Domain Definition Language) representations to solve tasks in environments where complete information is not available upfront.

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
conda create -n pddlego-plus python=3.10
conda activate pddlego-plus

# Install dependencies
pip install -r requirement.txt

# Set API keys (required)
export OPENAI_API_KEY="your_openai_api_key"
export deepseek_API="your_deepseek_api_key"
```

### Running Experiments
```bash
# Run CoinCollector experiments
python interactive_CoinCollector.py

# Run ALFWorld experiments
python interactive_Alfworld.py
```

### Data Analysis
```bash
# Parse experiment results (modify file_name variable first)
python results_parse_final.py

# Extract action statistics (update folder path first)
python extract_actions.py
```

## High-Level Architecture

PDDLego+ follows an iterative "formalize-plan-execute-refine" workflow:

1. **Observation Processing**: Receives observations from environments and maintains memory of all observations and actions
2. **PDDL Generation**: Uses LLMs to generate/update PDDL domain and problem files based on observations
3. **Planning**: Sends PDDL to external solver (dual-bfws-ffparser) via REST API
4. **Execution**: Executes planned actions one by one, monitoring for failures
5. **Refinement**: When errors occur, uses error messages to prompt LLM to refine PDDL

### Key Components

- **Environment Interface**: `TextWorldExpressEnv` (CoinCollector) and `AlfredExpert`/ALFWorld
- **LLM Integration**: `run_llm_model()` - Supports multiple models with structured JSON output
- **PDDL Management**: 
  - `llm_to_pddl()` - Generates/updates PDDL files
  - `apply_edit_domain/problem()` - Applies incremental edits
  - `validate_pddl()` - Defined but unused (calls commented out)
- **Planning Integration**: `get_action_from_pddl()` - Interfaces with solver API
- **Execution Control**: Two-level retry mechanism (small loop for PDDL errors, large loop for execution failures)

### Critical Implementation Details

1. **External Dependencies**: 
   - Planning solver API: Requires external solver accessible via REST API at https://solver.planning.domains:5001
   - VAL-master tools: Functions defined but NOT actually used (all calls commented out)

2. **Error Handling Loops**:
   - Small loop (5 tries): Fixes PDDL syntax/planning errors
   - Large loop (5 tries): Handles execution failures with environment reset

3. **Goal Types** (affects prompting strategy):
   - `detailed`: Full goal description
   - `subgoal`: Simple prompt only
   - `without_hint`: Simple + goal description (ALFWorld)
   - `without_detailed_goal`: Simple + hint (ALFWorld)

4. **Fixed Domain File Mode**: Can reuse cached domain files from `df_cache/` folder

### File Structure Patterns

- Main experiment runners: `interactive_CoinCollector.py`, `interactive_Alfworld.py`
- Results stored in `output/` with subfolders per experiment
- Generated PDDL files: `domain.pddl`, `problem.pddl` (overwritten each run)
- Trial logs: Detailed text files with full execution traces
- Summary CSVs: Aggregated metrics per model/configuration

### Model Configuration

Supported models configured in `close_source_model_lists`:
- OpenAI models (gpt-4.1-2025-04-14, o3-mini-2025-01-31, o4-mini-2025-04-16, gpt-4o-2024-05-13)
- DeepSeek models (accessed via custom API)

When adding new OpenAI models, update the `close_source_model_lists` variable.

### Development Notes

- No automated testing framework - experiments are run manually
- No linting configuration - code style is inconsistent
- Server configuration blocks at beginning of main scripts need uncommenting for non-OpenAI models
- External validator paths may need adjustment for different OS (currently macOS paths)