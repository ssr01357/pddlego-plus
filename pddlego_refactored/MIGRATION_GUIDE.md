# Migration Guide: From Original to Refactored PDDLego+

This guide helps you transition from the original monolithic scripts to the refactored modular structure.

## Key Changes

### 1. **Modular Structure**
- **Before**: All functionality in two large files (`interactive_CoinCollector.py`, `interactive_Alfworld.py`)
- **After**: Functionality split into logical modules:
  - `common/`: Shared utilities (solver, LLM client, PDDL utils)
  - `environments/`: Environment-specific code
  - `experiments/`: Experiment runners
  - `config/`: Centralized configuration

### 2. **Function Mapping**

| Original Function | Refactored Location |
|------------------|-------------------|
| `run_solver()` | `common.solver.PDDLSolver.solve()` |
| `get_action_from_pddl()` | `common.solver.PDDLSolver.get_actions_from_pddl()` |
| `run_llm_model()` | `common.llm_client.LLMClient.generate_pddl()` |
| `run_gpt_for_actions_baseline()` | `common.llm_client.LLMClient.generate_actions()` |
| `apply_edit_domain/problem()` | `common.pddl_utils.apply_edit_domain/problem()` |
| `detect_duplicates()` | `common.pddl_utils.detect_duplicates()` |
| `summarize_obs()` | `environments.coin_collector.CoinCollectorEnvironment.summarize_observation()` |
| `map_actions()` | `environments.coin_collector.CoinCollectorEnvironment.map_actions()` |

### 3. **Configuration Management**
- **Before**: Hard-coded values scattered throughout code
- **After**: Centralized in `config/settings.py`

```python
# Before
close_source_model_lists = ['gpt-4o-2024-05-13','o3-mini-2025-01-31',...]
NUM_LOCATIONS = 11

# After
from pddlego_refactored.config.settings import Config
models = Config.OPENAI_MODELS
locations = Config.COIN_COLLECTOR_LOCATIONS
```

### 4. **Environment Interface**
- **Before**: Direct TextWorldExpressEnv usage
- **After**: Wrapped in environment classes with consistent interface

```python
# Before
env = TextWorldExpressEnv(envStepLimit=100)
env.load(gameName="coin", gameParams=...)
obs, infos = env.reset()

# After
from pddlego_refactored.environments.coin_collector import CoinCollectorEnvironment
env = CoinCollectorEnvironment()
obs, info = env.reset()
```

### 5. **LLM Integration**
- **Before**: Single function handling all model types
- **After**: LLMClient class with separate methods

```python
# Before
df, pf = run_llm_model(prompt, model_name)

# After
from pddlego_refactored.common.llm_client import LLMClient
llm = LLMClient()
df, pf = llm.generate_pddl(prompt, model_name)
```

## Running Experiments

### Original Way
```python
# In interactive_CoinCollector.py
if __name__ == "__main__":
    model = "gpt-4o-2024-05-13"
    folder_name = "results_folder"
    result_name = "results_file"
    
    # Run iterative model
    run_iterative_model_50(model, folder_name, result_name, goal_type="detailed")
```

### Refactored Way
```python
from pddlego_refactored.experiments.coin_collector_experiment import CoinCollectorExperiment
from pddlego_refactored.config.settings import Config

# Create experiment
experiment = CoinCollectorExperiment(
    model_name="gpt-4o-2024-05-13",
    output_folder="results_folder",
    result_file="results_file.csv"
)

# Run trials
results = experiment.run_experiment(
    start_trial=0,
    end_trial=100,
    goal_type="detailed"
)
```

## Benefits of Refactored Structure

1. **Easier Testing**: Each module can be tested independently
2. **Better Reusability**: Common functions are in shared modules
3. **Clearer Organization**: Related functionality is grouped together
4. **Easier Maintenance**: Changes to one part don't affect others
5. **Better Type Safety**: Added type hints throughout
6. **Consistent Interfaces**: All environments follow same pattern

## Next Steps

1. **Gradual Migration**: You can use both old and new code side by side
2. **Test Equivalence**: Run same experiments with both versions to verify results
3. **Add Features**: New functionality can be added to refactored code more easily
4. **Extend Environments**: Add new environments by following the base class pattern

## Notes

- The refactored code maintains the same core logic as the original
- All prompts and PDDL manipulation logic are preserved
- The external solver API integration remains the same
- VAL-master validation functions are still included but unused (as in original)