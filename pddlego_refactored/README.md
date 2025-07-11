# PDDLego+ Refactored

This is a refactored version of the PDDLego+ codebase with improved modularity and maintainability.

## Key Improvements

1. **Modular Architecture**: Common functionality extracted into reusable modules
2. **Clear Separation**: Environment-specific code separated from core logic
3. **Configuration Management**: Centralized configuration system
4. **Better Error Handling**: Improved logging and error management
5. **Type Hints**: Added type annotations for better code clarity

## Directory Structure

```
pddlego_refactored/
├── common/           # Shared modules used by both environments
│   ├── solver.py     # PDDL solver integration
│   ├── llm_client.py # Unified LLM interface
│   ├── file_utils.py # File I/O utilities
│   └── pddl_utils.py # PDDL manipulation utilities
├── environments/     # Environment-specific implementations
│   ├── coin_collector.py
│   └── alfworld.py
├── experiments/      # Experiment runner modules
│   ├── base_experiment.py
│   ├── iterative_experiment.py
│   └── baseline_experiment.py
├── config/          # Configuration files
│   └── settings.py
└── utils/           # Additional utilities
    └── logging.py
```

## Usage

### Running CoinCollector Experiments

```python
from pddlego_refactored.experiments import run_coin_collector_experiment

# Run PDDLego+ approach
run_coin_collector_experiment(
    model="gpt-4o-2024-05-13",
    experiment_type="iterative",
    goal_type="detailed",
    num_trials=100
)

# Run baseline (direct action generation)
run_coin_collector_experiment(
    model="gpt-4o-2024-05-13", 
    experiment_type="baseline",
    num_trials=100
)
```

### Running ALFWorld Experiments

```python
from pddlego_refactored.experiments import run_alfworld_experiment

# Run PDDLego+ approach
run_alfworld_experiment(
    model="o3-mini-2025-01-31",
    experiment_type="iterative", 
    goal_type="detailed",
    num_trials=100
)
```

## Configuration

Edit `config/settings.py` to configure:
- Model lists
- API endpoints
- Experiment parameters
- Output directories