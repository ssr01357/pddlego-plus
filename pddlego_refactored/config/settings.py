"""
Configuration Settings

Central configuration for PDDLego+ experiments.
"""

import os
from typing import List, Dict, Any


class Config:
    """Configuration settings for PDDLego+."""
    
    # Model configurations
    OPENAI_MODELS: List[str] = [
        'gpt-4o-2024-05-13',
        'o3-mini-2025-01-31',
        'gpt-4.1-2025-04-14',
        'o4-mini-2025-04-16'
    ]
    
    DEEPSEEK_MODELS: List[str] = [
        'deepseek',  # DeepSeek-R1-671B
        'DeepSeek-R1-Distill-Llama-70B',
        'DeepSeek-R1-Distill-Qwen-32B'
    ]
    
    # API configurations
    SOLVER_URL: str = "https://solver.planning.domains:5001"
    DEFAULT_SOLVER: str = "dual-bfws-ffparser"
    SOLVER_MAX_RETRIES: int = 3
    
    # Experiment configurations
    MAX_STEPS_PER_EPISODE: int = 100
    SMALL_LOOP_RETRIES: int = 5  # Retries for PDDL errors
    LARGE_LOOP_RETRIES: int = 5  # Retries for execution failures
    
    # CoinCollector configurations
    COIN_COLLECTOR_LOCATIONS: int = 11
    COIN_COLLECTOR_SEED: int = 1
    
    # Output configurations
    OUTPUT_DIR: str = "output"
    DOMAIN_FILE_CACHE: str = "df_cache"
    
    # File paths
    DOMAIN_FILE: str = "domain.pddl"
    PROBLEM_FILE: str = "problem.pddl"
    PLAN_FILE: str = "plan.txt"
    
    # Goal type configurations
    GOAL_TYPES: List[str] = [
        "detailed",           # Full goal description
        "subgoal",           # Simple prompt only
        "without_hint",      # Simple + goal description (ALFWorld)
        "without_detailed_goal"  # Simple + hint (ALFWorld)
    ]
    
    # Environment variables
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("deepseek_API", "")
    
    # Prompt templates
    PDDL_OUTPUT_FORMAT: str = """
        Please provide the output in strict JSON format, without any additional text or explanation, including a PDDL domain file as 'df' and a PDDL problem file as 'pf'. 
        The format should strictly be:
            {
            "df": "...",
            "pf": "..."
            }
    """
    
    EDIT_OUTPUT_FORMAT: str = """
        Please provide the output in JSON format, including the edit suggestions for a domain file as 'df' and the edit suggestions for a problem file as 'pf'. 
        The output format should be: {"df": "...", "pf": "..."}
        You will modify the following df and pf using add, delete, and replace operations (in a JSON format). 
        You SHOULD NOT provide a domain file and a problem file directly.
    """
    
    ACTION_OUTPUT_FORMAT: str = """
        Please provide the output in strict JSON format, without any additional text or explanation.
        The format should strictly be:
        {
            "actions": ["action1", "action2", ...]
        }
    """
    
    @classmethod
    def get_model_list(cls) -> List[str]:
        """Get complete list of supported models."""
        return cls.OPENAI_MODELS + cls.DEEPSEEK_MODELS
    
    @classmethod
    def is_closed_source_model(cls, model_name: str) -> bool:
        """Check if model is closed source."""
        return model_name in cls.OPENAI_MODELS or model_name in cls.DEEPSEEK_MODELS
    
    @classmethod
    def get_output_path(cls, folder_name: str, filename: str) -> str:
        """Get full path for output file."""
        return os.path.join(cls.OUTPUT_DIR, folder_name, filename)