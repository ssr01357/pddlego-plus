"""
Base Experiment Module

Abstract base class for experiments.
"""

import os
import csv
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from ..config.settings import Config
from ..common.llm_client import LLMClient
from ..common.solver import PDDLSolver
from ..environments.base_environment import BaseEnvironment


class BaseExperiment(ABC):
    """Abstract base class for experiments."""
    
    def __init__(self, environment: BaseEnvironment, model_name: str,
                 output_folder: str, result_file: str):
        self.env = environment
        self.model_name = model_name
        self.output_folder = output_folder
        self.result_file = result_file
        
        self.llm_client = LLMClient()
        self.solver = PDDLSolver()
        
        # Ensure output directory exists
        os.makedirs(os.path.join(Config.OUTPUT_DIR, output_folder), exist_ok=True)
        
        # Initialize results CSV
        self.csv_path = os.path.join(Config.OUTPUT_DIR, result_file)
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date', 'model', 'experiment_type', 'environment', 
                    'goal_type', 'trial_id', 'success', 'steps', 
                    'final_action', 'action_history'
                ])
    
    @abstractmethod
    def run_trial(self, trial_id: int, goal_type: str = "detailed") -> Dict[str, Any]:
        """
        Run a single trial.
        
        Args:
            trial_id: ID of the trial
            goal_type: Type of goal specification
            
        Returns:
            Dictionary with trial results
        """
        pass
    
    def run_experiment(self, start_trial: int, end_trial: int, 
                      goal_type: str = "detailed") -> List[Dict[str, Any]]:
        """
        Run multiple trials.
        
        Args:
            start_trial: Starting trial ID
            end_trial: Ending trial ID (exclusive)
            goal_type: Type of goal specification
            
        Returns:
            List of trial results
        """
        results = []
        
        for trial_id in range(start_trial, end_trial):
            print(f"\n{'='*50}")
            print(f"Running trial {trial_id} with model {self.model_name}")
            print(f"Goal type: {goal_type}")
            print(f"{'='*50}")
            
            try:
                result = self.run_trial(trial_id, goal_type)
                results.append(result)
                self._save_result(result)
                
                print(f"\nTrial {trial_id} completed: {'SUCCESS' if result['success'] else 'FAILURE'}")
                print(f"Steps taken: {result['steps']}")
                
            except Exception as e:
                print(f"\nError in trial {trial_id}: {e}")
                error_result = {
                    'trial_id': trial_id,
                    'success': False,
                    'steps': 0,
                    'error': str(e),
                    'model': self.model_name,
                    'goal_type': goal_type
                }
                results.append(error_result)
                self._save_result(error_result)
        
        return results
    
    def _save_result(self, result: Dict[str, Any]):
        """Save result to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                result.get('model', self.model_name),
                result.get('experiment_type', 'unknown'),
                result.get('environment', 'unknown'),
                result.get('goal_type', 'unknown'),
                result.get('trial_id', -1),
                result.get('success', False),
                result.get('steps', 0),
                result.get('final_action', ''),
                str(result.get('action_history', []))
            ])
    
    def _save_trial_log(self, trial_id: int, content: str):
        """Save detailed trial log."""
        log_path = os.path.join(
            Config.OUTPUT_DIR, 
            self.output_folder,
            f"trial_{trial_id}_{self.model_name}.txt"
        )
        with open(log_path, 'w') as f:
            f.write(content)