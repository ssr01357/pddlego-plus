"""
Base Environment Module

Abstract base class for environments.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


class BaseEnvironment(ABC):
    """Abstract base class for environments."""
    
    def __init__(self):
        self.current_obs = None
        self.valid_actions = []
        self.step_count = 0
        self.max_steps = 100
    
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment.
        
        Returns:
            Tuple of (observation, info_dict)
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, done, info_dict)
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> List[str]:
        """Get list of valid actions in current state."""
        pass
    
    @abstractmethod
    def summarize_observation(self, obs: str) -> str:
        """
        Summarize observation for LLM processing.
        
        Args:
            obs: Raw observation
            
        Returns:
            Summarized observation
        """
        pass
    
    @abstractmethod
    def map_actions(self, actions: List[str]) -> List[str]:
        """
        Map PDDL actions to environment-specific format.
        
        Args:
            actions: List of PDDL actions
            
        Returns:
            List of environment-specific actions
        """
        pass
    
    def is_goal_achieved(self, obs: str) -> bool:
        """
        Check if goal is achieved.
        
        Args:
            obs: Current observation
            
        Returns:
            True if goal achieved
        """
        return False
    
    def is_failure(self, obs: str) -> bool:
        """
        Check if current state is a failure.
        
        Args:
            obs: Current observation
            
        Returns:
            True if failure state
        """
        return False