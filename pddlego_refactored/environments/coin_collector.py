"""
CoinCollector Environment Module

Interface for the CoinCollector text-based environment.
"""

from typing import List, Tuple, Dict, Any
from textworld_express import TextWorldExpressEnv

from .base_environment import BaseEnvironment
from ..config.settings import Config


class CoinCollectorEnvironment(BaseEnvironment):
    """CoinCollector environment wrapper."""
    
    def __init__(self, num_locations: int = Config.COIN_COLLECTOR_LOCATIONS,
                 step_limit: int = Config.MAX_STEPS_PER_EPISODE):
        super().__init__()
        self.env = TextWorldExpressEnv(envStepLimit=step_limit)
        self.num_locations = num_locations
        self.max_steps = step_limit
        
    def reset(self, seed: int = Config.COIN_COLLECTOR_SEED, 
              game_fold: str = "train") -> Tuple[str, Dict[str, Any]]:
        """Reset the CoinCollector environment."""
        self.env.load(
            gameName="coin",
            gameParams=f"numLocations={self.num_locations},numDistractorItems=0,includeDoors=1,limitInventorySize=0"
        )
        obs, infos = self.env.reset(seed=seed, gameFold=game_fold, generateGoldPath=True)
        
        # Update valid actions
        self.valid_actions = sorted(infos['validActions'])
        if 'look around' in self.valid_actions:
            self.valid_actions.remove('look around')
        if 'inventory' in self.valid_actions:
            self.valid_actions.remove('inventory')
        
        self.current_obs = obs
        self.step_count = 0
        
        return obs, infos
    
    def step(self, action: str) -> Tuple[str, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        obs, score, done, infos = self.env.step(action)
        self.step_count += 1
        
        # Update valid actions
        if 'validActions' in infos:
            self.valid_actions = sorted(infos['validActions'])
            if 'look around' in self.valid_actions:
                self.valid_actions.remove('look around')
            if 'inventory' in self.valid_actions:
                self.valid_actions.remove('inventory')
        
        self.current_obs = obs
        
        return obs, done, infos
    
    def get_valid_actions(self) -> List[str]:
        """Get current valid actions."""
        return self.valid_actions
    
    def summarize_observation(self, obs: str) -> str:
        """Summarize observation for LLM processing."""
        # If obs only has one line, return it
        if len(obs.split('\n')) == 1:
            return obs
        
        # Only keep where you are and location information
        lines = obs.split('\n')
        relevant_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('You are'):
                relevant_lines.append(line)
            elif 'door' in line.lower():
                relevant_lines.append(line)
            elif any(direction in line.lower() for direction in ['north', 'south', 'east', 'west']):
                relevant_lines.append(line)
        
        return '\n'.join(relevant_lines) if relevant_lines else obs
    
    def map_actions(self, actions: List[str]) -> List[str]:
        """Map PDDL actions to CoinCollector format."""
        mapped_actions = []
        
        for action in actions:
            action = action.strip().lower()
            
            if action.startswith('(') and action.endswith(')'):
                action = action[1:-1]
            
            parts = action.split()
            if not parts:
                continue
            
            action_type = parts[0]
            
            if action_type == 'move':
                if len(parts) >= 4:
                    direction = parts[3]
                    mapped_actions.append(f"go {direction}")
            elif action_type == 'open-door':
                if len(parts) >= 4:
                    direction = parts[3]
                    mapped_actions.append(f"open {direction} door")
            else:
                # Keep action as is if not recognized
                mapped_actions.append(action)
        
        return mapped_actions
    
    def is_goal_achieved(self, obs: str) -> bool:
        """Check if coin is found."""
        return "Coin found!" in obs
    
    def is_failure(self, obs: str) -> bool:
        """Check for failure conditions."""
        failure_phrases = [
            "I don't understand that.",
            "Invalid action",
            "Error:"
        ]
        return any(phrase in obs for phrase in failure_phrases)