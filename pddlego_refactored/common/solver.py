"""
PDDL Solver Integration Module

This module handles communication with the external PDDL solver API.
"""

import time
import requests
from typing import Tuple, List, Optional


class PDDLSolver:
    """Interface for external PDDL solver API."""
    
    DEFAULT_SOLVER = "dual-bfws-ffparser"
    SOLVER_URL = "https://solver.planning.domains:5001"
    
    def __init__(self, solver_name: str = DEFAULT_SOLVER, max_retries: int = 3):
        self.solver_name = solver_name
        self.max_retries = max_retries
    
    def solve(self, domain_file: str, problem_file: str) -> dict:
        """
        Send PDDL files to solver and get solution.
        
        Args:
            domain_file: PDDL domain file content as string
            problem_file: PDDL problem file content as string
            
        Returns:
            dict: Solver result containing plan and any errors
        """
        req_body = {"domain": domain_file, "problem": problem_file}
        retries = 0
        
        while retries < self.max_retries:
            try:
                # Send job request to solve endpoint
                solve_request_url = requests.post(
                    f"{self.SOLVER_URL}/package/{self.solver_name}/solve", 
                    json=req_body
                ).json()
                
                # Query the result in the job
                celery_result = requests.post(
                    self.SOLVER_URL + solve_request_url['result']
                )
                
                while celery_result.json().get("status", "") == 'PENDING':
                    time.sleep(0.5)
                    celery_result = requests.post(
                        self.SOLVER_URL + solve_request_url['result']
                    )
                
                return celery_result.json()['result']
                
            except Exception as e:
                print(f"Error encountered: {e}. Retrying in 5 seconds... "
                      f"(Attempt {retries+1}/{self.max_retries})")
                retries += 1
                time.sleep(5)
        
        raise RuntimeError("Max retries exceeded. Failed to get result from solver.")
    
    def get_actions_from_pddl(self, domain_file: str, problem_file: str, 
                              action_mapper=None) -> Tuple[List[str], str]:
        """
        Get action plan from PDDL files.
        
        Args:
            domain_file: PDDL domain file content
            problem_file: PDDL problem file content
            action_mapper: Optional function to map actions to environment format
            
        Returns:
            Tuple of (actions_list, error_message)
        """
        result = self.solve(domain_file, problem_file)
        actions = result['output']['plan']
        errors = result.get('stderr', '') + result.get('stdout', '')
        
        if action_mapper and actions:
            actions = action_mapper(actions)
        
        return actions, errors