"""
Example Usage of Refactored PDDLego+

This script demonstrates how to use the refactored codebase.
"""

import os
import sys

# Add parent directory to path to import pddlego_refactored
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pddlego_refactored.environments.coin_collector import CoinCollectorEnvironment
from pddlego_refactored.common.llm_client import LLMClient
from pddlego_refactored.common.solver import PDDLSolver
from pddlego_refactored.common.pddl_utils import detect_duplicates, apply_edits
from pddlego_refactored.config.settings import Config


def demo_coin_collector():
    """Demonstrate basic CoinCollector functionality."""
    print("=== PDDLego+ Refactored Demo ===\n")
    
    # Initialize environment
    env = CoinCollectorEnvironment()
    obs, info = env.reset()
    
    print("Initial observation:")
    print(env.summarize_observation(obs))
    print(f"\nValid actions: {env.get_valid_actions()}")
    
    # Initialize LLM client and solver
    llm = LLMClient()
    solver = PDDLSolver()
    
    # Example: Generate PDDL from observation
    prompt = f"""
    {Config.PDDL_OUTPUT_FORMAT}
    
    You are in a CoinCollector environment. Generate PDDL domain and problem files.
    
    Current observation: {env.summarize_observation(obs)}
    Valid actions: {env.get_valid_actions()}
    
    The domain should include:
    - Types: location, direction
    - Predicates: at, connected, door-open, coin-at
    - Actions: move, open-door
    
    The goal is to find and collect the coin.
    """
    
    print("\n=== Generating PDDL with LLM ===")
    # This would normally call the LLM, but for demo we'll show the structure
    print("Would send prompt to LLM model...")
    
    # Example of using the solver (with dummy PDDL)
    domain_example = """
    (define (domain coin-collector)
        (:requirements :strips :typing)
        (:types location direction)
        (:predicates
            (at ?loc - location)
            (connected ?from - location ?to - location ?dir - direction)
            (door-open ?from - location ?to - location)
        )
        (:action move
            :parameters (?from - location ?to - location ?dir - direction)
            :precondition (and (at ?from) (connected ?from ?to ?dir) (door-open ?from ?to))
            :effect (and (not (at ?from)) (at ?to))
        )
    )
    """
    
    problem_example = """
    (define (problem coin-search)
        (:domain coin-collector)
        (:objects 
            room1 room2 - location
            north south east west - direction
        )
        (:init
            (at room1)
            (connected room1 room2 north)
            (door-open room1 room2)
        )
        (:goal (at room2))
    )
    """
    
    print("\n=== Example PDDL Structure ===")
    print("Domain file structure:")
    print(domain_example[:200] + "...")
    print("\nProblem file structure:")
    print(problem_example[:200] + "...")
    
    # Demonstrate action mapping
    pddl_actions = ["(move room1 room2 north)", "(open-door room2 room3 east)"]
    env_actions = env.map_actions(pddl_actions)
    print(f"\n=== Action Mapping ===")
    print(f"PDDL actions: {pddl_actions}")
    print(f"Environment actions: {env_actions}")
    
    # Demonstrate duplicate detection
    action_history = ["go north", "go south", "go north", "go south", "go north", "go south"]
    has_duplicates = detect_duplicates(action_history, threshold=3)
    print(f"\n=== Duplicate Detection ===")
    print(f"Action history: {action_history}")
    print(f"Contains repeating pattern: {has_duplicates}")
    
    print("\n=== Demo Complete ===")
    print("This demonstrates the key components of the refactored PDDLego+ system.")
    print("The actual experiments would involve:")
    print("1. Iteratively generating/refining PDDL based on observations")
    print("2. Using the solver to get action plans")
    print("3. Executing actions and updating based on feedback")
    print("4. Handling errors and retrying with refined PDDL")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set")
    
    # Run demo
    demo_coin_collector()