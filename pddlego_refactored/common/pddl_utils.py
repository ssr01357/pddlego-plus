"""
PDDL Utilities Module

Functions for manipulating and editing PDDL files.
"""

import re
from typing import Dict, List, Any, Tuple, Optional


def detect_duplicates(action_list: List[str], threshold: int = 3) -> bool:
    """
    Detect if actions are repeating in a pattern.
    
    Args:
        action_list: List of actions taken
        threshold: Number of times a pattern must repeat to be considered duplicate
        
    Returns:
        True if duplicate pattern detected
    """
    if len(action_list) < threshold:
        return False
    
    # Check for various pattern lengths
    for pattern_length in range(1, len(action_list) // threshold + 1):
        for start in range(len(action_list) - pattern_length * threshold + 1):
            pattern = action_list[start:start + pattern_length]
            is_duplicate = True
            
            for i in range(1, threshold):
                if action_list[start + i * pattern_length:start + (i + 1) * pattern_length] != pattern:
                    is_duplicate = False
                    break
            
            if is_duplicate:
                return True
    
    return False


def apply_edit_domain(prev_df: str, edit_json: Dict[str, Any]) -> str:
    """
    Apply edit instructions to a PDDL domain file.
    
    Args:
        prev_df: Previous domain file content
        edit_json: Edit instructions
        
    Returns:
        Updated domain file content
    """
    if not edit_json or edit_json == {}:
        return prev_df
    
    df = prev_df
    
    # Edit predicates
    if 'predicates' in edit_json:
        pred_edits = edit_json['predicates']
        
        # Find predicates section
        pred_match = re.search(r'(\(:predicates[^)]*\))', df, re.DOTALL)
        if pred_match:
            pred_section = pred_match.group(1)
            new_pred_section = pred_section
            
            # Apply edits
            if 'add' in pred_edits:
                for pred in pred_edits['add']:
                    new_pred_section = new_pred_section.rstrip(')') + f'\n    {pred}\n)'
            
            if 'replace' in pred_edits:
                for old, new in pred_edits['replace'].items():
                    new_pred_section = new_pred_section.replace(old, new)
            
            if 'delete' in pred_edits:
                for pred in pred_edits['delete']:
                    new_pred_section = new_pred_section.replace(f'\n    {pred}', '')
            
            df = df.replace(pred_section, new_pred_section)
    
    # Edit actions
    if 'action' in edit_json:
        for action_name, action_edits in edit_json['action'].items():
            # Find action section
            action_pattern = rf'(:action {action_name}.*?)(?=:action|$)'
            action_match = re.search(action_pattern, df, re.DOTALL)
            
            if action_match:
                action_section = action_match.group(1)
                new_action_section = action_section
                
                # Replace precondition
                if 'precondition' in action_edits and action_edits['precondition']:
                    prec_pattern = r':precondition\s*\([^)]*\)'
                    new_prec = f":precondition ({' '.join(action_edits['precondition'])})"
                    new_action_section = re.sub(prec_pattern, new_prec, new_action_section)
                
                # Replace effect
                if 'effect' in action_edits and action_edits['effect']:
                    effect_pattern = r':effect\s*\([^)]*\)'
                    new_effect = f":effect ({' '.join(action_edits['effect'])})"
                    new_action_section = re.sub(effect_pattern, new_effect, new_action_section)
                
                df = df.replace(action_section, new_action_section)
    
    return df


def apply_edit_problem(prev_pf: str, edit_json: Dict[str, Any]) -> str:
    """
    Apply edit instructions to a PDDL problem file.
    
    Args:
        prev_pf: Previous problem file content
        edit_json: Edit instructions
        
    Returns:
        Updated problem file content
    """
    if not edit_json or edit_json == {}:
        return prev_pf
    
    pf = prev_pf
    
    # Edit objects
    if 'objects' in edit_json:
        obj_edits = edit_json['objects']
        
        # Find objects section
        obj_match = re.search(r'(\(:objects[^)]*\))', pf, re.DOTALL)
        if obj_match:
            obj_section = obj_match.group(1)
            new_obj_section = obj_section
            
            # Apply edits
            if 'add' in obj_edits:
                for obj in obj_edits['add']:
                    new_obj_section = new_obj_section.rstrip(')') + f'\n    {obj}\n)'
            
            if 'replace' in obj_edits:
                for old, new in obj_edits['replace'].items():
                    new_obj_section = new_obj_section.replace(old, new)
            
            if 'delete' in obj_edits:
                for obj in obj_edits['delete']:
                    new_obj_section = new_obj_section.replace(f'\n    {obj}', '')
            
            pf = pf.replace(obj_section, new_obj_section)
    
    # Edit init
    if 'init' in edit_json:
        init_edits = edit_json['init']
        
        # Find init section
        init_match = re.search(r'(\(:init[^)]*\))', pf, re.DOTALL)
        if init_match:
            init_section = init_match.group(1)
            new_init_section = init_section
            
            # Apply edits
            if 'add' in init_edits:
                for init_fact in init_edits['add']:
                    new_init_section = new_init_section.rstrip(')') + f'\n    {init_fact}\n)'
            
            if 'replace' in init_edits:
                for old, new in init_edits['replace'].items():
                    new_init_section = new_init_section.replace(old, new)
            
            if 'delete' in init_edits:
                for init_fact in init_edits['delete']:
                    new_init_section = new_init_section.replace(f'\n    {init_fact}', '')
            
            pf = pf.replace(init_section, new_init_section)
    
    # Edit goal
    if 'goal' in edit_json and edit_json['goal']:
        goal_pattern = r':goal\s*\([^)]*\)'
        new_goal = f":goal ({' '.join(edit_json['goal'])})"
        pf = re.sub(goal_pattern, new_goal, pf)
    
    return pf


def apply_edits(prev_df: str, prev_pf: str, 
                edit_json_df: Dict[str, Any], 
                edit_json_pf: Dict[str, Any]) -> Tuple[str, str]:
    """
    Apply edits to both domain and problem files.
    
    Args:
        prev_df: Previous domain file
        prev_pf: Previous problem file
        edit_json_df: Domain edit instructions
        edit_json_pf: Problem edit instructions
        
    Returns:
        Tuple of (new_domain, new_problem)
    """
    new_df = apply_edit_domain(prev_df, edit_json_df)
    new_pf = apply_edit_problem(prev_pf, edit_json_pf)
    
    return new_df, new_pf