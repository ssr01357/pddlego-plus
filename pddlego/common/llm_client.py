"""
LLM Client Module

Unified interface for interacting with various LLM providers.
"""

import json
import re
import os
import asyncio
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI

# Uncomment for open source models
# from kani import Kani
# from kani.engines.huggingface import HuggingEngine


class LLMClient:
    """Unified LLM client supporting multiple providers."""
    
    CLOSED_SOURCE_MODELS = [
        'gpt-4o-2024-05-13',
        'o3-mini-2025-01-31',
        'gpt-4.1-2025-04-14',
        'o4-mini-2025-04-16'
    ]
    
    def __init__(self):
        self.openai_client = OpenAI()
        # Uncomment for open source models
        # self._hf_engine_cache: Dict[str, HuggingEngine] = {}
        # self._kani_cache: Dict[str, Kani] = {}
    
    def extract_json_block(self, text: str) -> str:
        """Extract JSON content from markdown code blocks."""
        pattern = r"(?s)```json\s*(.*?)\s*```"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return text
    
    def generate_pddl(self, prompt: str, model_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate PDDL domain and problem files using LLM.
        
        Args:
            prompt: Prompt describing the task
            model_name: Name of the model to use
            
        Returns:
            Tuple of (domain_file, problem_file) or (None, None) on error
        """
        response_content = self._get_llm_response(prompt, model_name)
        
        if response_content.startswith("```json"):
            response_content = response_content.lstrip("```json").rstrip("```").strip()
        
        response_content = self.extract_json_block(response_content)
        
        try:
            result = json.loads(response_content)
            df = result.get("df", None)
            pf = result.get("pf", None)
            return df, pf
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response_content}")
            return None, None
    
    def generate_actions(self, prompt: str, model_name: str) -> Optional[list]:
        """
        Generate actions directly (baseline mode).
        
        Args:
            prompt: Prompt describing the task
            model_name: Name of the model to use
            
        Returns:
            List of actions or None on error
        """
        response_content = self._get_llm_response(prompt, model_name)
        
        if response_content.startswith("```json"):
            response_content = response_content.lstrip("```json").rstrip("```").strip()
        
        response_content = self.extract_json_block(response_content)
        
        try:
            result = json.loads(response_content)
            return result.get("actions", None)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response_content}")
            return None
    
    def _get_llm_response(self, prompt: str, model_name: str) -> str:
        """Get raw response from LLM."""
        if model_name in self.CLOSED_SOURCE_MODELS:
            return self._get_openai_response(prompt, model_name)
        elif model_name == 'deepseek':
            return self._get_deepseek_response(prompt)
        else:
            return self._get_opensource_response(prompt, model_name)
    
    def _get_openai_response(self, prompt: str, model_name: str) -> str:
        """Get response from OpenAI models."""
        if model_name == 'o4-mini-2025-04-16':
            response = self.openai_client.chat.completions.create(
                model="o4-mini-2025-04-16",
                reasoning_effort="high",
                messages=[{"role": "user", "content": prompt}]
            )
        else:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        
        return response.choices[0].message.content
    
    def _get_deepseek_response(self, prompt: str) -> str:
        """Get response from DeepSeek API."""
        deepseek_api_key = os.getenv("deepseek_API")
        client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are generating PDDL according to your observations."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        return response.choices[0].message.content
    
    def _get_opensource_response(self, prompt: str, model_name: str) -> str:
        """Get response from open source models using Kani."""
        # Uncomment when using open source models
        raise NotImplementedError("Open source model support is currently disabled")
        
        # async def _ask_model(model_name, user_prompt):
        #     if model_name not in self._hf_engine_cache:
        #         engine = HuggingEngine(
        #             model_id=model_name,
        #             use_auth_token=True,
        #             model_load_kwargs={"device_map": "auto", "trust_remote_code": True}
        #         )
        #         self._hf_engine_cache[model_name] = engine
        #         self._kani_cache[model_name] = Kani(engine, system_prompt="")
        #     ai = self._kani_cache[model_name]
        #     return await ai.chat_round_str(user_prompt)
        
        # response_content = asyncio.run(_ask_model(model_name, prompt))
        
        # # Handle DeepSeek-R1 thinking tags
        # if '</think>' in response_content:
        #     response_content = response_content[response_content.find('</think>')+10:]
        
        # return response_content