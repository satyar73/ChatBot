"""
Configuration for system prompts used by different types of agents.
"""
import json
from typing import Dict, Optional
from pathlib import Path

class PromptConfig:
    """
    Configuration class for managing system prompts used by different types of agents.
    """
    def __init__(self):
        # Load prompts from prompts.json
        prompts_path = Path(__file__).parent / "prompts.json"
        with open(prompts_path, 'r') as f:
            prompts_data = json.load(f)
            
        # Initialize prompts dictionary
        self._prompts = {}
        
        # Extract prompts from the loaded data
        for agent_type, styles in prompts_data.items():
            self._prompts[agent_type] = {}
            for style, data in styles.items():
                self._prompts[agent_type][style] = data["prompt"]

    def get_prompt(self, agent_type: str, style: str = "default") -> str:
        """
        Get the appropriate system prompt based on agent type and style.
        
        Args:
            agent_type: Type of agent ("rag", "non_rag", or "database")
            style: Prompt style ("default", "technical", "simple", or "detailed")
            
        Returns:
            The appropriate system prompt
        """
        if agent_type not in self._prompts:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        if style not in self._prompts[agent_type]:
            raise ValueError(f"Unknown prompt style: {style}")
            
        return self._prompts[agent_type][style]

# Create a singleton instance
prompt_config = PromptConfig()