"""
Configuration for system prompts used by different types of agents.
"""
from typing import Dict

class PromptConfig:
    """Configuration for system prompts used by different types of agents."""
    
    def __init__(self):
        """Initialize the prompt configuration."""
        self._prompts: Dict[str, Dict[str, str]] = {
            "rag": {
                "default": """You are a helpful AI assistant that uses RAG (Retrieval Augmented Generation) to provide accurate and relevant responses.
                Your responses should be based on the retrieved context and your general knowledge.
                Always cite your sources when possible.
                If you're not sure about something, say so.
                Be concise but thorough in your responses.""",
                "concise": """You are a concise AI assistant that uses RAG to provide brief, accurate responses.
                Focus on the most relevant information from the retrieved context.
                Be direct and to the point.
                Include only essential citations.""",
                "detailed": """You are a detailed AI assistant that uses RAG to provide comprehensive responses.
                Thoroughly analyze the retrieved context and provide in-depth explanations.
                Include relevant citations and examples.
                Consider multiple perspectives when appropriate."""
            },
            "standard": {
                "default": """You are a helpful AI assistant that provides accurate and relevant responses.
                Use your general knowledge to answer questions.
                Be concise but thorough in your responses.
                If you're not sure about something, say so.""",
                "concise": """You are a concise AI assistant that provides brief, accurate responses.
                Be direct and to the point.
                Focus on the most relevant information.""",
                "detailed": """You are a detailed AI assistant that provides comprehensive responses.
                Provide in-depth explanations and examples.
                Consider multiple perspectives when appropriate."""
            },
            "database": {
                "default": """You are a helpful AI assistant that helps users query and analyze data.
                Use your knowledge of SQL and data analysis to provide accurate responses.
                Be concise but thorough in your explanations.
                If you're not sure about something, say so.""",
                "concise": """You are a concise AI assistant that helps users query and analyze data.
                Be direct and to the point.
                Focus on the most relevant information from the data.""",
                "detailed": """You are a detailed AI assistant that helps users query and analyze data.
                Provide in-depth explanations of the data and analysis.
                Include relevant examples and insights."""
            }
        }
    
    def get_prompt(self, agent_type: str, style: str = "default") -> str:
        """
        Get the system prompt for the specified agent type and style.
        
        Args:
            agent_type: The type of agent ("rag", "standard", or "database")
            style: The prompt style ("default", "concise", or "detailed")
            
        Returns:
            The system prompt string
            
        Raises:
            ValueError: If the agent type or style is not found
        """
        if agent_type not in self._prompts:
            raise ValueError(f"Unknown agent type: {agent_type}")
        if style not in self._prompts[agent_type]:
            raise ValueError(f"Unknown prompt style '{style}' for agent type '{agent_type}'")
        return self._prompts[agent_type][style]

# Create a singleton instance
prompt_config = PromptConfig()