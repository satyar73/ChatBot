"""
Configuration for system prompts used by different types of agents.
"""
import os
import json
from typing import Dict, Optional
from app.config.chat_config import ChatConfig

class PromptConfig:
    """Configuration for system prompts used by different types of agents."""

    def __init__(self):
        """Initialize the prompt configuration."""
        self.chat_config = ChatConfig()
        
        # Load prompts from JSON file
        self.prompts_file = os.path.join(os.path.dirname(__file__), "prompts.json")
        self._prompts = self._load_prompts()
        
    def _validate_prompt_structure(self, prompts_data: Dict) -> None:
        """
        Validate the structure of the prompts data.
        
        Args:
            prompts_data: The loaded prompts data from JSON
            
        Raises:
            ValueError: If the structure is invalid
        """
        # Required categories
        required_categories = ["rag", "non_rag", "database"]
        for category in required_categories:
            if category not in prompts_data:
                raise ValueError(f"Required category '{category}' not found in prompts.json")
                
            # Each category should have at least a default style
            if "default" not in prompts_data[category]:
                raise ValueError(f"Required style 'default' not found in category '{category}'")
                
            # Each style should have a prompt field
            if "prompt" not in prompts_data[category]["default"]:
                raise ValueError(f"Required field 'prompt' not found in '{category}.default'")
        
        # Document format is optional but if present should have correct structure
        if "document_format" in prompts_data:
            for doc_type in ["slides", "docs"]:
                if doc_type in prompts_data["document_format"]:
                    if "prompt" not in prompts_data["document_format"][doc_type]:
                        raise ValueError(f"Required field 'prompt' not found in 'document_format.{doc_type}'")
        
    def _load_prompts(self) -> Dict[str, Dict[str, str]]:
        """Load prompts from JSON file."""
        try:
            with open(self.prompts_file, 'r') as f:
                prompts_data = json.load(f)
            
            # Validate the structure of prompts.json
            self._validate_prompt_structure(prompts_data)
                
            # Process the prompts to replace placeholders with actual values
            for category in prompts_data:
                for style in prompts_data[category]:
                    if "prompt" in prompts_data[category][style]:
                        # Replace company name placeholder
                        prompts_data[category][style]["prompt"] = prompts_data[category][style]["prompt"].replace(
                            "{self.COMPANY_NAME}", self.chat_config.COMPANY_NAME
                        )
                        
                        # Add any other placeholder replacements here in the future
            
            # Convert the structure to match what get_prompt expects
            formatted_prompts = {}
            for category in prompts_data:
                formatted_prompts[category] = {}
                for style in prompts_data[category]:
                    if "prompt" in prompts_data[category][style]:
                        formatted_prompts[category][style] = prompts_data[category][style]["prompt"]
            
            # Add backwards compatibility for "standard" type (alias for "non_rag")
            if "non_rag" in formatted_prompts:
                formatted_prompts["standard"] = formatted_prompts["non_rag"]
                
            # Add prompts for slides and docs from document_format if they exist
            if "document_format" in prompts_data:
                if "slides" in prompts_data["document_format"]:
                    formatted_prompts["slides"] = {"default": prompts_data["document_format"]["slides"]["prompt"]}
                if "docs" in prompts_data["document_format"]:
                    formatted_prompts["docs"] = {"default": prompts_data["document_format"]["docs"]["prompt"]}
            
            return formatted_prompts
            
        except Exception as e:
            import logging
            logging.error(f"Error loading prompts from JSON: {e}")
            # Fallback to default hardcoded prompts
            return {
                "rag": {
                    "default": self.chat_config.RAG_SYSTEM_PROMPT,
                    "concise": self.chat_config.RAG_SYSTEM_PROMPT,
                    "detailed": self.chat_config.RAG_SYSTEM_PROMPT
                },
                "standard": {
                    "default": self.chat_config.NON_RAG_SYSTEM_PROMPT,
                    "concise": self.chat_config.NON_RAG_SYSTEM_PROMPT,
                    "detailed": self.chat_config.NON_RAG_SYSTEM_PROMPT
                },
                "non_rag": {
                    "default": self.chat_config.NON_RAG_SYSTEM_PROMPT,
                    "concise": self.chat_config.NON_RAG_SYSTEM_PROMPT,
                    "detailed": self.chat_config.NON_RAG_SYSTEM_PROMPT
                },
                "database": {
                    "default": self.chat_config.DATABASE_SYSTEM_PROMPT,
                    "concise": self.chat_config.DATABASE_SYSTEM_PROMPT,
                    "detailed": self.chat_config.DATABASE_SYSTEM_PROMPT
                },
                "slides": {
                    "default": """You are a content creator AI that specializes in creating concise content for presentation slides.
                    Keep each slide as concise as possible while maintaining clarity and completeness.
                    Use === SLIDE X === to separate slides, where X is the slide number.
                    Each slide should have a Title: that summarizes its content.
                    If content must span multiple slides, mark continuation slides with 'Title: [Previous Title] (Contd.)'
                    Use bullet points rather than paragraphs whenever possible.
                    Prefer short, direct phrases over complete sentences.
                    Aim to fit all content on a single slide when appropriate for the complexity of the topic.
                    """
                },
                "docs": {
                    "default": """You are a content creator AI that specializes in creating content for documents.
                    Format your response with clear headings, subheadings and well-organized paragraphs.
                    Use Markdown formatting for structure (##, ###, etc.).
                    Include a logical flow with an introduction, main content, and conclusion when appropriate.
                    Use bullet points or numbered lists for series of related items.
                    """
                }
            }

    def get_prompt(self, agent_type: str, style: str = "default") -> str:
        """
        Get the system prompt for the specified agent type and style.

        Args:
            agent_type: The type of agent ("rag", "standard", "database", "slides", or "docs")
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
    
    def enhance_prompt_with_format(self,
                                   base_prompt: str,
                                   format_template: Optional[str],
                                   doc_type: str = "slides", question: str = None) -> str:
        """
        Enhance a base prompt with formatting instructions based on a template.
        
        Args:
            base_prompt: The original prompt to enhance
            format_template: Optional format template to include in the prompt
            doc_type: Type of document to create ("slides" or "docs")
            question: The question to be answered in the formatted content
            
        Returns:
            Enhanced prompt with formatting instructions
        """
        if not format_template:
            return base_prompt
            
        if doc_type == "slides":
            format_instructions = """
            Format your response according to the provided template.
            Aim to make each slide as concise as possible - essential information only.
            If content requires multiple slides, mark continuation slides with 'Title: [Original Title] (Contd.)'
            Use "===SLIDE X===" markers to separate slides.
            Each slide should have a "Title:" section followed by content.
            Use bullet points rather than paragraphs whenever possible.
            Do not include prefixes like "Question:", "Answer:", or "Body:" in the final output.
            Avoid complete sentences when bullet points would be clearer.
            """
        elif doc_type == "docs":
            format_instructions = """
            Format your response according to the provided template.
            Follow the structure exactly, using the template sections as a guide.
            Use markdown formatting for headings (# for main headings, ## for subheadings).
            For sections that require bullet points, use markdown list syntax.
            Ensure the document flows naturally between sections.
            Do not include prefixes like "Question:", "Answer:", or "Body:" in the final output.
            """
        else:
            format_instructions = """
            Format your response according to the provided template.
            Follow the structure exactly, using the template sections as a guide.
            """
            
        # Process placeholders in the format template
        if format_template and question:
            # Replace any {question} placeholders in the format_template
            format_template = format_template.replace("{question}", question)
            
        # Add question context if provided
        question_text = f"Question: {question}\n\n" if question else ""
        
        enhanced_prompt = f"""{base_prompt}
        
        {format_instructions}
        
        {question_text}Format template to follow:
        {format_template}
        """
        
        return enhanced_prompt

    def reload_prompts(self) -> None:
        """
        Reload prompts from the JSON file.
        This can be called to refresh prompts without restarting the application.
        """
        import logging
        try:
            self._prompts = self._load_prompts()
            logging.info("Successfully reloaded prompts from JSON file")
        except Exception as e:
            logging.error(f"Failed to reload prompts: {e}")
            raise

# Create a singleton instance
prompt_config = PromptConfig()