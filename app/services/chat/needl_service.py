"""
Service for interacting with the Needl.ai API.
"""
import os
import aiohttp
from typing import Dict, Any, Optional
import urllib.parse
from app.utils.logging_utils import get_logger

class NeedlService:
    """
    Service for making queries to the Needl.ai API.
    """
    
    def __init__(self):
        """Initialize the Needl service."""
        self.api_key = os.environ.get("NEEDL_API_KEY")
        self.base_url = "https://api.needl.ai/prod/enterprise/ask-needl"
        self.logger = get_logger(__name__)
        
        if not self.api_key:
            self.logger.warning("NEEDL_API_KEY not found in environment variables")
    
    async def query(self, prompt: str, generate_answer: bool = True, timezone: str = "UTC", pro: bool = True) -> Dict[str, Any]:
        """
        Query the Needl.ai API with a prompt.
        
        Args:
            prompt: The question to ask Needl
            generate_answer: Whether to generate an answer (default True)
            timezone: The timezone to use (default UTC)
            pro: Whether to use pro features (default True)
            
        Returns:
            The API response as a dictionary
        """
        if not self.api_key:
            return {
                "status": "ERROR",
                "message": "NEEDL_API_KEY environment variable not set"
            }
        
        # URL encode the prompt
        encoded_prompt = urllib.parse.quote(prompt)
        self.logger.info(f"Encoded prompt: {encoded_prompt}")
        
        # Build the URL with query parameters
        url = (f"{self.base_url}?prompt={encoded_prompt}&"
               f"generate_answer={str(generate_answer).lower()}"
               f"&timezone={timezone}&pro={str(pro).lower()}")
        
        # Set up headers with the API key
        headers = {
            'accept': 'application/json',
            'x-api-key': self.api_key
        }
        
        try:
            self.logger.info(f"Querying Needl API with prompt: {prompt[:50]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    # Check if the request was successful
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info("Successfully received response from Needl API")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Needl API request failed with status {response.status}: {error_text}")
                        return {
                            "status": "ERROR",
                            "message": f"API request failed with status {response.status}",
                            "details": error_text
                        }
        
        except Exception as e:
            self.logger.error(f"Error querying Needl API: {str(e)}")
            return {
                "status": "ERROR",
                "message": f"Exception while querying Needl API: {str(e)}"
            }

    def format_response_as_chat_message(self, needl_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the Needl API response as a chat message compatible with the chat service.
        
        Args:
            needl_response: The Needl API response
            
        Returns:
            A formatted response that matches the chat service's format
        """
        if needl_response.get("status") != "SUCCESS":
            return {
                "response": {
                    "input": needl_response.get("query_params", {}).get("prompt", ""),
                    "output": f"Error from Needl API: {needl_response.get('message', 'Unknown error')}",
                    "history": []
                },
                "sources": []
            }
        
        # Extract the answer
        answer = needl_response.get("generated_answer", {}).get("answer", "")
        
        # Format sources
        sources = []
        for sentence in needl_response.get("generated_answer", {}).get("sentences", []):
            for citation in sentence.get("citations", []):
                source = {
                    "title": citation.get("document_id", "Unknown document"),
                    "url": citation.get("original_source_link", citation.get("needl_document_link", "")),
                    "content": citation.get("context", "")
                }
                if source not in sources:  # Avoid duplicates
                    sources.append(source)
        
        # Also add retrieved results that might not be explicitly cited
        for result in needl_response.get("retrieved_results", []):
            source = {
                "title": result.get("document_id", "Unknown document"),
                "url": result.get("original_source_link", result.get("needl_document_link", "")),
                "content": "\n".join(result.get("highlights", []))
            }
            if source not in sources:  # Avoid duplicates
                sources.append(source)
        
        return {
            "response": {
                "input": needl_response.get("query_params", {}).get("prompt", ""),
                "output": answer,
                "history": []
            },
            "sources": sources
        }

# Create a singleton instance
needl_service = NeedlService()