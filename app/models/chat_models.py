"""
Data models for the agent system.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import AIMessage, HumanMessage


class ChatHistory(BaseChatMessageHistory):
    """Stores chat history in memory."""

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def get_messages(self) -> list:
        return self.messages

    def clear(self):
        """Clears the chat history."""
        self.messages = []


class Message(BaseModel):
    """Model for user message with session identification."""
    message: str
    session_id: str
    mode: str = "rag"
    system_prompt: Optional[str] = None

class ResponseContent(BaseModel):
    """Model for the content of a response."""
    input: str
    history: List = []
    output: str
    no_rag_output: Optional[str] = None  # Added field for non-RAG response
    intermediate_steps: List = []

class Source(BaseModel):
    """Model for source information from retrieved documents"""
    title: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None

class ResponseMessage(BaseModel):
    """Model for a complete response message with sources."""
    response: ResponseContent
    sources: List[Source] = []

    # Helper properties to access different response types
    @property
    def rag_response(self) -> str:
        """Get the RAG-based response"""
        return self.response.output

    @property
    def no_rag_response(self) -> str:
        """Get the non-RAG response"""
        return self.response.no_rag_output if self.response.no_rag_output else ""

    @property
    def has_dual_response(self) -> bool:
        """Check if the response contains both RAG and non-RAG outputs"""
        return self.response.no_rag_output is not None and self.response.no_rag_output != ""

    model_config = {
        """Pydantic model configuration"""
        "json_schema_extra": {
            "example": {
                "response": {
                    "input": "What services does MSquared offer?",
                    "output": "MSquared offers marketing attribution services that are accessible, affordable, and effective for every brand.",
                    "no_rag_output": "Based on general knowledge, MSquared appears to offer marketing and analytics services, though I don't have specific details about their current offerings.",
                    "intermediate_steps": []
                },
                "sources": [
                    {
                        "title": "MSquared Services",
                        "url": "https://msquared.com/services",
                        "content": "Service description excerpt..."
                    }
                ]
            }
        }
    }