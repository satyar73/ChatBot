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
        # Check if the exact same message already exists in the history
        # This prevents duplicate message issues when forwarding to LLM
        
        # First check if this exact message already exists in history
        for existing_msg in self.messages:
            if isinstance(existing_msg, HumanMessage) and existing_msg.content == message:
                # Message already exists in history, don't add again
                return
                
        # Message doesn't exist in history, add it (making sure not to add right after another user message)
        if not self.messages or not isinstance(self.messages[-1], HumanMessage):
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
    message: str = Field(..., description="The user's message or question")
    session_id: str = Field(..., description="Unique identifier for the chat session")
    mode: str = Field("rag", description="Query mode: 'rag' (knowledge-enhanced), 'no_rag' (general knowledge only), or 'compare' (both responses)")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt to override the default behavior")
    prompt_style: Optional[str] = Field("default", description="Response style: 'default', 'detailed' (comprehensive), or 'concise' (brief)")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Additional query parameters: 'client_name' (specific client data), 'content_type' (blog/article/product), 'topic' (attribution/testing/etc)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What are the benefits of multi-touch attribution?",
                    "session_id": "user123",
                    "mode": "rag",
                    "prompt_style": "default"
                },
                {
                    "message": "How did LaserAway's Facebook campaign perform?",
                    "session_id": "user456",
                    "mode": "rag",
                    "metadata": {
                        "client_name": "LaserAway",
                        "topic": "facebook"
                    }
                }
            ]
        }
    }

class ResponseContent(BaseModel):
    """Model for the content of a response."""
    input: str = Field(..., description="The original user input/question")
    history: List[Any] = Field(default_factory=lambda: [], description="List of previous interactions in the conversation")
    output: str = Field(..., description="The AI response (RAG-based if mode is 'rag' or 'compare')")
    no_rag_output: Optional[str] = Field(None, description="Response without RAG enhancements (only included if mode is 'compare' or 'no_rag')")
    intermediate_steps: List = Field([], description="Debug information about the reasoning process (if enabled)")

class Source(BaseModel):
    """Model for source information from retrieved documents"""
    title: Optional[str] = Field(None, description="Title of the source document")
    url: Optional[str] = Field(None, description="URL or reference to the source document")
    content: Optional[str] = Field(None, description="Relevant excerpt from the source document")

class ResponseMessage(BaseModel):
    """Model for a complete response message with sources."""
    response: ResponseContent = Field(..., description="The AI response content, including original input and outputs")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the response (empty for non-RAG responses)")

    # Helper properties to access different response types
    @property
    def rag_response(self) -> str:
        """
        Get the knowledge-enhanced (RAG) response.
        Always available regardless of the mode used.
        """
        return self.response.output

    @property
    def no_rag_response(self) -> str:
        """
        Get the response without knowledge enhancement.
        Only available if mode was 'no_rag' or 'compare'.
        Returns empty string if not available.
        """
        return self.response.no_rag_output if self.response.no_rag_output else ""

    @property
    def has_dual_response(self) -> bool:
        """
        Check if both RAG and non-RAG responses are included.
        Returns True if mode was 'compare' and both responses were generated.
        """
        return self.response.no_rag_output is not None and self.response.no_rag_output != ""

    model_config = {
        """Pydantic model configuration"""
        "json_schema_extra": {
            "example": {
                "response": {
                    "input": "What services does ChatBot offer?",
                    "output": "ChatBot offers marketing attribution services that are accessible, affordable, and effective for every brand.",
                    "no_rag_output": "Based on general knowledge, ChatBot appears to offer marketing and analytics services, though I don't have specific details about their current offerings.",
                    "intermediate_steps": []
                },
                "sources": [
                    {
                        "title": "ChatBot Services",
                        "url": "https://chatbot.com/services",
                        "content": "Service description excerpt..."
                    }
                ]
            }
        }
    }