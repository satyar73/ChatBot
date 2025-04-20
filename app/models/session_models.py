"""
Data models for chat session management.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
import json

from pydantic import BaseModel, Field


class ChatSession:
    """
    Data model representing a single chat session.
    Manages the structure and organization of prompts, responses, and relationships.
    """
    def __init__(self, session_id: str = None):
        """
        Initialize a new chat session.
        
        Args:
            session_id: Unique identifier for this session (generated if None)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.messages = []  # Chronological message history
        self.prompts_and_responses = []  # Structured prompt-response pairs
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": 0,
            "prompt_count": 0,
            "client_info": {},
            "tags": []
        }
    
    def add_user_prompt(self, 
                      prompt: str, 
                      mode: str, 
                      system_prompt: Optional[str] = None,
                      prompt_style: str = "default",
                      additional_attributes: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a user prompt to the session with complete attribute tracking.
        
        Args:
            prompt: The user's prompt text
            mode: Processing mode (rag, standard, needl, compare)
            system_prompt: Optional system prompt to guide response
            prompt_style: Style for response (default, detailed, concise)
            additional_attributes: Any additional attributes to store
            
        Returns:
            int: Index of the newly added prompt
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare attributes dict with required + optional fields
        attributes = {
            "mode": mode.lower(),
            "prompt_style": prompt_style
        }
        
        # Add system prompt if provided
        if system_prompt:
            attributes["system_prompt"] = system_prompt
            
        # Add any additional attributes
        if additional_attributes:
            attributes.update(additional_attributes)
        
        # Create message object for chronological history
        message = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": prompt,
            "timestamp": timestamp,
            "attributes": attributes
        }
        
        # Add to messages list
        self.messages.append(message)
        
        # Create prompt-response pair structure
        prompt_response_pair = {
            "id": str(uuid.uuid4()),
            "prompt": {
                "id": message["id"],  # Same ID as the message
                "content": prompt,
                "timestamp": timestamp,
                "attributes": attributes
            },
            "responses": {},  # Will store different response types
            "created_at": timestamp,
            "updated_at": timestamp
        }
        
        # Add to prompt-response pairs
        self.prompts_and_responses.append(prompt_response_pair)
        
        # Update metadata
        self.metadata["last_updated"] = timestamp
        self.metadata["message_count"] += 1
        self.metadata["prompt_count"] += 1
        
        # Return index for future reference
        return len(self.prompts_and_responses) - 1
    
    def add_response(self, 
                    prompt_index: int, 
                    response_type: str, 
                    content: str,
                    sources: Optional[List[Dict[str, str]]] = None,
                    intermediate_steps: Optional[List[Any]] = None,
                    additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a response to a specific prompt with comprehensive metadata.
        
        Args:
            prompt_index: Index of the prompt to attach response to
            response_type: Type of response (rag, standard, needl)
            content: The response text
            sources: Optional list of source references
            intermediate_steps: Optional processing steps (for transparency)
            additional_data: Any additional data to store
            
        Returns:
            str: ID of the newly added response
        """
        timestamp = datetime.now().isoformat()
        
        # Validate prompt index
        if prompt_index < 0 or prompt_index >= len(self.prompts_and_responses):
            raise ValueError(f"Invalid prompt index: {prompt_index}")
        
        # Generate ID for the response
        response_id = str(uuid.uuid4())
        
        # Create comprehensive response object
        response_obj = {
            "id": response_id,
            "type": response_type,
            "content": content,
            "timestamp": timestamp,
            "sources": sources or [],
            "intermediate_steps": intermediate_steps or [],
            "metadata": additional_data or {}
        }
        
        # Store in responses dict by type
        self.prompts_and_responses[prompt_index]["responses"][response_type] = response_obj
        self.prompts_and_responses[prompt_index]["updated_at"] = timestamp
        
        # Also add to chronological messages
        message = {
            "id": response_id,
            "role": "assistant",
            "response_type": response_type,  # Always include response_type in the message
            "content": content,
            "timestamp": timestamp,
            "sources": sources or [],
            "intermediate_steps": intermediate_steps or []
        }
        
        # Add any additional data fields
        if additional_data:
            # Also include the response type in additional_data for redundancy
            if "originalResponseType" not in additional_data:
                additional_data["originalResponseType"] = response_type
                
            # Store the complete additional_data dict
            message["additional_data"] = additional_data
            
            # Also flatten some critical fields to the message level for easier access
            for key, value in additional_data.items():
                if key not in message:  # Avoid overwriting existing fields
                    message[key] = value
            
        self.messages.append(message)
        
        # Update metadata
        self.metadata["last_updated"] = timestamp
        self.metadata["message_count"] += 1
        
        return response_id
    
    def get_responses_for_prompt(self, prompt_index: int) -> Dict[str, Any]:
        """
        Get all responses for a specific prompt.
        
        Args:
            prompt_index: Index of the prompt
            
        Returns:
            Dict mapping response types to response data
        """
        if prompt_index < 0 or prompt_index >= len(self.prompts_and_responses):
            raise ValueError(f"Invalid prompt index: {prompt_index}")
        
        return self.prompts_and_responses[prompt_index]["responses"]
    
    def get_latest_prompt_index(self) -> int:
        """Get the index of the most recent prompt, or -1 if none exist."""
        if not self.prompts_and_responses:
            return -1
        return len(self.prompts_and_responses) - 1
    
    def get_prompt_by_id(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a prompt by its ID.
        
        Args:
            prompt_id: The ID of the prompt to find
            
        Returns:
            The prompt dict or None if not found
        """
        for pair in self.prompts_and_responses:
            if pair["prompt"]["id"] == prompt_id:
                return pair["prompt"]
        return None
    
    def get_response_by_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a response by its ID.
        
        Args:
            response_id: The ID of the response to find
            
        Returns:
            The response dict or None if not found
        """
        for pair in self.prompts_and_responses:
            for response_type, response in pair["responses"].items():
                if response["id"] == response_id:
                    return response
        return None
    
    def get_prompts_by_mode(self, mode: str) -> List[Dict[str, Any]]:
        """
        Get all prompts with a specific mode.
        
        Args:
            mode: The query mode to filter by
            
        Returns:
            List of matching prompt dictionaries
        """
        normalized_mode = mode.lower()
        return [
            pair["prompt"] for pair in self.prompts_and_responses
            if pair["prompt"]["attributes"].get("mode", "").lower() == normalized_mode
        ]
    
    def get_prompt_response_pairs_by_mode(self, mode: str) -> List[Dict[str, Any]]:
        """
        Get all prompt-response pairs where the prompt has a specific mode.
        
        Args:
            mode: The query mode to filter by
            
        Returns:
            List of matching prompt-response dictionaries
        """
        normalized_mode = mode.lower()
        return [
            pair for pair in self.prompts_and_responses
            if pair["prompt"]["attributes"].get("mode", "").lower() == normalized_mode
        ]
    
    def get_chronological_messages(self, 
                                  limit: Optional[int] = None, 
                                  offset: int = 0,
                                  include_sources: bool = True) -> List[Dict[str, Any]]:
        """
        Get messages in chronological order with optional pagination.
        
        Args:
            limit: Optional maximum number of messages to return
            offset: Optional starting offset
            include_sources: Whether to include source references
            
        Returns:
            List of message dictionaries
        """
        messages = self.messages[offset:]
        if limit is not None:
            messages = messages[:limit]
            
        if not include_sources:
            # Create shallow copies without sources to reduce size
            messages = [
                {k: v for k, v in msg.items() if k != 'sources'} 
                for msg in messages
            ]
            
        return messages
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the session for categorization.
        
        Args:
            tag: The tag to add
        """
        if "tags" not in self.metadata:
            self.metadata["tags"] = []
            
        if tag not in self.metadata["tags"]:
            self.metadata["tags"].append(tag)
            self.metadata["last_updated"] = datetime.now().isoformat()
    
    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag from the session.
        
        Args:
            tag: The tag to remove
            
        Returns:
            Boolean indicating if the tag was found and removed
        """
        if "tags" in self.metadata and tag in self.metadata["tags"]:
            self.metadata["tags"].remove(tag)
            self.metadata["last_updated"] = datetime.now().isoformat()
            return True
        return False
    
    def update_client_info(self, client_info: Dict[str, Any]) -> None:
        """
        Update client information.
        
        Args:
            client_info: Dictionary of client details
        """
        self.metadata["client_info"].update(client_info)
        self.metadata["last_updated"] = datetime.now().isoformat()
    
    def purge_old_messages(self, max_age_days: int = 30) -> int:
        """
        Remove messages older than specified age.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of messages removed
        """
        if not self.messages:
            return 0
        
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        old_count = len(self.messages)
        
        # Filter messages by timestamp
        self.messages = [msg for msg in self.messages if msg["timestamp"] >= cutoff]
        
        # Calculate and return the number of removed messages
        removed_count = old_count - len(self.messages)
        if removed_count > 0:
            self.metadata["message_count"] = len(self.messages)
            self.metadata["last_updated"] = datetime.now().isoformat()
        
        return removed_count
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for serialization.
        
        Returns:
            Dictionary representation of the session
        """
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "prompts_and_responses": self.prompts_and_responses,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """
        Create session from dictionary representation.
        
        Args:
            data: Dictionary containing session data
            
        Returns:
            New ChatSession instance
        """
        session = cls(data["session_id"])
        session.messages = data.get("messages", [])
        session.prompts_and_responses = data.get("prompts_and_responses", [])
        session.metadata = data.get("metadata", {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": len(session.messages),
            "prompt_count": len(session.prompts_and_responses),
            "client_info": {},
            "tags": []
        })
        
        # Update counts if not present
        if "message_count" not in session.metadata:
            session.metadata["message_count"] = len(session.messages)
        if "prompt_count" not in session.metadata:
            session.metadata["prompt_count"] = len(session.prompts_and_responses)
            
        return session
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert session to JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ChatSession":
        """
        Create session from JSON string.
        
        Args:
            json_str: JSON string containing session data
            
        Returns:
            New ChatSession instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class SessionMetadata(BaseModel):
    """Pydantic model for session metadata in API responses."""
    session_id: str = Field(..., description="Unique identifier for the session")
    created_at: str = Field(..., description="ISO timestamp when the session was created")
    last_updated: str = Field(..., description="ISO timestamp when the session was last updated")
    message_count: int = Field(0, description="Total number of messages in the session")
    prompt_count: int = Field(0, description="Total number of prompts in the session")
    tags: List[str] = Field(default_factory=list, description="List of tags associated with the session")
    client_info: Dict[str, Any] = Field(default_factory=dict, description="Client-specific information")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "created_at": "2025-04-18T14:32:25.123456",
                "last_updated": "2025-04-18T15:45:12.654321",
                "message_count": 12,
                "prompt_count": 6,
                "tags": ["attribution", "facebook"],
                "client_info": {"brand": "LaserAway", "domain": "marketing"}
            }
        }


class SessionListItem(BaseModel):
    """Pydantic model for session list items."""
    session_id: str = Field(..., description="Unique identifier for the session")
    created_at: str = Field(..., description="ISO timestamp when the session was created")
    last_updated: str = Field(..., description="ISO timestamp when the session was last updated")
    message_count: int = Field(0, description="Total number of messages in the session")
    prompt_count: int = Field(0, description="Total number of prompts in the session")
    tags: List[str] = Field(default_factory=list, description="List of tags associated with the session")
    file_size: Optional[int] = Field(None, description="Size of the session file in bytes (if applicable)")


class SessionListResponse(BaseModel):
    """Pydantic model for session listing response."""
    sessions: List[SessionListItem] = Field(..., description="List of available sessions")
    total_count: int = Field(..., description="Total number of sessions")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(..., description="Number of sessions per page")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "sessions": [
                    {
                        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "created_at": "2025-04-18T14:32:25.123456",
                        "last_updated": "2025-04-18T15:45:12.654321",
                        "message_count": 12,
                        "prompt_count": 6,
                        "tags": ["attribution", "facebook"],
                        "file_size": 8192
                    }
                ],
                "total_count": 25,
                "page": 1,
                "page_size": 10
            }
        }


class SessionFilterOptions(BaseModel):
    """Pydantic model for session filtering options."""
    tags: Optional[List[str]] = Field(None, description="Filter sessions by tags (AND logic)")
    modes: Optional[List[str]] = Field(None, description="Filter sessions by modes (OR logic)")
    start_date: Optional[str] = Field(None, description="Filter sessions created after this date (ISO format)")
    end_date: Optional[str] = Field(None, description="Filter sessions created before this date (ISO format)")
    client_name: Optional[str] = Field(None, description="Filter sessions by client name")