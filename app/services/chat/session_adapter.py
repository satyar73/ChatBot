"""
Adapter service for bridging between LangChain ChatHistory and our ChatSession models.
"""
from typing import List, Optional, Dict, Any

from langchain.schema import AIMessage, HumanMessage

from app.models.chat_models import ChatHistory
from app.services.chat.session_service import session_manager
from app.utils.logging_utils import get_logger

class SessionAdapter:
    """
    Service for adapting between LangChain and ChatSession models.
    Provides methods to convert and sync between different chat history formats.
    """
    
    def __init__(self):
        """Initialize the session adapter."""
        self.logger = get_logger(__name__)
    
    def get_chat_history(self, session_id: str) -> ChatHistory:
        """
        Get a LangChain ChatHistory from a persistent ChatSession.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            A LangChain-compatible ChatHistory
        """
        # Get or create the session
        session = session_manager.get_session(session_id)
        assert session is not None, f"Expected session for ID '{session_id}', but got None."

        # Create a new ChatHistory
        chat_history = ChatHistory()
        
        # Convert messages from the session to LangChain format
        for message in session.messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "user":
                chat_history.add_user_message(content)
            elif role == "assistant":
                # Preserve the response_type when converting to LangChain message
                response_type = message.get("response_type")
                
                # Create additional metadata from any other important attributes
                additional_metadata = {}
                for field in ["originalMode", "isNeedlResponse", "type"]:
                    if field in message:
                        additional_metadata[field] = message[field]
                
                # Log the preserved response type for debugging
                self.logger.info(f"Preserving response_type={response_type} when loading message into chat history")
                
                # Add the AI message with preserved response_type and metadata
                chat_history.add_ai_message(
                    message=content, 
                    response_type=response_type,
                    additional_metadata=additional_metadata
                )
        
        return chat_history
    
    def save_from_chat_history(self, 
                              session_id: str, 
                              chat_history: ChatHistory, 
                              mode: str = "rag",
                              system_prompt: Optional[str] = None,
                              prompt_style: str = "default",
                              additional_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a LangChain ChatHistory to our persistent ChatSession.
        
        Args:
            session_id: The session ID to save to
            chat_history: The LangChain ChatHistory to save
            mode: The chat mode (rag, standard, needl, compare)
            system_prompt: Optional system prompt
            prompt_style: The prompt style
            additional_metadata: Any additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the session
            session = session_manager.get_session(session_id)
            assert session is not None, f"Expected session for ID '{session_id}', but got None."
            
            # Create a lookup of existing messages by content to avoid adding duplicates
            # and track existing prompts
            existing_responses = {}
            existing_response_contents = set()  # Just track content for de-duplication
            user_prompts_map = {}
            
            # Map existing user prompts to their indices
            for i, pair in enumerate(session.prompts_and_responses):
                prompt_content = pair["prompt"]["content"]
                user_prompts_map[prompt_content] = i
                
                # Also track existing responses by content and type
                for response_type, response in pair["responses"].items():
                    content = response.get("content", "")
                    existing_responses[(content, response_type)] = response.get("id", "")
                    # Add content to the set for faster lookups
                    existing_response_contents.add(content)
            
            # Add metadata if provided
            if additional_metadata:
                session.metadata.update(additional_metadata)
            
            # Get all messages from chat history
            messages = chat_history.get_messages()
            self.logger.info(f"Processing {len(messages)} messages from chat history for session {session_id}")
            
            # Process messages to add/update
            i = 0
            while i < len(messages):
                if isinstance(messages[i], HumanMessage):
                    # Get user prompt content
                    prompt_text = messages[i].content
                    
                    # Check if this prompt already exists
                    if prompt_text in user_prompts_map:
                        # Use existing prompt index
                        prompt_index = user_prompts_map[prompt_text]
                        self.logger.info(f"Found existing prompt: {prompt_text[:30]}...")
                    else:
                        # Add new user prompt
                        prompt_index = session.add_user_prompt(
                            prompt=prompt_text,
                            mode=mode,
                            system_prompt=system_prompt,
                            prompt_style=prompt_style
                        )
                        user_prompts_map[prompt_text] = prompt_index
                    
                    # Check for corresponding AI messages
                    while i + 1 < len(messages) and isinstance(messages[i + 1], AIMessage):
                        i += 1
                        ai_message = messages[i]
                        ai_content = ai_message.content
                        
                        # Get response_type, treating it as immutable once set
                        message_response_type = getattr(ai_message, "response_type", mode)
                        
                        # Extract any additional data
                        additional_data = {}
                        for attr_name in ["originalMode", "isNeedlResponse", "type", "hiddenContent"]:
                            if hasattr(ai_message, attr_name):
                                additional_data[attr_name] = getattr(ai_message, attr_name)
                                self.logger.info(f"Found metadata on message: {attr_name}={getattr(ai_message, attr_name)}")
                        
                        # Check if we already have this exact response content, regardless of type
                        if ai_content in existing_response_contents:
                            # Skip duplicate response content
                            self.logger.info(f"Skipping duplicate response content (ignoring type): {ai_content[:50]}...")
                        else:
                            # Alternatively, look for exact content+type match
                            content_and_type = (ai_content, message_response_type)
                            if content_and_type in existing_responses:
                                # Skip duplicate response by content+type
                                self.logger.info(f"Skipping duplicate response with type={message_response_type}")
                            else:
                                # Store the message ID if available
                                if hasattr(ai_message, "id") and ai_message.id:
                                    additional_data["original_message_id"] = ai_message.id
                                
                                # Add detailed logging
                                self.logger.info(f"Adding new response with type={message_response_type} to prompt: {prompt_text[:30]}...")
                                
                                # Add the new response
                                response_id = session.add_response(
                                    prompt_index=prompt_index,
                                    response_type=message_response_type,
                                    content=ai_content,
                                    additional_data=additional_data
                                )
                                existing_responses[content_and_type] = response_id
                                existing_response_contents.add(ai_content)  # Add to content tracking
                                self.logger.info(f"Added new response with ID={response_id} and type={message_response_type}")
                    
                    i += 1
                else:
                    self.logger.warning(f"Unexpected message type in chat history: {type(messages[i])}")
                    i += 1
            
            # Update message count in metadata
            session.metadata["message_count"] = len(session.messages)
            session.metadata["prompt_count"] = len(session.prompts_and_responses)
            
            # Save the session
            return session_manager.save_session(session)
            
        except Exception as e:
            self.logger.error(f"Error saving chat history to session: {e}")
            return False
    
    def update_from_message(self,
                           session_id: str,
                           is_user: bool,
                           content: str,
                           mode: str = "rag",
                           system_prompt: Optional[str] = None,
                           prompt_style: str = "default",
                           additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a session with a new message.
        
        Args:
            session_id: The session ID to update
            is_user: Whether this is a user message
            content: The message content
            mode: The chat mode (for user messages)
            system_prompt: The system prompt (for user messages)
            prompt_style: The prompt style (for user messages)
            additional_data: Any additional data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the session
            session = session_manager.get_session(session_id)
            
            if is_user:
                # Add user prompt
                session.add_user_prompt(
                    prompt=content,
                    mode=mode,
                    system_prompt=system_prompt,
                    prompt_style=prompt_style,
                    additional_attributes=additional_data
                )
            else:
                # Add assistant response to the latest prompt
                latest_prompt_index = session.get_latest_prompt_index()
                if latest_prompt_index >= 0:
                    session.add_response(
                        prompt_index=latest_prompt_index,
                        response_type=mode,
                        content=content,
                        additional_data=additional_data
                    )
                else:
                    self.logger.error(f"Cannot add assistant response: no prompts in session {session_id}")
                    return False
            
            # Save the session
            return session_manager.save_session(session)
            
        except Exception as e:
            self.logger.error(f"Error updating session with message: {e}")
            return False
    
    def get_messages_for_api(self, 
                           session_id: str, 
                           limit: Optional[int] = None,
                           include_sources: bool = False,
                           mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get messages in a format suitable for API responses.
        
        Args:
            session_id: The session ID to retrieve
            limit: Optional maximum number of messages to return
            include_sources: Whether to include source information
            mode: Optional mode to filter messages by (rag, no_rag/standard, needl, compare)
            
        Returns:
            List of message dictionaries
        """
        try:
            session = session_manager.get_session(session_id)
            messages = session.get_chronological_messages(limit=limit, include_sources=include_sources)
            
            # Format messages for API
            formatted_messages = []
            
            # Map mode names for compatibility (frontend/backend sometimes use different terms)
            mode_mapping = {
                "standard": "no_rag",
                "no_rag": "standard"
            }
            
            # Special case for "all" mode or no mode - skip filtering entirely
            if not mode or mode.lower() == "all":
                self.logger.info(f"Mode='{mode}' (all or None), skipping message filtering")
                # Just format all messages and return them
                for msg in messages:
                    formatted_msg = {
                        "id": msg.get("id", ""),
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "timestamp": msg.get("timestamp", "")
                    }
                    
                    # Include response type if it's an assistant message
                    if msg.get("role") == "assistant" and "response_type" in msg:
                        formatted_msg["response_type"] = msg["response_type"]
                    
                    # Include sources if requested
                    if include_sources and "sources" in msg and msg["sources"]:
                        formatted_msg["sources"] = msg["sources"]
                    
                    formatted_messages.append(formatted_msg)
                    
                return formatted_messages
                
            # Regular filtering for specific modes
            # Keep track of user messages that should be included
            user_messages_to_include = set()
            
            # First pass: identify assistant messages that match the mode and their corresponding user messages
            if mode:
                for i, msg in enumerate(messages):
                    # Check if this is an assistant message that matches the requested mode
                    if msg.get("role") == "assistant":
                        # Get response_type from various possible locations
                        response_type = msg.get("response_type") 
                        
                        # Simple mode matching - treat response_type as immutable
                        mode_match = (
                            response_type == mode or  # Direct match
                            (response_type in mode_mapping and mode_mapping[response_type] == mode) or  # Mapped match
                            (mode in mode_mapping and mode_mapping[mode] == response_type)  # Reverse mapped match
                        )
                        
                        if mode_match:
                            # Find the preceding user message if any
                            for j in range(i-1, -1, -1):
                                if messages[j].get("role") == "user":
                                    user_messages_to_include.add(messages[j].get("id", ""))
                                    break
            
            # Second pass: build the formatted message list
            for msg in messages:
                # If filtering by mode, apply filters
                if mode:
                    # For assistant messages, check if the response type matches
                    if msg.get("role") == "assistant":
                        # Get the response_type - treat it as immutable
                        response_type = msg.get("response_type")
                        
                        # Log for detailed debugging
                        self.logger.info(f"Filtering message - requested mode={mode}, " +
                                      f"message response_type={response_type}")
                        
                        # Simple strict mode matching logic
                        # Only match exact response type to avoid mixing modes
                        mode_match = (
                            response_type == mode or  # Direct match - this is the main one we care about
                            (response_type in mode_mapping and mode_mapping[response_type] == mode) or  # Mapped match for compatibility
                            (mode in mode_mapping and mode_mapping[mode] == response_type)  # Reverse mapped match
                        )
                        
                        if not mode_match:
                            self.logger.info(f"Filtering out message with response_type={response_type}, no match for mode={mode}")
                            continue
                    
                    # For user messages, only include ones that have a matching response
                    # or are at the end of the conversation (waiting for a response)
                    elif msg.get("role") == "user":
                        msg_id = msg.get("id", "")
                        if msg_id not in user_messages_to_include:
                            # Special case: include the very last user message even without a response
                            if msg == messages[-1]:
                                pass  # Include it
                            else:
                                continue  # Skip it
                
                # Format the message
                formatted_msg = {
                    "id": msg.get("id", ""),
                    "role": msg.get("role", "unknown"),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", "")
                }
                
                # Include response type if it's an assistant message
                if msg.get("role") == "assistant":
                    # Always include the response_type if present
                    if "response_type" in msg:
                        formatted_msg["response_type"] = msg["response_type"]
                        
                    # Log the response type for debugging
                    self.logger.debug(f"Message response_type={msg.get('response_type', 'None')}")
                
                # Include sources if requested and available
                if include_sources and "sources" in msg and msg["sources"]:
                    formatted_msg["sources"] = msg["sources"]
                
                formatted_messages.append(formatted_msg)
            
            return formatted_messages
            
        except Exception as e:
            self.logger.error(f"Error getting messages for API: {e}")
            return []
            
    def check_session_response_types(self) -> None:
        """
        Utility method to examine all sessions and log response type data.
        Useful for debugging issues with message filtering and response type preservation.
        Should be called at application startup.
        """
        self.logger.info("Running response type integrity check on all sessions...")
        try:
            # Get all sessions
            session_ids = session_manager.list_sessions(limit=None)
            session_count = len(session_ids)
            self.logger.info(f"Found {session_count} sessions to check")
            
            # Keep count of issues
            assistant_messages_total = 0
            messages_with_response_type = 0
            messages_missing_response_type = 0
            
            # Check each session
            for i, session_meta in enumerate(session_ids):
                session_id = session_meta.get('session_id')
                try:
                    session = session_manager.get_session(session_id)
                    self.logger.info(f"Checking session {session_id} ({i+1}/{session_count})")
                    
                    # Check each message for proper response type
                    for msg in session.messages:
                        if msg.get("role") == "assistant":
                            assistant_messages_total += 1
                            
                            # Check for response_type attribute
                            response_type = msg.get("response_type")

                            if response_type:
                                messages_with_response_type += 1
                            else:
                                messages_missing_response_type += 1
                                self.logger.warning(f"Assistant message missing response_type in session {session_id}")
                                
                except Exception as e:
                    self.logger.error(f"Error checking session {session_id}: {e}")
                    
            # Log results
            self.logger.info(f"Session response type integrity check complete:")
            self.logger.info(f"Total assistant messages: {assistant_messages_total}")
            self.logger.info(f"Messages with response_type: {messages_with_response_type} ({(messages_with_response_type/assistant_messages_total)*100:.1f}% if total > 0)")
            self.logger.info(f"Messages missing response_type: {messages_missing_response_type} ({(messages_missing_response_type/assistant_messages_total)*100:.1f}% if total > 0)")
            
        except Exception as e:
            self.logger.error(f"Error during session response type integrity check: {e}")

# Create a singleton instance
session_adapter = SessionAdapter()

# Run a check on session response types during initialization
# Commented out to avoid running in tests - can be enabled for debugging
# session_adapter.check_session_response_types()