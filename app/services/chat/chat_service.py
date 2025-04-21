"""
Service layer for handling agent queries and responses with enhanced RAG query rewriting.
"""
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from app.agents.chat_agents import AgentManager
from app.agents.response_strategies import ResponseStrategy
from app.config.chat_config import chat_config
from app.models.chat_models import ChatHistory, ResponseContent, ResponseMessage, Message
from app.services.chat.chat_cache_service import chat_cache
from app.services.chat.session_adapter import session_adapter
from app.services.chat.session_service import session_manager
from app.services.common.enhancement_service import enhancement_service
from app.utils.logging_utils import get_logger

class ChatService:
    """Service for managing chat interactions with agents."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("ChatService initialized")
        
        # Using singletons
        self.config = chat_config
        self.enhancement_service = enhancement_service
        self.session_adapter = session_adapter
        self.session_manager = session_manager

        self.agent_manager = AgentManager()

    async def chat(self, data: Message) -> ResponseMessage:
        """
        Process chat message and return response with both RAG and non-RAG outputs.
        Uses cache to avoid redundant API calls for identical queries.

        Args:
            data: Message object containing
                user query,
                session ID,
                mode,
                system prompt [Optional] (default: System prompt),
                prompt style [Optional] (default: "default")

        Returns:
            ResponseMessage with RAG and non-RAG responses and sources
        """
        start_time = time.time()
        user_input = data.message
        session_id = data.session_id
        mode = data.mode
        custom_system_prompt = data.system_prompt
        prompt_style = data.prompt_style or "default"

        # XXX for now just work on only one model; Future support running against more than one LLMs
        first_key = next(iter(self.config.chat_model_configs))
        chat_model_config = self.config.chat_model_configs[first_key]

        # Log the incoming request
        self.logger.info(f"Chat request: session={session_id}, input_length={len(user_input)}")

        # Get chat history from session manager via adapter
        chat_history = self.session_adapter.get_chat_history(session_id)

        # Extract directives including test mode and client name
        query_directives = self.enhancement_service.extract_query_directives(user_input)
        force_refresh = query_directives["force_refresh"]
        client_name = query_directives["client_name"]
        actual_query = query_directives["actual_query"]

        if force_refresh:
            self.logger.info(f"TEST MODE: Force refresh detected, bypassing cache")
            self.logger.info(f"TEST MODE: Using test query: {actual_query}")
            
        if client_name:
            self.logger.info(f"CLIENT SPECIFIED: Using client '{client_name}' for query")

        # Generate query hash for cache lookup - include client name if specified
        # Not using the context for now
        query_hash = chat_cache.generate_query_hash(
                                        query=user_input,
                                        history=None,
                                        session_id=session_id,
                                        system_prompt=custom_system_prompt,
                                        prompt_style=prompt_style,
                                        mode=mode,
                                        client_name=client_name
                                    )

        # Check cache if not in force refresh mode
        if not force_refresh:
            # Check cache for existing response
            cached_response, cache_hit = chat_cache.get_cached_response(query_hash)
            
            if cache_hit:
                self.logger.info(f"Cache hit for query_hash={query_hash}")
                
                # Add the user message to chat history
                chat_history.add_user_message(user_input)

                response_type = mode
                # Extract cached data based on mode
                additional_metadata = {
                    "originalMode": mode,
                    "type": mode,
                    "actualResponseType": mode,
                    "cachedResponse": True
                }
                output = cached_response.get("rag_response")
                sources = cached_response.get("sources", [])
                additional_metadata["isNeedlResponse"] = True if mode == "needl" else False

                # Add enhanced debug logging
                self.logger.info(f"Adding cached AI message with response_type={response_type}, originalMode={mode}")
                chat_history.add_ai_message(
                    message=output,
                    response_type=response_type,
                    additional_metadata=additional_metadata
                )
                
                # Format message history
                formatted_history = self._format_history(chat_history.get_messages())
                
                # Standard modes (rag, no_rag, both)
                response_content = ResponseContent(
                                            input=user_input,
                                            history=formatted_history,
                                            output=output,
                                            no_rag_output=None,
                                            intermediate_steps=[]  # No intermediate steps for cached responses
                    )
                
                # Log cache hit stats
                response_time = time.time() - start_time
                chat_cache.log_cache_access(
                                    session_id=session_id,
                                    user_input=user_input,
                                    query_hash=query_hash,
                                    cache_hit=True,
                                    response_time=response_time
                                )
                
                # Save the updated chat history to the session even for cached responses
                self.logger.debug(f"Saving chat history with cached response to session {session_id}")
                self.session_adapter.save_from_chat_history(
                    session_id=session_id,
                    chat_history=chat_history,
                    mode=mode,
                    system_prompt=custom_system_prompt,
                    prompt_style=prompt_style,
                    additional_metadata={
                        "last_query": user_input,
                        "last_response_time": response_time,
                        "cache_hit": True,
                        "client_info": {"name": client_name} if client_name else {}
                    }
                )
                
                return ResponseMessage(
                    response=response_content,
                    sources=sources
                )

        # Cache miss or force refresh - generate a new response
        self.logger.info(f"Cache miss or force refresh for query_hash={query_hash}, generating new response")
        
        # Special handling for document service to avoid duplication
        # Only add the user message to history if this isn't a document service request
        # or if the history is empty (we want at least one message)
        is_document_service = hasattr(data, 'metadata') and data.metadata and 'document_question' in data.metadata
        
        if not is_document_service or not chat_history.get_messages():
            # Add user message to history only if not from document service or empty history
            self.logger.debug(f"Adding user message to history (length: {len(chat_history.get_messages())})")
            chat_history.add_user_message(user_input)
        else:
            # For document service requests, we don't add the message to history
            # this prevents duplication since the full message will be sent as input
            self.logger.debug("Document service request - not adding user message to history to prevent duplication")
            
        # Get a temporary chat service instance to pass to the strategy
        chat_service = ChatService()
        
        # Check if this is a document service request
        document_question = None
        if hasattr(data, 'metadata') and data.metadata and 'document_question' in data.metadata:
            document_question = data.metadata['document_question']
            self.logger.info(f"Document question found in metadata: {document_question}")
        
        # We support only three modes. However, the no_rag can be called by other names (Standard, ...)
        # always call such modes as no_rag
        if mode not in ["rag", "needl"]:
            mode = "no_rag"

        # Get the appropriate strategy
        strategy = ResponseStrategy.get_strategy(actual_query, mode, chat_service, chat_service.agent_manager)
        
        # Execute the strategy
        # If we have a document question, pass it separately for expected answer lookup
        if document_question:
            self.logger.info(f"Passing document question to strategy: {document_question}")
            
            # Check if this document question has an expected answer
            expected_answer_match = self.enhancement_service.get_answer(document_question)
            if expected_answer_match:
                self.logger.info(f"Found expected answer for document question")
                expected_answer = expected_answer_match.get("answer")
                
                # If we have a custom system prompt already, enhance it with the expected answer
                if custom_system_prompt:
                    custom_system_prompt = self.enhancement_service.enhance_prompt_with_expected_answer(
                        base_prompt=custom_system_prompt,
                        expected_answer=expected_answer
                    )
                    self.logger.info("Enhanced custom system prompt with expected answer")
        
        # Execute the strategy - pass client_name and processed query
        response, queries_tried = await strategy.execute(
            chat_model_config,
            actual_query, 
            chat_history,
            custom_system_prompt=custom_system_prompt,  # Pass through the custom_system_prompt value
            prompt_style=prompt_style,
            client_name=client_name  # Pass client name to strategy for namespace-specific retrieval
        )

        self.logger.debug(f"Creating final response - Mode: {mode}")
        self.logger.debug(f"Selecting primary response - mode: {mode}, "
                          f"response: {type(response)}, "
                          f"queries_tried: {type(queries_tried)}")

        if response is not None:
            if isinstance(response, dict):
                    if mode == "needl":
                        output = response["response"].get("output", "")
                    else:
                        output = response.get('output', "No output available in response")
                    self.logger.debug(f"response is dict, extracted output: {output[:30]}...")
            else:
                if mode == "needl":
                    output = ""
                    response = {"output": "Could not parse Needl response correctly"}
                else:
                    output = str(response)
                self.logger.debug(f"response is {type(response)}, converted to string: {output[:30]}...")
        else:
            output = ""
            self.logger.debug(f"response is None")

        # Create a temporary strategy just to format sources
        temp_strategy = ResponseStrategy(self, self.agent_manager)
        # Extract sources from the RAG response if available
        sources = temp_strategy.format_sources(response)
        
        if mode == "needl":
            response_type = "needl"
            additional_metadata = {"isNeedlResponse": True, "originalMode": "needl"}
            self.logger.info(f"Setting response_type=needl")
        elif mode == "no_rag":
            response_type = "no_rag"
            additional_metadata = {"originalMode": mode, "type": "standard"}
            self.logger.info(f"Setting response_type=no_rag")
        else:
            assert mode == "rag"
            response_type = "rag"
            additional_metadata = {"originalMode": mode, "type": "rag"}
            self.logger.info(f"Setting response_type=rag")
            
        # Add the message with proper metadata
        # Ensure we're setting the correct response_type that will be preserved
        # Add extra data to track both the original mode and the actual response type used
        enhanced_metadata = additional_metadata.copy() if additional_metadata else {}
        enhanced_metadata["actualResponseType"] = response_type
        enhanced_metadata["originalMode"] = mode
        
        self.logger.info(f"Adding AI message with response_type={response_type}, originalMode={mode}")
        
        chat_history.add_ai_message(
            message=output,
            response_type=response_type,
            additional_metadata=enhanced_metadata
        )
        
        # Format message history for response
        formatted_history = self._format_history(chat_history.get_messages())

        # Intermediate steps only valid when using internal RAG
        intermediate_steps = response.get('intermediate_steps') \
                            if mode == "rag" and isinstance(response, dict) else []

        response_content = ResponseContent(
                                    input=user_input,
                                    history=formatted_history,
                                    output=output,
                                    no_rag_output=None,
                                    intermediate_steps=intermediate_steps
                                )
        
        # Add the queries tried to the intermediate steps for transparency/debugging
        if ('intermediate_steps' in response_content.dict() and
                isinstance(response_content.intermediate_steps, list)):
            # Add feature flags status and query information
            # Initialize expected_answer variable if it might not be defined
            expected_answer = locals().get('expected_answer', None)
            
            feature_info = {
                "queries_tried": queries_tried,
                "query_count": len(queries_tried),
                "feature_flags": {
                    "semantic_filtering_enabled":
                        self.config.CHAT_FEATURE_FLAGS.get("semantic_similarity_filtering",
                                                           False),
                    "expected_answer_enabled": self.config.CHAT_FEATURE_FLAGS.get("expected_answer_enrichment", False),
                    "expected_answer_used": expected_answer is not None and self.config.CHAT_FEATURE_FLAGS.get(
                        "expected_answer_enrichment", False)
                }
            }
            response_content.intermediate_steps.append(feature_info)
        
        # Cache the generated response
        chat_cache.cache_response(
                                query_hash=query_hash,
                                user_input=user_input,
                                rag_response=output,
                                sources=sources,
                                system_prompt=custom_system_prompt,
                                prompt_style=prompt_style,
                                mode=mode,
                                client_name=client_name
            )
        
        # Log cache miss stats
        response_time = time.time() - start_time
        chat_cache.log_cache_access(
                                session_id=session_id,
                                user_input=user_input,
                                query_hash=query_hash,
                                cache_hit=False,
                                response_time=response_time
                            )
        
        # Save the updated chat history to the session
        self.logger.debug(f"Saving updated chat history to session {session_id}")
        
        # Combine all additional metadata
        additional_metadata = {
            "last_query": user_input,
            "last_response_time": response_time,
            "client_info": {"name": client_name} if client_name else {}
        }
        
        # Save the session with all the metadata
        success = self.session_adapter.save_from_chat_history(
            session_id=session_id,
            chat_history=chat_history,
            mode=mode,
            system_prompt=custom_system_prompt,
            prompt_style=prompt_style,
            additional_metadata=additional_metadata
        )
        
        if success:
            self.logger.info(f"Response generated and chat history saved in {response_time:.2f}s")
        else:
            self.logger.error(f"Failed to save chat history for session {session_id}")
        
        return ResponseMessage(
            response=response_content,
            sources=sources
        )

    def delete_chat(self, session_id: str) -> bool:
        """
        Delete chat history for a session.

        Args:
            session_id: The session ID to delete, or "ALL_CHATS" to clear all sessions

        Returns:
            True if deletion was successful, False otherwise
        """
        self.logger.info(f"delete_chat called with {session_id}")
        if session_id == "ALL_CHATS":
            # Delete all persistent sessions
            deleted_count = self.session_manager.delete_all_sessions()
            self.logger.info(f"Deleted {deleted_count} persistent sessions")
            return True
        else:
            # Delete from persistent storage
            success = self.session_manager.delete_session(session_id)
            return success

    def get_chat(self, session_id: str) -> Dict[str, Any]:
        """
        Get chat history for a session.

        Args:
            session_id: The session ID to retrieve, or "ALL_CHATS" to get all sessions

        Returns:
            Dictionary containing chat history or empty dict if not found
        """
        self.logger.info(f"get_chat called with {session_id}")
        
        if session_id == "ALL_CHATS":
            # Get paginated list of available sessions with metadata
            sessions = self.session_manager.list_sessions(limit=100)
            return {"sessions": sessions, "total_count": len(sessions)}
        
        try:
            # Get the session through adapter
            messages = self.session_adapter.get_messages_for_api(
                session_id=session_id,
                include_sources=True
            )
            
            # Get session metadata
            session = self.session_manager.get_session(session_id)
            
            return {
                "session_id": session_id,
                "messages": messages,
                "metadata": session.metadata
            }
        except Exception as e:
            self.logger.error(f"Error retrieving session {session_id}: {e}")
            return {"error": f"Session not found or error retrieving session: {str(e)}"}
            
    def list_sessions(self, limit: Optional[int] = 100, offset: int = 0) -> Dict[str, Any]:
        """
        List available sessions with pagination.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Starting offset for pagination
            
        Returns:
            Dictionary with sessions list and metadata
        """
        sessions = self.session_manager.list_sessions(limit=limit, offset=offset)
        
        return {
            "sessions": sessions,
            "total_count": len(sessions) + offset,  # Approximate if paginated
            "page": offset // limit + 1 if limit > 0 else 1,
            "page_size": limit
        }
        
    def get_session_by_id(self, session_id: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific session.
        
        Args:
            session_id: Session ID to retrieve
            mode: Optional mode to filter messages by (rag, no_rag, needl, compare)
            
        Returns:
            Dictionary with session data
        """
        try:
            session = self.session_manager.get_session(session_id)
            
            return {
                "session_id": session_id,
                "metadata": session.metadata,
                "messages": self.session_adapter.get_messages_for_api(
                    session_id=session_id,
                    include_sources=True,
                    mode=mode
                )
            }
        except Exception as e:
            self.logger.error(f"Error retrieving session {session_id}: {e}")
            return {"error": f"Error retrieving session: {str(e)}"}
            
    def find_sessions_by_tag(self, tag: str) -> Dict[str, Any]:
        """
        Find sessions with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            Dictionary with matching session IDs
        """
        session_ids = self.session_manager.find_sessions_by_tag(tag)
        
        return {
            "tag": tag,
            "session_ids": session_ids,
            "count": len(session_ids)
        }
        
    def find_sessions_by_mode(self, mode: str) -> Dict[str, Any]:
        """
        Find sessions containing prompts with a specific mode.
        
        Args:
            mode: Mode to search for (rag, standard, needl, compare)
            
        Returns:
            Dictionary with matching session IDs
        """
        session_ids = self.session_manager.find_sessions_by_mode(mode)
        
        return {
            "mode": mode,
            "session_ids": session_ids,
            "count": len(session_ids)
        }
        
    def list_sessions(self, limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
        """
        List available sessions with pagination.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Starting offset for pagination
            
        Returns:
            Dictionary with sessions and total count
        """
        try:
            # Get sessions from session manager
            sessions = self.session_manager.list_sessions(limit=limit, offset=offset)
            
            # Count total sessions with a more efficient approach for SQLite
            # Get a separate count without limits to avoid loading all sessions
            if self.session_manager.storage_type == "sqlite":
                try:
                    db_path = self.session_manager.storage_path / "sessions.db"
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM sessions")
                    total_count = cursor.fetchone()[0]
                    conn.close()
                except Exception as e:
                    self.logger.error(f"Error counting sessions: {e}")
                    # Fallback to estimating from returned results
                    total_count = len(sessions) + offset
            else:
                # For other storage types, get all sessions (could be inefficient for large sets)
                total_count = len(self.session_manager.list_sessions())
            
            return {
                "sessions": sessions,
                "total_count": total_count
            }
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            return {
                "sessions": [],
                "total_count": 0,
                "error": str(e)
            }

    # Removed _format_sources method - now implemented in ResponseStrategy class

    def _format_history(self, messages: List) -> List:
        """
        Format message history for response.

        Args:
            messages: List of message objects

        Returns:
            List of formatted message dictionaries
        """
        return [
            msg.dict() if hasattr(msg, 'dict') else {
                "role": getattr(msg, "type", "unknown"),
                "content": getattr(msg, "content", str(msg))
            }
            for msg in messages
        ]


class AgentService:
    """
    Static service for direct agent interactions without chat history management.
    Use ChatService for most applications that require session management.
    """

    @staticmethod
    async def process_query(
            query: str,
            history: List = None,
            use_rag: bool = True,
            use_dual_response: bool = False
    ) -> ResponseMessage:
        """
        Process a user query through the appropriate agent(s).

        Args:
            query: The user's query text
            history: Optional chat history
            use_rag: Whether to use the RAG-enabled agent
            use_dual_response: Whether to generate both RAG and non-RAG responses

        Returns:
            A ResponseMessage object containing the response(s) and sources
        """
        # TODO for now just work on only one model; Future support running against more than one LLMs
        first_key = next(iter(chat_config.config.chat_model_configs))
        chat_model_config = chat_config.config.chat_model_configs[first_key]

        # Create a minimal chat history
        from app.models.chat_models import ChatHistory
        chat_history = ChatHistory()
        if history:
            for msg in history:
                if msg.get('role') == 'user':
                    chat_history.add_user_message(msg.get('content', ''))
                elif msg.get('role') == 'assistant':
                    chat_history.add_ai_message(msg.get('content', ''))
        
        # Get a temporary chat service instance to pass to the strategy
        chat_service = ChatService()
        
        # Determine which mode/strategy to use
        mode = "both" if use_dual_response else ("rag" if use_rag else "no_rag")
        
        # Get the appropriate strategy
        strategy = ResponseStrategy.get_strategy(query, mode, chat_service, chat_service.agent_manager)
        
        # Execute the strategy
        rag_response, no_rag_response, queries_tried = await strategy.execute(
            chat_model_config,
            query, 
            chat_history,
            custom_system_prompt=None,  # Pass None as the default system prompt
            prompt_style="default"
        )
        
        # Extract sources from the RAG response if available
        sources = strategy.format_sources(rag_response)
        
        # Get the appropriate outputs
        rag_output = rag_response.get('output', '') if rag_response else None
        non_rag_output = no_rag_response.get('output', '') if no_rag_response else None
        
        # Determine primary output based on the mode
        primary_output = rag_output if use_rag else non_rag_output
        secondary_output = non_rag_output if use_rag and use_dual_response else None
        
        # Get intermediate steps if available
        intermediate_steps = rag_response.get('intermediate_steps', []) if rag_response else []
        
        # Create response content
        response_content = ResponseContent(
            input=query,
            history=history or [],
            output=primary_output,
            no_rag_output=secondary_output,
            intermediate_steps=intermediate_steps
        )

        return ResponseMessage(
            response=response_content,
            sources=sources
        )

    # Removed _extract_sources method - now using ChatService._format_sources instead