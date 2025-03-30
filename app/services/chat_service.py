"""
Service layer for handling agent queries and responses with enhanced RAG query rewriting.
"""
import sys
import time
from typing import List, Any

from app.agents.chat_agents import AgentManager
from app.agents.response_strategies import ResponseStrategy
from app.config.chat_config import chat_config
from app.models.chat_models import ChatHistory, ResponseContent, ResponseMessage, Message
from app.services.cache_service import chat_cache
from app.services.enhancement_service import enhancement_service
from app.utils.logging_utils import get_logger


class ChatService:
    """Service for managing chat interactions with agents."""

    def __init__(self):
        # Create a dictionary to store chat histories for different sessions
        self.chat_histories = {}
        self.logger = get_logger(f"{__name__}.ChatService", "DEBUG")
        self.logger.debug("ChatService initialized")
        # Explicit print to check if output is working at all
        print("ChatService initialized", file=sys.stderr)

        self.config = chat_config
        self.agent_manager = AgentManager()

        # Use the enhancement service for query enhancement and QA
        self.enhancement_service = enhancement_service

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

        # TODO for now just work on only one model; Future support running against more than one LLMs
        first_key = next(iter(self.config.chat_model_configs))
        chat_model_config = self.config.chat_model_configs[first_key]

        # Log the incoming request
        self.logger.info(f"Chat request: session={session_id}, input_length={len(user_input)}")

        # Get or create chat history for the session
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatHistory()
        chat_history = self.chat_histories[session_id]

        # Check for special testing flags in the query
        force_refresh = user_input.strip().lower().startswith("test_routing:")

        # Extract the actual query if in test mode
        actual_query = user_input
        if force_refresh:
            self.logger.info(f"TEST MODE: Force refresh detected, bypassing cache")
            # Extract the actual query after the test_routing: prefix
            actual_query = user_input.split("test_routing:", 1)[1].strip()
            self.logger.info(f"TEST MODE: Using test query: {actual_query}")

        # Generate query hash for cache lookup
        query_hash = chat_cache.generate_query_hash(
                                        query=user_input,
                                        history=chat_history.get_messages(),
                                        session_id=session_id,
                                        system_prompt=custom_system_prompt,
                                        prompt_style=prompt_style,
                                        mode=mode  # Include mode in hash to differentiate between different mode requests
                                    )

        # Check cache if not in force refresh mode
        if not force_refresh:
            # Check cache for existing response
            cached_response, cache_hit = chat_cache.get_cached_response(query_hash)
            
            if cache_hit:
                self.logger.info(f"Cache hit for query_hash={query_hash}")
                
                # Extract cached data
                rag_output = cached_response["rag_response"]
                no_rag_output = cached_response["no_rag_response"]
                sources = cached_response.get("sources", [])
                
                # Add the user message and the cached response to chat history
                chat_history.add_user_message(user_input)
                
                # Determine which response to add to history based on mode
                primary_output = no_rag_output if mode == "no_rag" else rag_output
                chat_history.add_ai_message(primary_output)
                
                # Format message history
                formatted_history = self._format_history(chat_history.get_messages())
                
                # Create response content with cached data
                response_content = ResponseContent(
                                        input=user_input,
                                        history=formatted_history,
                                        output=rag_output if mode != "no_rag" else no_rag_output,
                                        no_rag_output=no_rag_output if mode != "rag" else None,
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
                
                return ResponseMessage(
                    response=response_content,
                    sources=sources
                )

        # Cache miss or force refresh - generate a new response
        self.logger.info(f"Cache miss or force refresh for query_hash={query_hash}, generating new response")
        
        # Add user message to history
        chat_history.add_user_message(user_input)
        
        # Get a temporary chat service instance to pass to the strategy
        chat_service = ChatService()
        
        # Determine which mode/strategy to use
        mode = "both" if mode == "both" else ("rag" if mode == "rag" else "no_rag")
        
        # Get the appropriate strategy
        strategy = ResponseStrategy.get_strategy(actual_query, mode, chat_service, chat_service.agent_manager)
        
        # Execute the strategy
        rag_response, no_rag_response, queries_tried = await strategy.execute(
            chat_model_config,
            actual_query, 
            chat_history,
            custom_system_prompt=None,  # Pass None as the default system prompt
            prompt_style=prompt_style
        )
        
        # Create a temporary strategy just to format sources
        temp_strategy = ResponseStrategy(self, self.agent_manager)
        # Extract sources from the RAG response if available
        sources = temp_strategy.format_sources(rag_response)
        
        # Determine which response to add to the chat history based on mode
        self.logger.debug(f"Selecting primary response - mode: {mode}, rag_response: {type(rag_response)}, no_rag_response: {type(no_rag_response)}")
        
        if mode == "no_rag" and no_rag_response is not None:
            self.logger.debug("Using no_rag_response as primary response (mode=no_rag)")
            primary_response = no_rag_response
        elif rag_response is not None:
            self.logger.debug("Using rag_response as primary response")
            primary_response = rag_response
        elif no_rag_response is not None:
            self.logger.debug("Using no_rag_response as fallback primary response")
            primary_response = no_rag_response
        else:
            self.logger.error("No valid response generated from any agent")
            primary_response = {"output": "I'm sorry, I couldn't generate a response at this time."}
        
        # Add the primary response to chat history
        if isinstance(primary_response, dict):
            chat_history.add_ai_message(primary_response['output'])
        else:
            chat_history.add_ai_message(primary_response)
        
        # Format message history for response
        formatted_history = self._format_history(chat_history.get_messages())
        
        # Extract the actual content from responses
        self.logger.debug(f"Extracting response content - rag_response: {type(rag_response)}, no_rag_response: {type(no_rag_response)}")
        
        if rag_response is None:
            rag_output = None
            self.logger.debug("rag_response is None")
        elif isinstance(rag_response, dict):
            rag_output = rag_response.get('output', "No output available in RAG response")
            self.logger.debug(f"rag_response is dict, extracted output: {rag_output[:30]}...")
        else:
            rag_output = str(rag_response)
            self.logger.debug(f"rag_response is {type(rag_response)}, converted to string: {rag_output[:30]}...")
            
        if no_rag_response is None:
            no_rag_output = None
            self.logger.debug("no_rag_response is None")
        elif isinstance(no_rag_response, dict):
            no_rag_output = no_rag_response.get('output', "No output available in non-RAG response")
            self.logger.debug(f"no_rag_response is dict, extracted output: {no_rag_output[:30]}...")
        else:
            no_rag_output = str(no_rag_response)
            self.logger.debug(f"no_rag_response is {type(no_rag_response)}, converted to string: {no_rag_output[:30]}...")
        
        # Determine intermediate steps
        intermediate_steps = []
        if mode != "no_rag" and isinstance(rag_response, dict):
            intermediate_steps = rag_response.get('intermediate_steps', [])
        
        # Create the response content
        primary_output = rag_output if mode != "no_rag" else no_rag_output
        secondary_output = no_rag_output if mode != "rag" else None
        
        self.logger.debug(f"Creating final response - Mode: {mode}")
        self.logger.debug(f"primary_output type: {type(primary_output)}, content: {primary_output[:50] if primary_output else 'None'}")
        self.logger.debug(f"secondary_output type: {type(secondary_output)}, content: {secondary_output[:50] if secondary_output else 'None'}")
        
        response_content = ResponseContent(
                                    input=user_input,
                                    history=formatted_history,
                                    output=primary_output,
                                    no_rag_output=secondary_output,
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
                            rag_response=rag_output,
                            no_rag_response=no_rag_output,
                            sources=sources,
                            system_prompt=custom_system_prompt,
                            prompt_style=prompt_style,
                            mode=mode
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
        
        self.logger.info(f"Response generated and cached in {response_time:.2f}s")
        
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
        self.logger.debug(f"delete_chat called with {session_id}")
        if session_id == "ALL_CHATS":
            self.chat_histories = {}
            return True
        elif session_id in self.chat_histories:
            del self.chat_histories[session_id]
            return True
        return False

    def get_chat(self, session_id: str) -> dict[Any, Any] | None:
        """
        Get chat history for a session.

        Args:
            session_id: The session ID to retrieve, or "ALL_CHATS" to get all sessions

        Returns:
            Dictionary containing chat history or None if not found
        """
        self.logger.debug(f"get_chat called with {session_id}")
        if session_id == "ALL_CHATS":
            return self.chat_histories
        if session_id in self.chat_histories:
            return {session_id: self.chat_histories[session_id]}
        return None

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