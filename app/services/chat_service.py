"""
Service layer for handling agent queries and responses.
"""
from typing import Dict, List, Optional, Any
import logging
import os
import sys
import time
import json
import re
from app.agents.chat_agents import agent_manager
from app.models.chat_models import ChatHistory, ResponseContent, ResponseMessage, Source, Message
from app.utils.logging_utils import get_logger, diagnose_logger, ensure_debug_logging
from app.services.cache_service import chat_cache

# Determine the environment
environment = os.getenv("ENVIRONMENT", "development").lower()  # Default to "development"
# Set the logging level based on the environment
log_level = logging.INFO if environment == "production" else logging.DEBUG
print(f"Setting log level to {log_level} for {__name__}", file=sys.stderr)

class ChatService:
    """Service for managing chat interactions with agents."""

    def __init__(self):
        # Create a dictionary to store chat histories for different sessions
        self.chat_histories = {}
        self.logger = get_logger(f"{__name__}.ChatService", "DEBUG")
        self.logger.debug("ChatService initialized")
        # Explicit print to check if output is working at all
        print("ChatService initialized", file=sys.stderr)
        
    def _is_database_query(self, query: str) -> bool:
        """
        Determine if a user query is likely a database/analytics query.
        
        Args:
            query: The user's input query
            
        Returns:
            bool: True if the query appears to be a database query
        """
        # Exclusion patterns - don't route these to database agent
        exclusion_patterns = [
            r'\bincrementality test',   # Incrementality tests should go to RAG
            r'\bprospecting\b',         # Prospecting questions should go to RAG
            r'\bmarketing strateg',     # Marketing strategy questions to RAG
            r'\bmsquared\b',            # Company-specific questions to RAG
            r'\bcase stud',             # Case studies to RAG
            r'\bwhite paper',           # Documentation to RAG
            r'\bbest practice',         # Best practices to RAG
            r'\brecommend'              # Recommendation requests to RAG
        ]
        
        self.logger.debug(f"PATTERN MATCHING: Evaluating query against exclusion patterns")
        # First check exclusions - if any match, don't use database agent
        for pattern in exclusion_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.logger.debug(f"PATTERN MATCHING: EXCLUDED - Found matching exclusion pattern: '{pattern}'")
                return False
        
        self.logger.debug(f"PATTERN MATCHING: No exclusion patterns matched, continuing...")
        
        # Keywords that suggest a database or analytics query
        db_keywords = [
            r'\b(database|data warehouse)\b',  # More specific data terms 
            r'\b(analytics dashboard)\b',
            r'\b(metrics|kpi)\b',
            r'\b(revenue|sales figures)\b',
            r'\b(report numbers|chart data)\b',
            r'\b(trend analysis)\b',
            r'\b(statistics|statistical)\b',
            r'\b(measure results)\b',
            r'\b(top performers)\b',
            r'\b(percentage breakdown)\b'
        ]
        
        # Patterns that suggest data requests
        query_patterns = [
            r'how many (customers|users|sales)',
            r'show me (the|our) (data|numbers|figures)',
            r'what (is|are) the (metrics|kpis|numbers)',
            r'calculate (the|our)',
            r'average (value|rate|number)',
            r'tell me about our numbers',
            r'list the (top|bottom)'
        ]
        
        self.logger.debug(f"PATTERN MATCHING: Checking for database keyword matches")
        # Check for keyword matches
        for keyword in db_keywords:
            if re.search(keyword, query, re.IGNORECASE):
                self.logger.debug(f"PATTERN MATCHING: MATCH - Database keyword pattern matched: '{keyword}'")
                return True
                
        self.logger.debug(f"PATTERN MATCHING: Checking for database query pattern matches")
        # Check for query pattern matches
        for pattern in query_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.logger.debug(f"PATTERN MATCHING: MATCH - Database query pattern matched: '{pattern}'")
                return True
                
        self.logger.debug(f"PATTERN MATCHING: No database patterns matched, routing to RAG")
        return False

    async def chat(self, data: Message) -> ResponseMessage:
        """
        Process chat message and return response with both RAG and non-RAG outputs.
        Uses cache to avoid redundant API calls for identical queries.

        Args:
            data: Message object containing user input and session ID

        Returns:
            ResponseMessage with RAG and non-RAG responses and sources
        """
        start_time = time.time()
        user_input = data.message
        session_id = data.session_id
        mode = getattr(data, "mode", "rag")

        cache_hit = False
        
        # Log the incoming request
        self.logger.info(f"Chat request: session={session_id}, input_length={len(user_input)}")

        # Get or create chat history for the session
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatHistory()
        chat_history = self.chat_histories[session_id]
        
        # Generate query hash for cache lookup
        query_hash = chat_cache.generate_query_hash(
            query=user_input,
            history=chat_history.get_messages(),
            session_id=session_id
        )
        
        # Check for special testing flags in the query
        force_refresh = "test_routing:" in user_input.lower()
        
        # For testing, extract the real query if it contains the test flag
        test_query = user_input
        if force_refresh:
            self.logger.info(f"TEST MODE: Force refresh detected, bypassing cache")
            # Extract the actual query after the test_routing: prefix
            test_query = user_input.split("test_routing:", 1)[1].strip()
            self.logger.info(f"TEST MODE: Using test query: {test_query}")
        
        # Check if this is a database query before checking cache
        is_database_query = self._is_database_query(test_query if force_refresh else user_input)
        self.logger.debug(f"Query identified as database query: {is_database_query}")
        
        # For database queries or force refresh, skip cache
        if is_database_query or force_refresh:
            if is_database_query:
                self.logger.info(f"Database query detected, bypassing cache: {user_input}")
            cache_hit = False
            cached_rag_response, cached_no_rag_response = (None, None)
        else:
            # Check cache for existing response
            cached_rag_response, cached_no_rag_response = (None, None)
            cached_response, cache_hit = chat_cache.get_cached_response(query_hash)
            if cache_hit and ((mode != "no_rag" and cached_response["rag_response"] is None)
                        or (mode != "rag" and cached_response["no_rag_response"] is None)):
                    # Previous request was for a different mode
                cached_rag_response, cached_no_rag_response = (
                cached_response["rag_response"], cached_response["no_rag_response"])
                cache_hit = False
                
        # If we're in test mode, use the extracted test query
        actual_query = test_query if force_refresh else user_input

        if cache_hit:
            self.logger.info(f"Cache hit for query_hash={query_hash}")
            
            # Extract cached data
            rag_output = cached_response["rag_response"]
            no_rag_output = cached_response["no_rag_response"]
            sources = cached_response.get("sources", [])
            
            # Add the user message and the cached response to chat history
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(rag_output)

            # Format message history
            formatted_history = self._format_history(chat_history.get_messages())
            
            # Create response content with cached data
            response_content = ResponseContent(
                input=user_input,
                history=formatted_history,
                output=rag_output,
                no_rag_output=no_rag_output,
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
        
        # Cache miss - need to generate a new response
        self.logger.info(f"Cache miss for query_hash={query_hash}, generating new response")
        
        # Add user message to history
        chat_history.add_user_message(user_input)
        
        # Detect if this is a database query
        self.logger.debug(f"==== AGENT ROUTING: Analyzing query for agent selection ====")
        self.logger.debug(f"QUERY: {actual_query}")
        is_database_query = self._is_database_query(actual_query)
        self.logger.debug(f"AGENT ROUTING DECISION: Query classified as database query: {is_database_query}")
        
        if is_database_query:
            self.logger.debug(f"==== AGENT ROUTING: Selected DATABASE AGENT ====")
            self.logger.debug(f"AGENT ROUTING: Calling database agent with query: {actual_query[:100]}...")
            rag_response = await agent_manager.database_agent.ainvoke(
                {"input": actual_query, "history": chat_history.get_messages()},
                include_run_info=True
            )
            self.logger.debug(f"AGENT ROUTING: Database response received, length: {len(str(rag_response))}")
            # For database queries, we'll use same response for no_rag
            no_rag_response = None
        else:
            # Generate response using agent executor with RAG
            if mode != "no_rag":
                self.logger.debug(f"==== AGENT ROUTING: Selected RAG AGENT ====")
                self.logger.debug(f"AGENT ROUTING: Calling RAG agent with query: {actual_query[:100]}...")
                rag_response = await agent_manager.rag_agent.ainvoke(
                    {"input": actual_query, "history": chat_history.get_messages()},
                    include_run_info=True
                )
                self.logger.debug(f"AGENT ROUTING: RAG response received, length: {len(str(rag_response))}")
                self.logger.debug(f"AGENT ROUTING: RAG output: {rag_response['output'][:200]}...")
            else:
                self.logger.debug(f"AGENT ROUTING: Using cached RAG response")
                rag_response = cached_rag_response
            
            # Generate response using agent executor without RAG
            if mode != "rag" and not is_database_query:
                self.logger.debug(f"==== AGENT ROUTING: Also selected STANDARD AGENT ====")
                self.logger.debug(f"AGENT ROUTING: Calling standard agent with query: {actual_query[:100]}...")
                no_rag_response = await agent_manager.standard_agent.ainvoke(
                    {"input": actual_query, "history": chat_history.get_messages()},
                    include_run_info=True
                )
                self.logger.debug(f"AGENT ROUTING: Standard response received, length: {len(str(no_rag_response))}")
                self.logger.debug(f"AGENT ROUTING: Standard output: {no_rag_response['output'][:200]}...")
            else:
                self.logger.debug(f"AGENT ROUTING: Using cached non-RAG response")
                no_rag_response = cached_no_rag_response
        
        # Extract sources from the RAG response
        sources = self._format_sources(rag_response)
        
        # Save AI's message to chat history (using the RAG response as primary)
        chat_history.add_ai_message(rag_response['output'])
        
        # Format message history for response
        formatted_history = self._format_history(chat_history.get_messages())
        
        # Create the response content
        response_content = ResponseContent(
            input=user_input,
            history=formatted_history,
            output=rag_response['output'],
            no_rag_output=no_rag_response['output'] if no_rag_response is not None else None,
            intermediate_steps=rag_response.get('intermediate_steps', [])
        )

        # Cache the generated response
        chat_cache.cache_response(
            query_hash=query_hash,
            user_input=user_input,
            rag_response=rag_response['output'],
            no_rag_response=no_rag_response['output'] if no_rag_response is not None else None,
            sources=sources
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

    def get_chat(self, session_id: str) -> Dict:
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

    def _format_sources(self, rag_response: Dict) -> List[Source]:
        """
        Format sources from the RAG response.

        Args:
            rag_response: Response from the RAG agent

        Returns:
            List of formatted Source objects
        """
        raw_sources = rag_response.get("sources", [])
        formatted_sources = []

        for source in raw_sources:
            if isinstance(source, dict):
                formatted_sources.append(Source(
                    title=source.get("title", ""),
                    url=source.get("url", ""),
                    content=source.get("content", "")
                ))
            else:
                # Handle string or other non-dict sources
                formatted_sources.append(Source(content=str(source)))

        return formatted_sources

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
        history = history or []
        sources = []

        # Get RAG response if requested
        rag_output = None
        rag_steps = None
        if use_rag or use_dual_response:
            rag_result = await agent_manager.rag_agent.ainvoke({
                "input": query,
                "history": history
            })
            rag_output = rag_result.get("output", "")
            rag_steps = rag_result.get("intermediate_steps", [])

            # Extract sources from RAG response
            sources = AgentService._extract_sources(rag_steps)

        # Get non-RAG response if requested
        non_rag_output = None
        if not use_rag or use_dual_response:
            non_rag_result = await agent_manager.standard_agent.ainvoke({
                "input": query,
                "history": history
            })
            non_rag_output = non_rag_result.get("output", "")

        # Determine primary output
        primary_output = rag_output if use_rag else non_rag_output
        secondary_output = non_rag_output if use_rag and use_dual_response else None

        # Create response content
        response_content = ResponseContent(
            input=query,
            history=history,
            output=primary_output,
            no_rag_output=secondary_output,
            intermediate_steps=rag_steps if use_rag else []
        )

        return ResponseMessage(
            response=response_content,
            sources=sources
        )

    @staticmethod
    def _extract_sources(steps: List) -> List[Source]:
        """
        Extract source information from agent intermediate steps.

        Args:
            steps: List of intermediate steps from agent execution

        Returns:
            List of Source objects
        """
        sources = []

        for step in steps:
            if (len(step) > 1 and step[0].tool == "search_msquared_docs"
                    and isinstance(step[1], dict)):
                docs = step[1].get("documents", [])
                for doc in docs:
                    metadata = doc.metadata or {}
                    source = Source(
                        title=metadata.get("title", None),
                        url=metadata.get("url", None),
                        content=doc.page_content
                    )
                    sources.append(source)

        return sources