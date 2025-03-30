"""
Response strategy classes for different types of chat responses.
These strategies encapsulate the logic for generating different types of responses
(RAG, non-RAG, database, etc.) in a modular way.
"""
from typing import Dict, List, Tuple, Any, Optional, Union
import re

from app.agents.chat_agents import agent_manager
from app.config import prompt_config
from app.models.chat_models import ChatHistory, Source


class ResponseStrategy:
    """Base class for response generation strategies."""
    
    def __init__(self, chat_service):
        """
        Initialize the strategy with a reference to the chat service.
        
        Args:
            chat_service: The ChatService instance
        """
        self.chat_service = chat_service
        self.logger = chat_service.logger
    
    @staticmethod
    def get_strategy(query: str, mode: str, chat_service) -> 'ResponseStrategy':
        """
        Factory method to get the appropriate response strategy based on query and mode.
        
        Args:
            query: The user query
            mode: The response mode (rag, no_rag, both)
            chat_service: The ChatService instance
            
        Returns:
            A ResponseStrategy instance
        """
        # Create a temporary strategy to use its methods
        temp_strategy = ResponseStrategy(chat_service)
        
        # First check if this is a database query
        if temp_strategy._is_database_query(query):
            chat_service.logger.debug(f"Using DatabaseResponseStrategy for query: {query[:50]}...")
            return DatabaseResponseStrategy(chat_service)
        
        # If not a database query, select based on mode
        if mode == "no_rag":
            chat_service.logger.debug(f"Using NonRAGResponseStrategy for query: {query[:50]}...")
            return NonRAGResponseStrategy(chat_service)
        elif mode == "both":
            chat_service.logger.debug(f"Using DualResponseStrategy for query: {query[:50]}...")
            return DualResponseStrategy(chat_service)
        else:  # "rag" or default
            chat_service.logger.debug(f"Using RAGResponseStrategy for query: {query[:50]}...")
            return RAGResponseStrategy(chat_service)
    
    async def execute(self, query: str, chat_history: ChatHistory, 
                     custom_system_prompt: str = None,
                      prompt_style: str = "default")\
                -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Execute the strategy to generate a response.
        
        Args:
            query: The user query
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            Either rag_response or no_rag_response may be None depending on strategy
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    async def execute_rag_with_retry(self, query: str, history: List, 
                                max_attempts: int = 3,
                                custom_system_prompt: str = None,
                                rag_agent=None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute RAG with multiple query formulations and retry logic.
        
        Args:
            query: The original user query
            history: Chat history
            max_attempts: Maximum number of query attempts
            custom_system_prompt: Optional custom system prompt
            rag_agent: Optional RAG agent object to use for execution
            
        Returns:
            Tuple of (best_response, queries_tried)
        """
        # Implementation duplicates ChatService._execute_rag_with_retry to avoid access to protected method
        self.logger.debug(f"QUERY REWRITING: Starting execute_rag_with_retry with query: {query[:100]}")

        # Get the appropriate RAG agent
        if rag_agent is None:
            # Use the system prompt to get a RAG agent if no agent provided
            # Configure the agent with the original query to ensure proper filter selection
            rag_agent = agent_manager.get_rag_agent(
                custom_system_prompt=custom_system_prompt,
                query=query  # Always use original query for filtering
            )
            self.logger.debug(
                f"QUERY REWRITING: Using RAG agent with custom system prompt: {custom_system_prompt is not None}")
        else:
            # Use the provided agent (which may have expected answer)
            self.logger.debug(f"QUERY REWRITING: Using pre-configured RAG agent")

        # Create a processing function for the agent
        # Note: We use the agent configured with the original query's filter,
        # but send the rewritten queries to the LLM
        async def process_query(q):
            return await rag_agent.ainvoke(
                {"input": q, "history": history},
                include_run_info=True
            )

        # Use the enhancement service to try alternative queries
        response, queries_tried = await self.chat_service.enhancement_service.try_alternative_queries(
            original_query=query,
            process_function=process_query,
            history=history,
            max_attempts=max_attempts
        )

        return response, queries_tried
    
    async def invoke_agent_with_fallback(self, query: str,
                                        agent_name: str,
                                        chat_history: Any,
                                        prompt_style: str = "default") \
                                -> Tuple[Dict[str, Any], List[str]]:
        """
        Invoke an agent with fallback to default prompt style if needed.
        
        Args:
            query: The query to send to the agent
            agent_name: The name of the agent to use
            chat_history: The chat history
            prompt_style: The prompt style to use
            
        Returns:
            Tuple of (response, queries_tried)
        """
        # Implementation duplicates ChatService._invoke_agent_with_fallback to avoid access to protected method
        self.logger.debug(f"==== AGENT ROUTING: Selected {agent_name} AGENT with {prompt_style} style ====")
        self.logger.debug(f"AGENT ROUTING: Calling {agent_name} agent with query: {query[:100]}...")

        # Map internal agent names to prompt config names
        prompt_config_mapping = {
            "standard": "non_rag",
            "rag": "rag",
            "database": "database"
        }

        # Get the correct config section name
        config_name = prompt_config_mapping.get(agent_name, agent_name)

        try:
            # Get the appropriate prompt from the prompt cache using the mapped config name
            system_prompt = prompt_config.get_prompt(config_name, prompt_style)
            self.logger.info(f"Using '{prompt_style}' style prompt for {agent_name} agent")
        except ValueError as e:
            self.logger.warning(f"Error retrieving prompt style '{prompt_style}' for {agent_name}: {e}. Using default.")
            system_prompt = prompt_config.get_prompt(config_name, "default")

        # Get the agent with the selected prompt
        agent = agent_manager.get_agent(agent_name, system_prompt)

        response = await agent.ainvoke(
            {"input": query, "history": chat_history.get_messages()},
            include_run_info=True
        )
        self.logger.debug(f"AGENT ROUTING: {agent_name} response received, length: {len(str(response))}")

        return response, [query]
        
    def format_sources(self, rag_response: Union[Dict[str, Any], str]) -> List[Source]:
        """
        Format sources from the RAG response.

        Args:
            rag_response: Response from the RAG agent, can be a dict or a string

        Returns:
            List of formatted Source objects
        """
        if rag_response is None:
            self.logger.warning("Received None rag_response in format_sources")
            return []

        # Handle string responses (this happens when a custom system prompt is used)
        if isinstance(rag_response, str):
            self.logger.debug("Received string rag_response in format_sources, no sources available")
            return []

        # For dictionary responses, extract sources normally
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
            r'\bincrementality test',  # Incrementality tests should go to RAG
            r'\bprospecting\b',  # Prospecting questions should go to RAG
            r'\bmarketing strategy',  # Marketing strategy questions to RAG
            r'\bmarketing mix\b',  # Marketing mix model questions to RAG
            r'\bmmm\b',  # MMM questions to RAG
            r'\bmodel\b',  # Model-related questions to RAG
            r'\bvalidat',  # Validation questions to RAG
            r'\bmsquared\b',  # Company-specific questions to RAG
            r'\bcase stud',  # Case studies to RAG
            r'\bwhite paper',  # Documentation to RAG
            r'\bbest practice',  # Best practices to RAG
            r'\brecommend'  # Recommendation requests to RAG
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
                self.logger.debug(f"PATTERN MATCHING: MATCH - Database keyword "
                                  f"pattern matched: '{keyword}'")
                return True

        self.logger.debug(f"PATTERN MATCHING: Checking for database query pattern matches")
        # Check for query pattern matches
        for pattern in query_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.logger.debug(f"PATTERN MATCHING: MATCH - Database query "
                                  f"pattern matched: '{pattern}'")
                return True

        self.logger.debug(f"PATTERN MATCHING: No database patterns matched, routing to RAG")
        return False


class RAGResponseStrategy(ResponseStrategy):
    """Strategy for generating RAG responses."""
    
    async def execute(self, query: str, chat_history: ChatHistory, 
                     custom_system_prompt: str = None, prompt_style: str = "default") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Generate a RAG response.
        
        Args:
            query: The user query
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            The no_rag_response will be None for this strategy
        """
        self.logger.debug(f"RAGResponseStrategy.execute - Starting with query: {query[:50]}...")
        
        rag_response, queries_tried = await self._generate_rag_response(
            query, 
            chat_history,
            custom_system_prompt, 
            prompt_style
        )
        
        # Validate response before returning
        if rag_response is None:
            self.logger.warning("RAGResponseStrategy.execute - Received None rag_response")
            rag_response = {"output": f"Error: Failed to generate RAG response for '{query[:30]}...'"}
        elif not isinstance(rag_response, dict):
            self.logger.warning(f"RAGResponseStrategy.execute - Expected dict, got {type(rag_response)}")
            # Convert string or other type to dict format
            if isinstance(rag_response, str):
                rag_response = {"output": rag_response}
            else:
                rag_response = {"output": f"Unexpected response type: {type(rag_response)}"}
                
        self.logger.debug(f"RAGResponseStrategy.execute - Completed, returning response type: {type(rag_response)}")
        
        return rag_response, None, queries_tried
    
    async def _generate_rag_response(self, query: str, chat_history: ChatHistory, 
                                   custom_system_prompt: str = None, prompt_style: str = "default") -> Tuple[Dict[str, Any], List[str]]:
        """
        Generate a RAG response with retry logic.
        
        Args:
            query: The user query
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            
        Returns:
            Tuple of (rag_response, queries_tried)
        """
        self.logger.debug(f"==== AGENT ROUTING: Selected RAG AGENT with QUERY REWRITING ====")
        self.logger.debug(f"AGENT ROUTING: Calling RAG agent with potential query rewrites: {query[:100]}...")
        
        # Get expected answer for this query (if it exists) - Use enhancement_service directly
        expected_answer = self.chat_service.enhancement_service.get_answer(query)
        if expected_answer:
            expected_answer = expected_answer.get('answer')
        
        # Get the appropriate system prompt based on the requested style
        try:
            # If no custom prompt is provided, use the prompt from the prompt cache
            if not custom_system_prompt:
                system_prompt = prompt_config.get_prompt("rag", prompt_style)
                self.logger.info(f"Using '{prompt_style}' style prompt from prompt cache")
            else:
                system_prompt = custom_system_prompt
                self.logger.info(f"Using custom system prompt, ignoring prompt style: '{prompt_style}'")
        except ValueError as e:
            self.logger.warning(f"Error retrieving prompt style '{prompt_style}': {e}. Using default.")
            system_prompt = prompt_config.get_prompt("rag", "default")
        
        # Check if expected answer enrichment is enabled in feature flags
        use_expected_answer = self.chat_service.config.CHAT_FEATURE_FLAGS.get("expected_answer_enrichment", False)
        self.logger.debug(f"Expected answer enrichment enabled: {use_expected_answer}")
        
        # Skip query rewriting and expected answer handling when using custom system prompt
        if custom_system_prompt:
            self.logger.info(f"Using custom system prompt - skipping query rewriting and expected answer handling")
            # Use the original query directly with the custom system prompt
            rag_agent = agent_manager.get_rag_agent(
                custom_system_prompt=custom_system_prompt,
                query=query  # Pass query for metadata filtering
            )
            rag_response = await rag_agent.ainvoke(
                {"input": query, "history": chat_history.get_messages()},
                include_run_info=True
            )
            queries_tried = [query]  # Only tried the original query
        elif not use_expected_answer:
            if expected_answer:
                self.logger.info(f"Found expected answer but feature flag disabled: '{query[:50]}...'")
            else:
                self.logger.info(f"Did not find expected answer and feature flag disabled: '{query[:50]}...'")
            
            rag_response, queries_tried = await self.execute_rag_with_retry(
                query,
                chat_history.get_messages(),
                custom_system_prompt=system_prompt
            )
        elif expected_answer:
            self.logger.info(f"Found expected answer for query and feature flag enabled: '{query[:50]}...'")
            # Get RAG agent with both system prompt and expected answer
            rag_agent = agent_manager.get_rag_agent(
                custom_system_prompt=system_prompt,
                expected_answer=expected_answer
            )
            
            # Use the method that tries multiple query formulations
            rag_response, queries_tried = await self.execute_rag_with_retry(
                query,
                chat_history.get_messages(),
                rag_agent=rag_agent  # Pass the custom agent with expected answer
            )
        else:
            self.logger.info(f"Did not find expected answer for query and feature flag is not enabled: '{query[:50]}...'")
            rag_response, queries_tried = await self.execute_rag_with_retry(
                query,
                chat_history.get_messages(),
                custom_system_prompt=system_prompt
            )
        
        self.logger.debug(f"AGENT ROUTING: RAG response received after trying {len(queries_tried)} queries,"
                         f" response length: {len(str(rag_response))}")
        self.logger.debug(f"AGENT ROUTING: RAG output: {rag_response['output'][:200]}...")
        self.logger.debug(f"AGENT ROUTING: Queries tried: {queries_tried}")
        
        return rag_response, queries_tried


class NonRAGResponseStrategy(ResponseStrategy):
    """Strategy for generating non-RAG responses."""
    
    async def execute(self, query: str, chat_history: ChatHistory, 
                     custom_system_prompt: str = None, prompt_style: str = "default") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Generate a non-RAG response.
        
        Args:
            query: The user query
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            The rag_response will be None for this strategy
        """
        self.logger.debug(f"NonRAGResponseStrategy.execute - Starting with query: {query[:50]}...")
        
        no_rag_response, queries_tried = await self.invoke_agent_with_fallback(
            query,
            "standard",
            chat_history,
            prompt_style
        )
        
        # Validate response before returning
        if no_rag_response is None:
            self.logger.warning("NonRAGResponseStrategy.execute - Received None no_rag_response")
            no_rag_response = {"output": f"Error: Failed to generate non-RAG response for '{query[:30]}...'"}
        elif not isinstance(no_rag_response, dict):
            self.logger.warning(f"NonRAGResponseStrategy.execute - Expected dict, got {type(no_rag_response)}")
            # Convert string or other type to dict format
            if isinstance(no_rag_response, str):
                no_rag_response = {"output": no_rag_response}
            else:
                no_rag_response = {"output": f"Unexpected response type: {type(no_rag_response)}"}
        
        self.logger.debug(f"NonRAGResponseStrategy.execute - Completed, returning response type: {type(no_rag_response)}")
        
        return None, no_rag_response, queries_tried


class DualResponseStrategy(ResponseStrategy):
    """Strategy for generating both RAG and non-RAG responses."""
    
    async def execute(self, query: str, chat_history: ChatHistory, 
                     custom_system_prompt: str = None, prompt_style: str = "default") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Generate both RAG and non-RAG responses.
        
        Args:
            query: The user query
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            Both rag_response and no_rag_response will be provided
        """
        self.logger.debug(f"DualResponseStrategy.execute - Starting with query: {query[:50]}...")
        
        # Generate the RAG response
        rag_strategy = RAGResponseStrategy(self.chat_service)
        rag_response, _, rag_queries = await rag_strategy.execute(
            query, 
            chat_history,
            custom_system_prompt, 
            prompt_style
        )
        
        # Generate the non-RAG response
        no_rag_response, queries_tried_std = await self.invoke_agent_with_fallback(
            query,
            "standard",
            chat_history,
            prompt_style
        )
        
        # Validate responses before returning
        if no_rag_response is None:
            self.logger.warning("DualResponseStrategy.execute - Received None no_rag_response")
            no_rag_response = {"output": f"Error: Failed to generate non-RAG response for '{query[:30]}...'"}
        elif not isinstance(no_rag_response, dict):
            self.logger.warning(f"DualResponseStrategy.execute - Expected dict for no_rag_response, got {type(no_rag_response)}")
            # Convert string or other type to dict format
            if isinstance(no_rag_response, str):
                no_rag_response = {"output": no_rag_response}
            else:
                no_rag_response = {"output": f"Unexpected response type: {type(no_rag_response)}"}
        
        # Don't need to validate rag_response since RAGResponseStrategy already does that
        
        # Combine the queries tried
        queries_tried = rag_queries + queries_tried_std
        
        self.logger.debug(f"DualResponseStrategy.execute - Completed, returning response types: RAG={type(rag_response)}, non-RAG={type(no_rag_response)}")
        
        return rag_response, no_rag_response, queries_tried


class DatabaseResponseStrategy(ResponseStrategy):
    """Strategy for handling database queries."""
    
    async def execute(self, query: str, chat_history: ChatHistory, 
                     custom_system_prompt: str = None, prompt_style: str = "default") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Generate responses for database queries.
        
        Args:
            query: The user query
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            Both database response (as rag_response) and standard response (as no_rag_response) will be provided
        """
        self.logger.debug(f"DatabaseResponseStrategy.execute - Starting with query: {query[:50]}...")
        self.logger.debug(f"==== AGENT ROUTING: Selected DATABASE AGENT ====")
        
        # Generate database agent response (primary)
        rag_response, queries_tried_db = await self.invoke_agent_with_fallback(
            query,
            "database",
            chat_history,
            prompt_style
        )
        
        # Also generate standard agent response (secondary)
        no_rag_response, queries_tried_std = await self.invoke_agent_with_fallback(
            query,
            "standard",
            chat_history,
            prompt_style
        )
        
        # Validate responses before returning
        if rag_response is None:
            self.logger.warning("DatabaseResponseStrategy.execute - Received None rag_response")
            rag_response = {"output": f"Error: Failed to generate database response for '{query[:30]}...'"}
        elif not isinstance(rag_response, dict):
            self.logger.warning(f"DatabaseResponseStrategy.execute - Expected dict for rag_response, got {type(rag_response)}")
            # Convert string or other type to dict format
            if isinstance(rag_response, str):
                rag_response = {"output": rag_response}
            else:
                rag_response = {"output": f"Unexpected response type: {type(rag_response)}"}
                
        if no_rag_response is None:
            self.logger.warning("DatabaseResponseStrategy.execute - Received None no_rag_response")
            no_rag_response = {"output": f"Error: Failed to generate standard response for '{query[:30]}...'"}
        elif not isinstance(no_rag_response, dict):
            self.logger.warning(f"DatabaseResponseStrategy.execute - Expected dict for no_rag_response, got {type(no_rag_response)}")
            # Convert string or other type to dict format
            if isinstance(no_rag_response, str):
                no_rag_response = {"output": no_rag_response}
            else:
                no_rag_response = {"output": f"Unexpected response type: {type(no_rag_response)}"}
        
        # Combine the queries tried
        queries_tried = queries_tried_db + queries_tried_std
        
        self.logger.debug(f"DatabaseResponseStrategy.execute - Completed, returning response types: DB={type(rag_response)}, STD={type(no_rag_response)}")
        
        return rag_response, no_rag_response, queries_tried