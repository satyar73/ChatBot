"""
Response strategies for handling different types of chat queries.
"""
from typing import Dict, List, Tuple, Any, Optional, Union, TYPE_CHECKING
import re

from app.config.chat_model_config import ChatModelConfig
from app.models.chat_models import ChatHistory, Source
from langchain.agents import AgentExecutor
from app.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from app.agents.chat_agents import AgentManager

class ResponseStrategy:
    """Base class for response generation strategies."""
    
    def __init__(self, chat_service, agent_manager: "AgentManager"):
        """
        Initialize the strategy with a reference to the chat service.
        
        Args:
            chat_service: The ChatService instance
            agent_manager: The AgentManager instance
        """
        self.chat_service = chat_service
        self.agent_manager = agent_manager
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", "DEBUG")
    
    @staticmethod
    def get_strategy(query: str, mode: str, chat_service, agent_manager: "AgentManager") -> "ResponseStrategy":
        """
        Get the appropriate response strategy based on the query and mode.
        
        Args:
            query: The user's query
            mode: The response mode (rag, no_rag, both, needl)
            chat_service: The chat service instance
            agent_manager: The agent manager instance
            
        Returns:
            An instance of the appropriate ResponseStrategy subclass
        """
        if mode == "no_rag":
            return NonRAGResponseStrategy(chat_service, agent_manager)
        elif mode == "both":
            return DualResponseStrategy(chat_service, agent_manager)
        elif mode == "needl":
            return NeedlResponseStrategy(chat_service, agent_manager)
        elif query.lower().startswith("database:"):
            return DatabaseResponseStrategy(chat_service, agent_manager)
        else:
            return RAGResponseStrategy(chat_service, agent_manager)
    
    async def execute(self, 
                    chat_model_config: ChatModelConfig, 
                    query: str, 
                    chat_history: ChatHistory, 
                    custom_system_prompt: str = None, 
                    prompt_style: str = "default",
                    client_name: str = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Execute the strategy to generate a response.
        
        Args:
            chat_model_config: The chat model configuration
            query: The user query
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            client_name: Optional client name for namespace-specific retrieval
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            Either rag_response or no_rag_response may be None depending on strategy
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    async def execute_rag_with_retry(self,
                                chat_model_config: ChatModelConfig, 
                                query: str, 
                                history: List, 
                                max_attempts: int = 3,
                                custom_system_prompt: str = None,
                                rag_agent=None,
                                prompt_style: str = "default",
                                client_name: str = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute RAG with multiple query formulations and retry logic.
        
        Args:
            chat_model_config: Configuration for the chat model
            query: The original user query
            history: Chat history
            max_attempts: Maximum number of query attempts
            custom_system_prompt: Optional custom system prompt
            rag_agent: Optional RAG agent object to use for execution
            prompt_style: The prompt style to use
            client_name: Optional client name for namespace-specific retrieval
            
        Returns:
            Tuple of (best_response, queries_tried)
        """
        # Implementation duplicates ChatService._execute_rag_with_retry to avoid access to protected method
        self.logger.debug(f"QUERY REWRITING: Starting execute_rag_with_retry with query: {query[:100]}")

        # Get the appropriate RAG agent
        if rag_agent is None:
            # Use the system prompt to get a RAG agent if no agent provided
            rag_agent = self._get_rag_agent(
                custom_system_prompt=custom_system_prompt,
                query=query,
                chat_model_config=chat_model_config,
                prompt_style=prompt_style,
                client_name=client_name
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
            # Create a clean input to the agent without duplicating history
            # This is critical to avoid message duplication
            
            # Check if the query is already in the history to avoid duplication
            # This happens with document service requests where the full formatting template
            # is both sent as input and added to history
            is_query_in_history = False
            if history:
                for msg in history:
                    # Check if this message is from the human and contains the identical query
                    # We use exact matching instead of containment to be more precise
                    if hasattr(msg, 'type') and msg.type == 'human' and msg.content.strip() == q.strip():
                        is_query_in_history = True
                        self.logger.debug(f"Identical query already in history, using empty history to prevent duplication")
                        break
            
            # If the query is already in history, don't pass the history again
            # This prevents duplication in the prompt
            return await rag_agent.ainvoke(
                {"input": q, "history": [] if is_query_in_history else history},
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
    
    async def invoke_agent_with_fallback(self,
                                  chat_model_config: ChatModelConfig,
                                  query: str,
                                  agent_name: str,
                                  chat_history: List,
                                  system_prompt: str = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Invoke an agent with fallback to standard agent if needed.
        
        Args:
            chat_model_config: Configuration for the chat model
            query: The query to process
            agent_name: The name of the agent to use ("rag", "standard", or "database")
            chat_history: The chat history
            system_prompt: Optional system prompt
            
        Returns:
            Tuple of (agent_response, queries_tried)
        """
        # Map agent_name to the correct agent_type string
        agent_type = agent_name.lower()  # Convert to lowercase to match the expected values
        
        # Get the appropriate agent
        agent = self.agent_manager.get_agent(
            chat_model_config=chat_model_config,
            agent_type=agent_type,
            custom_system_prompt=system_prompt,
            query=query
        )
        
        # Execute the agent
        response = await agent.ainvoke(
            {"input": query, "history": chat_history},
            include_run_info=True
        )
        
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
                # Ensure all required fields have default values
                formatted_sources.append(Source(
                    title=source.get("title", "Unknown Source"),
                    url=source.get("url", ""),
                    content=source.get("content", "")
                ))
            else:
                # Handle string or other non-dict sources by providing defaults for all required fields
                formatted_sources.append(Source(
                    title="Unknown Source",
                    url="",
                    content=str(source)
                ))

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

    def _get_rag_agent(self,
                      expected_answer: str = None,
                      custom_system_prompt: str = None,
                      query: str = None,
                      chat_model_config: ChatModelConfig = None,
                      prompt_style: str = "default",
                      client_name: str = None) -> "AgentExecutor":
        """
        Get a RAG agent with the specified configuration.
        
        Args:
            expected_answer: Optional expected answer to enhance the system prompt
            custom_system_prompt: Optional custom system prompt
            query: Optional query to use for agent configuration
            chat_model_config: Optional chat model configuration
            prompt_style: The prompt style to use
            client_name: Optional client name for namespace-specific retrieval
            
        Returns:
            An AgentExecutor instance configured for RAG
        """
        # If we have an expected answer, enhance the system prompt
        if expected_answer:
            if custom_system_prompt:
                custom_system_prompt = f"{custom_system_prompt}\n\nExpected answer: {expected_answer}"
            else:
                custom_system_prompt = f"Expected answer: {expected_answer}"
                
        # Get the RAG agent with the appropriate configuration
        return self.agent_manager.get_agent(
            chat_model_config=chat_model_config,
            query=query,
            agent_type="rag",
            custom_system_prompt=custom_system_prompt,
            prompt_style=prompt_style,
            client_name=client_name
        )

    async def _get_rag_response(self,
                             query: str,
                             chat_history: List,
                             custom_system_prompt: str = None,
                             chat_model_config: ChatModelConfig = None) -> Dict[str, Any]:
        """
        Get a RAG response for the given query.
        
        Args:
            query: The query to process
            chat_history: The chat history
            custom_system_prompt: Optional custom system prompt
            chat_model_config: Optional chat model configuration
            
        Returns:
            The RAG response
        """
        rag_agent = self._get_rag_agent(
            custom_system_prompt=custom_system_prompt,
            query=query,
            chat_model_config=chat_model_config
        )
        
        return await rag_agent.ainvoke(
            {"input": query, "history": chat_history},
            include_run_info=True
        )

class RAGResponseStrategy(ResponseStrategy):
    """Strategy for generating responses using RAG."""

    async def execute(self, 
                    chat_model_config: ChatModelConfig, 
                    query: str, 
                    chat_history: ChatHistory, 
                    custom_system_prompt: str = None,
                    prompt_style: str = "default",
                    client_name: str = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Execute the RAG response strategy.
        
        Args:
            chat_model_config: Configuration for the chat model
            query: The user's query
            chat_history: Chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            client_name: Optional client name for namespace-specific retrieval
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            Either rag_response or no_rag_response may be None depending on strategy
        """
        
        rag_response, queries_tried = await self.execute_rag_with_retry(
                                                        chat_model_config=chat_model_config,
                                                        query=query,
                                                        history=chat_history.get_messages(),
                                                        custom_system_prompt=custom_system_prompt,
                                                        prompt_style=prompt_style,
                                                        client_name=client_name
                                                    )
        return rag_response, None, queries_tried

class NonRAGResponseStrategy(ResponseStrategy):
    """Strategy for generating responses without using RAG."""

    async def execute(self, 
                    chat_model_config: ChatModelConfig, 
                    query: str, 
                    chat_history: ChatHistory, 
                    custom_system_prompt: str = None,
                    prompt_style: str = "default",
                    client_name: str = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Execute the non-RAG response strategy.
        
        Args:
            chat_model_config: Configuration for the chat model
            query: The user's query
            chat_history: Chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            client_name: Optional client name (not used in non-RAG strategy)
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            Either rag_response or no_rag_response may be None depending on strategy
        """
        # Get non-RAG agent
        non_rag_agent = self.agent_manager.get_agent(
            chat_model_config=chat_model_config,
            agent_type="standard",
            custom_system_prompt=custom_system_prompt
        )
        
        # Execute non-RAG agent
        non_rag_response = await non_rag_agent.ainvoke(
            {"input": query, "history": chat_history.get_messages()},
            include_run_info=True
        )
        
        return None, non_rag_response, [query]

class DualResponseStrategy(ResponseStrategy):
    """Strategy for generating both RAG and non-RAG responses."""

    async def execute(self, 
                    chat_model_config: ChatModelConfig, 
                    query: str, 
                    chat_history: ChatHistory, 
                    custom_system_prompt: str = None,
                    prompt_style: str = "default",
                    client_name: str = None) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """
        Execute both RAG and non-RAG response strategies.
        
        Args:
            chat_model_config: Configuration for the chat model
            query: The user's query
            chat_history: Chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            client_name: Optional client name for namespace-specific retrieval
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
        """
        # Get RAG response
        rag_response, queries_tried = await self.execute_rag_with_retry(
            chat_model_config=chat_model_config,
            query=query,
            history=chat_history.get_messages(),
            custom_system_prompt=custom_system_prompt,
            prompt_style=prompt_style,
            client_name=client_name
        )
        
        # Create a temporary strategy just to format sources
        temp_strategy = ResponseStrategy(self.chat_service, self.agent_manager)
        # Extract sources from the RAG response if available
        sources = temp_strategy.format_sources(rag_response)

        # Get non-RAG response
        non_rag_agent = self.agent_manager.get_agent(
            chat_model_config=chat_model_config,
            agent_type="standard",
            custom_system_prompt=custom_system_prompt
        )
        non_rag_response = await non_rag_agent.ainvoke(
            {"input": query, "history": chat_history.get_messages()},
            include_run_info=True
        )
        
        return rag_response, non_rag_response, queries_tried

class DatabaseResponseStrategy(ResponseStrategy):
    """Strategy for handling database-related queries."""

    async def execute(self, 
                    chat_model_config: ChatModelConfig, 
                    query: str, 
                    chat_history: ChatHistory, 
                    custom_system_prompt: str = None,
                    prompt_style: str = "default",
                    client_name: str = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Execute the database response strategy.
        
        Args:
            chat_model_config: Configuration for the chat model
            query: The user's query
            chat_history: Chat history
            custom_system_prompt: Optional custom system prompt
            prompt_style: The prompt style to use
            client_name: Optional client name (not used in database strategy)
            
        Returns:
            Tuple of (rag_response, no_rag_response, queries_tried)
            Either rag_response or no_rag_response may be None depending on strategy
        """
        # Get database agent
        db_agent = self.agent_manager.get_agent(
            chat_model_config=chat_model_config,
            agent_type="database",
            custom_system_prompt=custom_system_prompt
        )
        
        # Execute database agent
        db_response = await db_agent.ainvoke(
            {"input": query, "history": chat_history.get_messages()},
            include_run_info=True
        )
        
        # Return the database response as the RAG response (since it's the primary response)
        # and None as the non-RAG response
        return db_response, None, [query]
        
class NeedlResponseStrategy(ResponseStrategy):
    """Strategy for handling queries using the Needl.ai API."""
    
    async def execute(self, 
                    chat_model_config: ChatModelConfig, 
                    query: str, 
                    chat_history: ChatHistory, 
                    custom_system_prompt: str = None,
                    prompt_style: str = "default",
                    client_name: str = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Execute the Needl response strategy.
        
        Args:
            chat_model_config: Configuration for the chat model (not used for Needl)
            query: The user's query to send to Needl
            chat_history: Chat history (not used for Needl)
            custom_system_prompt: Not used for Needl
            prompt_style: Not used for Needl
            client_name: Not used for Needl
            
        Returns:
            Tuple of (needl_response, None, [query])
            The needl_response is formatted to match the expected RAG response format
        """
        from app.services.chat.needl_service import needl_service
        
        self.logger.info(f"Executing Needl query: {query[:100]}...")
        
        # Query the Needl API
        needl_response = await needl_service.query(query)
        
        # Format the response to match the expected format
        formatted_response = needl_service.format_response_as_chat_message(needl_response)
        
        # Add the raw Needl response for debugging/reference
        if isinstance(formatted_response, dict) and "response" in formatted_response:
            formatted_response["response"]["needl_raw_response"] = needl_response
        
        self.logger.info("Needl query executed successfully")
        
        # Return the Needl response as the RAG response (since it's the primary response)
        # and None as the non-RAG response
        return formatted_response, None, [query]