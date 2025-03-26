"""
Service layer for handling agent queries and responses with enhanced RAG query rewriting.
"""
import re
import sys
import time
from typing import Dict, List, Any, Tuple, Union

from app.agents.chat_agents import agent_manager
from app.config import prompt_config
from app.config.chat_config import ChatConfig
from app.models.chat_models import ChatHistory, ResponseContent, ResponseMessage, Source, Message
from app.services.cache_service import chat_cache
from app.services.qa_service import qa_service
from app.utils.logging_utils import get_logger
from app.utils.semantic_filtering import SemanticFilter


# Add this new class for query rewriting
class QueryRewriter:
    """Class for rewriting and expanding user queries to improve RAG retrieval."""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.QueryRewriter", "DEBUG")
        self.logger.debug("QueryRewriter initialized")

        # Get config to access feature flags
        self.config = ChatConfig()
        self.logger.debug(
            f"Semantic similarity filtering enabled: {self.config.CHAT_FEATURE_FLAGS.get('semantic_similarity_filtering', False)}")

        # Common marketing and attribution terminology mapping for expansions
        self.term_expansions = {
            "mmm": ["marketing mix modeling", "marketing mix model"],
            "ltv": ["lifetime value", "customer lifetime value"],
            "roas": ["return on ad spend", "return on advertising spend"],
            "roi": ["return on investment"],
            "cac": ["customer acquisition cost", "cost of acquisition"],
            "cpa": ["cost per acquisition", "cost per action"],
            "cpc": ["cost per click"],
            "cpm": ["cost per mille", "cost per thousand impressions"],
            "ctr": ["click-through rate", "clickthrough rate"],
            "cvr": ["conversion rate"],
            "kpi": ["key performance indicator"],
            "sem": ["search engine marketing"],
            "seo": ["search engine optimization"],
            "ppc": ["pay per click", "pay-per-click"],
            "attribution multiplier": ["advanced attribution multiplier", "attribution factor",
                                       "incrementality factor"],
        }

        # Marketing concept synonyms for query expansion
        self.synonym_mappings = {
            "attribution": ["credit assignment", "channel impact", "marketing impact", "touchpoint analysis"],
            "incrementality": ["lift", "causal impact", "true impact", "incremental value"],
            "channel": ["platform", "medium", "touchpoint", "marketing source"],
            "conversion": ["purchase", "transaction", "sale", "signup", "acquisition"],
            "effectiveness": ["performance", "impact", "results", "efficiency", "success"],
            "marketing": ["advertising", "promotion", "campaign", "media spend"],
            "measurement": ["tracking", "analytics", "analysis", "evaluation", "assessment"],
            "optimization": ["improvement", "enhancement", "refinement", "maximization"],
            "advanced attribution": ["advanced attribution multiplier", "attribution adjustment",
                                     "incrementality coefficient"],
        }

        self.technical_terms = {
            "advanced attribution multiplier": (
                "A coefficient used in advanced attribution to adjust platform-reported metrics "
                "to reflect true incremental value of marketing channels"
            ),
            "attribution multiplier": (
                "A factor applied to channel attribution to account for over-reporting "
                "or under-reporting in marketing platforms"
            )
        }

    def expand_abbreviations(self, query: str) -> str:
        """Expand common marketing abbreviations in the query."""
        expanded_query = query
        for abbr, expansions in self.term_expansions.items():
            # Only match whole words (with word boundaries)
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, expanded_query, re.IGNORECASE):
                # Choose the first expansion as default
                expanded_query = re.sub(pattern, expansions[0], expanded_query, flags=re.IGNORECASE)

        return expanded_query

    def add_synonyms(self, query: str) -> str:
        """Add relevant synonyms to the query to improve matching."""
        for term, synonyms in self.synonym_mappings.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                # Add the first two synonyms to the query
                additional_terms = " " + " ".join(synonyms[:2])
                return query + additional_terms

        return query

    @staticmethod
    def create_broader_query(query: str) -> str:
        """Create a more general version of the query by removing specific constraints."""
        # Remove specific qualifiers to make query more general
        broader_query = re.sub(r'\b(specific|exactly|precise|only|detailed)\b', '', query, flags=re.IGNORECASE)

        # Remove time constraints that might limit results
        broader_query = re.sub(r'\b(in the last|recent|latest|current|today\'s|this month\'s|this year\'s)\b', '',
                               broader_query, flags=re.IGNORECASE)

        # Remove format requests that might narrow results
        broader_query = re.sub(r'\b(table|graph|chart|report|dashboard)\b', '', broader_query, flags=re.IGNORECASE)

        # Focus on core marketing concepts if the query is getting too diluted
        marketing_terms = [
            "attribution", "marketing", "campaign", "channel", "advertising",
            "measurement", "performance", "metrics", "conversion", "funnel"
        ]

        # Extract core marketing concepts from the query
        core_terms = []
        for term in marketing_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', query, re.IGNORECASE):
                core_terms.append(term)

        # If we found core marketing terms, construct a query focused on them
        if core_terms:
            core_query = " ".join(core_terms)
            # Return both the broader query and a core concept query
            return f"{broader_query.strip()} {core_query}"

        return broader_query.strip()

    def generate_alt_queries(self, original_query: str) -> List[str]:
        """Generate alternative formulations of the query to improve retrieval."""
        self.logger.debug(f"Generating alternative queries for: {original_query}")

        alt_queries = [original_query]  # Always include the original query

        for term in self.technical_terms:
            if term.lower() in original_query.lower():
                # Add the term by itself for precise matching
                alt_queries.append(term)
                # Add the term with its definition
                alt_queries.append(f"{term} {self.technical_terms[term]}")
                self.logger.debug(f"Added technical term query: {term}")

        # Apply basic query expansions
        expanded = self.expand_abbreviations(original_query)
        if expanded != original_query:
            alt_queries.append(expanded)
            self.logger.debug(f"Added abbreviation-expanded query: {expanded}")

        # Add synonyms
        with_synonyms = self.add_synonyms(original_query)
        if with_synonyms != original_query:
            alt_queries.append(with_synonyms)
            self.logger.debug(f"Added synonym-enhanced query: {with_synonyms}")

        # Also add synonyms to the expanded version if different
        if expanded != original_query:
            expanded_with_synonyms = self.add_synonyms(expanded)
            if expanded_with_synonyms != expanded:
                alt_queries.append(expanded_with_synonyms)
                self.logger.debug(f"Added expanded+synonym query: {expanded_with_synonyms}")

        # Generate broader queries for fallback
        broader_result = self.create_broader_query(original_query)

        # Check if broader_result is a tuple (has both broader and core queries)
        if isinstance(broader_result, tuple):
            broader, core = broader_result
            alt_queries.append(broader)
            alt_queries.append(core)
            self.logger.debug(f"Added broader query: {broader}")
            self.logger.debug(f"Added core concept query: {core}")
        else:
            alt_queries.append(broader_result)
            self.logger.debug(f"Added broader query: {broader_result}")

        # Make sure we don't have duplicates (text matching)
        unique_queries = list(dict.fromkeys(alt_queries))

        # Use semantic similarity filtering if enabled
        if self.config.CHAT_FEATURE_FLAGS.get("semantic_similarity_filtering", False):
            self.logger.debug(f"Applying semantic similarity filtering to {len(unique_queries)} queries")

            # Apply semantic filtering to remove similar queries
            filtered_queries = SemanticFilter.filter_similar_queries(
                unique_queries,
                similarity_threshold=0.7  # Configurable threshold
            )

            # Log the filtering results
            removed_count = len(unique_queries) - len(filtered_queries)
            self.logger.debug(f"Semantic filtering removed {removed_count} similar queries")

            # Rank by diversity (optional, helps prioritize diverse queries)
            ranked_queries = SemanticFilter.rank_queries_by_diversity(filtered_queries, original_query)

            self.logger.debug(f"Generated {len(ranked_queries)} semantically diverse queries")
            return ranked_queries
        else:
            self.logger.debug(f"Generated {len(unique_queries)} alternative queries (semantic filtering disabled)")
            return unique_queries


class ChatService:
    """Service for managing chat interactions with agents."""

    def __init__(self):
        # Create a dictionary to store chat histories for different sessions
        self.chat_histories = {}
        self.logger = get_logger(f"{__name__}.ChatService", "DEBUG")
        self.logger.debug("ChatService initialized")
        # Explicit print to check if output is working at all
        print("ChatService initialized", file=sys.stderr)

        self.config = ChatConfig()

        # Initialize the query rewriter
        self.query_rewriter = QueryRewriter()

        # Use the QA service
        self.qa_service = qa_service

    def _get_answer(self, question):
        """
        Get the expected answer for a question if it exists in the cache.

        Args:
            question (str): The question to look up

        Returns:
            str or None: The expected answer if found, None otherwise
        """
        # Use the QA service to get the answer
        return self.qa_service.get_answer(question)

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

    def _is_empty_or_inadequate_response(self, response: Dict) -> bool:
        """
        Enhanced function to check if a response doesn't adequately answer the query.
        Looks for both "no information" phrases and phrases that indicate the term wasn't found.

        Args:
            response: The response from the RAG agent

        Returns:
            bool: True if the response is empty or inadequate, False otherwise
        """
        output = response.get('output', '').lower()

        # Check for standard "no information" phrases
        no_info_phrases = [
            "i don't have information",
            "i don't have specific information",
            "no information available",
            "i couldn't find",
            "not found in",
            "i don't have access",
            "i don't have details",
            "information is not provided"
        ]

        # Check for phrases indicating a term wasn't found in documents
        term_not_found_phrases = [
            "does not appear in",
            "doesn't appear in",
            "is not mentioned in",
            "isn't mentioned in",
            "not present in",
            "could not find",
            "couldn't find",
            "not explicitly",
            "not specifically",
            "is not defined in",
            "isn't defined in",
            "is absent from",
            "no specific mention of",
            "no explicit reference to",
            "not explicitly mentioned",
            "not specifically mentioned",
            "not explicitly defined",
            "not specifically defined",
            "does not appear explicitly",
            "doesn't appear explicitly",
            "not appear explicitly"
        ]

        # Check if any "no information" phrase is present
        has_no_info = any(phrase in output for phrase in no_info_phrases)

        # Check if any "term not found" phrase is present
        term_not_found = any(phrase in output for phrase in term_not_found_phrases)

        # Log the result for debugging
        if has_no_info or term_not_found:
            self.logger.debug(f"QUERY REWRITING: Response deemed inadequate because: " +
                              ("contains 'no info' phrases" if has_no_info else "") +
                              ("contains 'term not found' phrases" if term_not_found else ""))

        return has_no_info or term_not_found

    async def _execute_rag_with_retry(self, query: str, history: List, max_attempts: int = 3,
                                      custom_system_prompt: str = None,
                                      rag_agent=None) -> Tuple[Dict, List[str]]:
        """
        Execute RAG with multiple query formulations and retry logic.

        Args:
            query: The original user query
            history: Chat history
            max_attempts: Maximum number of query attempts
            custom_system_prompt: Optional custom system prompt
            rag_agent: Optional RAG agent object to use for execution (if provided, custom_system_prompt is ignored)
        Returns:
            Tuple of (best_response, queries_tried)
        """
        self.logger.debug(f"QUERY REWRITING: Starting _execute_rag_with_retry with query: {query[:100]}")

        # Generate alternative queries
        alt_queries = self.query_rewriter.generate_alt_queries(query)
        self.logger.debug(f"QUERY REWRITING: Generated {len(alt_queries)} alternative queries for RAG")
        self.logger.debug(f"QUERY REWRITING: Alternatives: {alt_queries}")

        all_responses = []
        queries_tried = []

        # Get the appropriate RAG agent
        if rag_agent is None:
            # Use the system prompt to get a RAG agent if no agent provided
            rag_agent = agent_manager.get_rag_agent(custom_system_prompt)
            self.logger.debug(
                f"QUERY REWRITING: Using RAG agent with custom system prompt: {custom_system_prompt is not None}")
        else:
            # Use the provided agent (which may have expected answer)
            self.logger.debug(f"QUERY REWRITING: Using pre-configured RAG agent")

        # XXX this is a hack. Ideally I should be able to have the LLM detect this
        # Special handling for technical terms
        for term in self.query_rewriter.technical_terms:
            if term.lower() in query.lower():
                # For technical terms, try the exact term query first
                technical_query = term
                self.logger.debug(f"QUERY REWRITING: Technical term detected, trying exact term: {technical_query}")

                technical_response = await rag_agent.ainvoke(
                    {"input": technical_query, "history": history},
                    include_run_info=True
                )

                # If this gives a good result, use it
                if not self._is_empty_or_inadequate_response(technical_response):
                    self.logger.debug(f"QUERY REWRITING: Exact technical term query produced good results")
                    return technical_response, [query, technical_query]
                else:
                    self.logger.debug(
                        f"QUERY REWRITING: Technical term query produced inadequate results, will try alternatives")

        # Try the original query first
        original_query = alt_queries[0]
        queries_tried.append(original_query)

        self.logger.debug(f"QUERY REWRITING: Trying original query: {original_query}")
        original_response = await rag_agent.ainvoke(
            {"input": original_query, "history": history},
            include_run_info=True
        )
        all_responses.append((original_response, original_query))

        # If the first response seems good, return it
        if not self._is_empty_or_inadequate_response(original_response):
            self.logger.debug(f"QUERY REWRITING: Original query produced good results, no need for alternatives")
            return original_response, queries_tried

        # Otherwise, try alternative queries
        self.logger.debug(f"QUERY REWRITING: Original query produced inadequate results, trying alternatives")

        # Remove the original query as we've already tried it
        alt_queries = alt_queries[1:]

        for i, alt_query in enumerate(alt_queries):
            if i >= max_attempts - 1:  # We already tried the original query
                break

            queries_tried.append(alt_query)
            self.logger.debug(f"QUERY REWRITING: Trying alternative query {i + 1}: {alt_query}")

            alt_response = await rag_agent.ainvoke(
                {"input": alt_query, "history": history},
                include_run_info=True
            )
            all_responses.append((alt_response, alt_query))

            # If this alternative query worked well, return it
            if not self._is_empty_or_inadequate_response(alt_response):
                self.logger.debug(f"QUERY REWRITING: Alternative query {i + 1} produced good results")
                return alt_response, queries_tried
            else:
                self.logger.debug(f"QUERY REWRITING: Alternative query {i + 1} produced inadequate results")

        # If we get here, none of the queries produced good results
        # Find the best response among all attempts
        best_response = original_response

        # Optional: Implement a scoring mechanism to pick the best response
        # For now, just returning the original response as a fallback
        self.logger.debug(f"QUERY REWRITING: No alternative queries produced better results, returning original")

        return best_response, queries_tried

    async def _invoke_agent_with_fallback(self,
                                          actual_query: str,
                                          agent_name: str,
                                          chat_history: ChatHistory,
                                          prompt_style: str = "default") -> Tuple[Dict, List[str]]:
        # Call the agent with the specified style
        self.logger.debug(f"==== AGENT ROUTING: Selected {agent_name} AGENT with {prompt_style} style ====")
        self.logger.debug(f"AGENT ROUTING: Calling {agent_name} agent with query: {actual_query[:100]}...")

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
            {"input": actual_query, "history": chat_history.get_messages()},
            include_run_info=True
        )
        self.logger.debug(f"AGENT ROUTING: {agent_name} response received, length: {len(str(response))}")

        return response, [actual_query]

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
            session_id=session_id,
            system_prompt=custom_system_prompt,
            prompt_style=prompt_style
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

        cached_response, cached_rag_response, cached_no_rag_response = (None, None, None)
        # For database queries or force refresh, skip cache
        if is_database_query or force_refresh:
            if is_database_query:
                self.logger.info(f"Database query detected, bypassing cache: {user_input}")
        else:
            # Check cache for existing response
            cached_response, cache_hit = chat_cache.get_cached_response(query_hash)
            if cache_hit and ((mode != "no_rag" and cached_response["rag_response"] is None)
                              or (mode != "rag" and cached_response["no_rag_response"] is None)):
                # Previous request was for a different mode
                cached_rag_response, cached_no_rag_response = (
                    cached_response["rag_response"], cached_response["no_rag_response"])
                # Reset cache_hit to False
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
            rag_response, queries_tried_db = await self._invoke_agent_with_fallback(
                actual_query,
                "database",
                chat_history,
                prompt_style)
            no_rag_response, queries_tried_std = await self._invoke_agent_with_fallback(
                actual_query,
                "standard",
                chat_history,
                prompt_style)

            queries_tried = queries_tried_db + queries_tried_std
        else:
            # Generate response using agent executor with RAG - now with query rewriting
            if mode != "no_rag":
                self.logger.debug(f"==== AGENT ROUTING: Selected RAG AGENT with QUERY REWRITING ====")
                self.logger.debug(
                    f"AGENT ROUTING: Calling RAG agent with potential query rewrites: {actual_query[:100]}...")

                # Get expected answer for this query (if it exists)
                # Use test_query instead of user_input when in test mode
                lookup_query = test_query if force_refresh else user_input
                expected_answer = self._get_answer(lookup_query)
                if force_refresh:
                    self.logger.debug(f"TEST MODE: Looking up expected answer using: '{lookup_query}'")

                # Debug feature flags
                self.logger.debug(f"Feature flags: {self.config.CHAT_FEATURE_FLAGS}")

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
                use_expected_answer = self.config.CHAT_FEATURE_FLAGS.get("expected_answer_enrichment", False)
                self.logger.debug(f"Expected answer enrichment enabled: {use_expected_answer}")

                # Skip query rewriting and expected answer handling when using custom system prompt
                if custom_system_prompt:
                    self.logger.info(f"Using custom system prompt - skipping query rewriting"
                                     f" and expected answer handling")
                    # Use the original query directly with the custom system prompt
                    rag_agent = agent_manager.get_rag_agent(custom_system_prompt=custom_system_prompt)
                    rag_response = await rag_agent.ainvoke(
                        {"input": actual_query, "history": chat_history.get_messages()},
                        include_run_info=True
                    )
                    queries_tried = [actual_query]  # Only tried the original query
                elif not use_expected_answer:
                    if expected_answer:
                        self.logger.info(f"Found expected answer but feature flag disabled:"
                                         f" '{user_input[:50]}...'")
                    else:
                        self.logger.info(f"Did not find expected answer and feature flag disabled:"
                                         f" '{user_input[:50]}...'")

                    rag_response, queries_tried = await self._execute_rag_with_retry(
                        actual_query,
                        chat_history.get_messages(),
                        custom_system_prompt=system_prompt
                    )
                elif expected_answer:
                    self.logger.info(f"Found expected answer for query and feature flag enabled:"
                                     f" '{user_input[:50]}...'")
                    # Get RAG agent with both system prompt and expected answer
                    rag_agent = agent_manager.get_rag_agent(
                        custom_system_prompt=system_prompt,
                        expected_answer=expected_answer
                    )

                    # Use the new method that tries multiple query formulations
                    rag_response, queries_tried = await self._execute_rag_with_retry(
                        actual_query,
                        chat_history.get_messages(),
                        rag_agent=rag_agent  # Pass the custom agent with expected answer
                    )
                else:
                    self.logger.info(f"Did not find expected answer for query and feature flag"
                                     f" is not enabled: '{user_input[:50]}...'")
                    rag_response, queries_tried = await self._execute_rag_with_retry(
                        actual_query,
                        chat_history.get_messages(),
                        custom_system_prompt=system_prompt
                    )

                self.logger.debug(
                    f"AGENT ROUTING: RAG response received after trying {len(queries_tried)} queries,"
                    f" response length: {len(str(rag_response))}")
                self.logger.debug(f"AGENT ROUTING: RAG output: {rag_response['output'][:200]}...")
                self.logger.debug(f"AGENT ROUTING: Queries tried: {queries_tried}")
            else:
                self.logger.debug(f"AGENT ROUTING: Using cached RAG response")
                rag_response = cached_rag_response
                queries_tried = [actual_query]  # Only for tracking

            # Generate response using agent executor without RAG
            if mode != "rag" and not is_database_query:
                no_rag_response, queries_tried_std = await self._invoke_agent_with_fallback(
                    actual_query,
                    "standard",
                    chat_history,
                    prompt_style
                )
            else:
                self.logger.debug(f"AGENT ROUTING: Using cached non-RAG response")
                no_rag_response = cached_no_rag_response

        # Extract sources from the RAG response
        primary_response = rag_response if mode != "no_rag" else no_rag_response

        sources = self._format_sources(rag_response) if rag_response is not None else []

        if primary_response is not None:
            # Handle both dictionary and string responses
            if isinstance(primary_response, dict):
                chat_history.add_ai_message(primary_response['output'])
            else:
                # For string responses (when custom system prompt is used)
                chat_history.add_ai_message(primary_response)
        elif rag_response is not None:
            if isinstance(rag_response, dict):
                chat_history.add_ai_message(rag_response['output'])
            else:
                chat_history.add_ai_message(rag_response)
        elif no_rag_response is not None:
            if isinstance(no_rag_response, dict):
                chat_history.add_ai_message(no_rag_response['output'])
            else:
                chat_history.add_ai_message(no_rag_response)
        else:
            self.logger.error("No valid response generated from RAG or non-RAG agents")

        # Format message history for response
        formatted_history = self._format_history(chat_history.get_messages())

        # Create the response content, ensuring we handle None values properly
        output = None
        if primary_response is not None:
            output = primary_response['output'] if isinstance(primary_response, dict) \
                else primary_response

        no_rag_output = None
        if no_rag_response is not None:
            no_rag_output = no_rag_response['output'] if isinstance(no_rag_response, dict) \
                else no_rag_response

        intermediate_steps = []
        if primary_response is not None and isinstance(primary_response, dict):
            intermediate_steps = primary_response.get('intermediate_steps', [])

        response_content = ResponseContent(
            input=user_input,
            history=formatted_history,
            output=output,
            no_rag_output=no_rag_output,
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
        rag_output_to_cache = None
        if rag_response is not None:
            rag_output_to_cache = rag_response['output'] if isinstance(rag_response, dict) else rag_response

        no_rag_output_to_cache = None
        if no_rag_response is not None:
            no_rag_output_to_cache = no_rag_response['output'] if isinstance(no_rag_response, dict) else no_rag_response

        chat_cache.cache_response(
            query_hash=query_hash,
            user_input=user_input,
            rag_response=rag_output_to_cache,
            no_rag_response=no_rag_output_to_cache,
            sources=sources,
            system_prompt=custom_system_prompt,
            prompt_style=prompt_style
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

    def _format_sources(self, rag_response: Union[Dict, str]) -> List[Source]:
        """
        Format sources from the RAG response.

        Args:
            rag_response: Response from the RAG agent, can be a dict or a string

        Returns:
            List of formatted Source objects
        """
        if rag_response is None:
            self.logger.warning("Received None rag_response in _format_sources")
            return []

        # Handle string responses (this happens when a custom system prompt is used)
        if isinstance(rag_response, str):
            self.logger.debug("Received string rag_response in _format_sources, no sources available")
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
            if (len(step) > 1 and step[0].tool == "search_rag_docs"
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
