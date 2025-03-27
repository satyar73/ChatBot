"""
Service layer for query enhancement and content enrichment.

This service extends the capabilities of the former QA service with more
sophisticated content enhancement for both chat prompts and content processing.
"""
import json
import re
from typing import Dict, List, Any, Optional, Union

from app.config.chat_config import ChatConfig
from app.utils.llm_client import LLMClientManager
from app.utils.logging_utils import get_logger
from app.utils.other_utlis import load_json
from app.utils.semantic_filtering import SemanticFilter


class EnhancementService:
    """Service for enhancing queries, prompts, and content processing."""

    def __init__(self):
        """Initialize the enhancement service."""
        self.config = ChatConfig()
        self.logger = get_logger(__name__, "DEBUG")
        self.logger.debug("EnhancementService initialized")
        
        # Initialize the QA data
        self.qa_data = {}
        self.load_qa_data()
        
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

        self.technical_definitions = {
            "advanced attribution multiplier": (
                "A coefficient used in advanced attribution to adjust platform-reported metrics "
                "to reflect true incremental value of marketing channels"
            ),
            "attribution multiplier": (
                "A factor applied to channel attribution to account for over-reporting "
                "or under-reporting in marketing platforms"
            ),
            "marketing mix modeling": (
                "A statistical analysis technique used to estimate the impact of various "
                "marketing activities on sales or other KPIs"
            )
        }

    def load_qa_data(self) -> None:
        """Load QA pairs from the configured JSON file and cache them in memory."""
        if hasattr(self.config, 'QA_SOURCE_FILE_JSON') and self.config.QA_SOURCE_FILE_JSON:
            json_file = self.config.QA_SOURCE_FILE_JSON
            self.qa_data = load_json(json_file)
            self.logger.info(f"Loaded {len(self.qa_data)} QA pairs from {json_file}")

    def get_answer(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Get the expected answer for a question if it exists in the cache.

        Args:
            question (str): The question to look up

        Returns:
            dict or None: The expected answer if found, None otherwise
        """
        if question is None:
            self.logger.warning("get_answer received None question")
            return None
            
        try:
            # Look for exact match
            if question in self.qa_data:
                return {
                    "question": question,
                    "answer": self.qa_data[question],
                    "confidence": 1.0
                }

            # Case-insensitive match
            for q, a in self.qa_data.items():
                if q.lower() == question.lower():
                    return {
                        "question": q,
                        "answer": a,
                        "confidence": 1.0
                    }

            # No match found
            return None
        except Exception as e:
            self.logger.error(f"Error in get_answer: {str(e)}")
            return None
    
    def expand_abbreviations(self, query: str) -> str:
        """
        Expand common marketing abbreviations in the query.
        
        Args:
            query: The original query
            
        Returns:
            Query with common abbreviations expanded
        """
        expanded_query = query
        for abbr, expansions in self.term_expansions.items():
            # Only match whole words (with word boundaries)
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, expanded_query, re.IGNORECASE):
                # Choose the first expansion as default
                expanded_query = re.sub(pattern, expansions[0], expanded_query, flags=re.IGNORECASE)

        return expanded_query

    def add_synonyms(self, query: str) -> str:
        """
        Add relevant synonyms to the query to improve matching.
        
        Args:
            query: The original query
            
        Returns:
            Query with relevant synonyms added
        """
        for term, synonyms in self.synonym_mappings.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                # Add the first two synonyms to the query
                additional_terms = " " + " ".join(synonyms[:2])
                return query + additional_terms

        return query
    
    def create_broader_query(self, query: str) -> str:
        """
        Create a more general version of the query by removing specific constraints.
        
        Args:
            query: The original query
            
        Returns:
            Broader version of the query
        """
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
    
    def enhance_query(self, original_query: str, conversation_context: List = None) -> Dict[str, Any]:
        """
        Generate enhanced query and alternative formulations to improve retrieval.
        
        Args:
            original_query: The original user query
            conversation_context: Optional conversation history context
            
        Returns:
            Dictionary with original query, enhanced query, and alternatives
        """
        self.logger.debug(f"Enhancing query: {original_query}")
        
        # Initialize the result dictionary
        result = {
            "original_query": original_query,
            "enhanced_query": original_query,
            "alt_queries": [original_query],
            "intent": "information",  # Default intent
            "qa_match": {"matched": False},
            "context_topics": [],
            "related_questions": []
        }
        
        # Check for matched QA pair
        qa_match = self.get_answer(original_query)
        if qa_match:
            result["qa_match"] = {"matched": True, **qa_match}
            # Also add as a related question
            result["related_questions"].append(original_query)
            
        # Process conversation context if provided
        context_topics = []
        if conversation_context:
            # Extract topic keywords from recent conversation
            for message in conversation_context[-3:]:  # Last 3 messages
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    # Extract marketing topics
                    for term in self.technical_definitions.keys():
                        if term.lower() in content.lower():
                            context_topics.append(term)
            
            result["context_topics"] = list(set(context_topics))  # Deduplicate
            
        # Generate alternative queries
        alt_queries = [original_query]  # Always include the original query

        # Check for technical terms and add them directly
        for term in self.technical_definitions:
            if term.lower() in original_query.lower():
                # Add the term by itself for precise matching
                alt_queries.append(term)
                # Add the term with its definition
                alt_queries.append(f"{term} {self.technical_definitions[term]}")

        # Apply basic query expansions
        expanded = self.expand_abbreviations(original_query)
        if expanded != original_query:
            alt_queries.append(expanded)

        # Add synonyms
        with_synonyms = self.add_synonyms(original_query)
        if with_synonyms != original_query:
            alt_queries.append(with_synonyms)

        # Create an enhanced query that combines expansions, technical definitions and context
        enhanced_parts = []
        # Add the original query
        enhanced_parts.append(original_query)
        
        # Add any technical term definitions that match
        for term, definition in self.technical_definitions.items():
            if term.lower() in original_query.lower():
                enhanced_parts.append(f"{term} {definition}")
                break  # Just add the first one to avoid making the query too long
                
        # Add alternate formulation if we expanded abbreviations
        if expanded != original_query:
            enhanced_parts.append(expanded)
            
        # Create the enhanced query
        enhanced_query = " ".join(enhanced_parts)
        
        # Add the enhanced query to the result
        result["enhanced_query"] = enhanced_query
        
        # Make sure we don't have duplicates in alt_queries
        unique_queries = list(dict.fromkeys(alt_queries))
        
        # Use semantic similarity filtering if enabled
        if getattr(self.config, 'CHAT_FEATURE_FLAGS', {}).get("semantic_similarity_filtering", False):
            self.logger.debug(f"Applying semantic similarity filtering to {len(unique_queries)} queries")

            # Apply semantic filtering to remove similar queries
            filtered_queries = SemanticFilter.filter_similar_queries(
                unique_queries,
                similarity_threshold=0.7  # Configurable threshold
            )

            # Rank by diversity (optional, helps prioritize diverse queries)
            ranked_queries = SemanticFilter.rank_queries_by_diversity(filtered_queries, original_query)
            result["alt_queries"] = ranked_queries
        else:
            result["alt_queries"] = unique_queries
            
        self.logger.debug(f"Enhanced query: {result}")
        return result
        
    def extract_key_concepts(self, answer: str) -> str:
        """
        Extract key concepts from an expected answer to guide response generation
        without revealing the exact answer.
        
        Args:
            answer (str): The full expected answer
            
        Returns:
            str: Formatted key concepts suitable for inclusion in a prompt
        """
        # Handle None or empty answers
        if answer is None:
            self.logger.warning("extract_key_concepts received None answer")
            return "- No key concepts available"
            
        if not answer.strip():
            self.logger.warning("extract_key_concepts received empty answer")
            return "- No key concepts available"
            
        try:
            # Break answer into sentences
            sentences = answer.split('.')
            
            # Filter out empty sentences
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            # Handle short expected answers differently
            if len(sentences) <= 2:
                # For very short answers, extract key concepts instead of using full sentences
                key_concepts = answer.replace('.', ',').split(',')
                key_concepts = [c.strip() for c in key_concepts if c.strip()]
                
                # Format as bullet points of key ideas
                concepts_list = [f"- Concept: {concept}" for concept in key_concepts if len(concept) > 5]
                if concepts_list:
                    key_points = "\n".join(concepts_list)
                else:
                    key_points = "- Concept: " + answer[:min(50, len(answer))]
            else:
                # For longer answers, use a summary approach
                # Take alternate sentences or first and last, depending on length
                if len(sentences) <= 4:
                    selected_sentences = [sentences[0]] + [sentences[-1]]
                else:
                    # Take first, one from middle, and last sentence
                    selected_sentences = [sentences[0], sentences[len(sentences) // 2], sentences[-1]]
                
                # Create bullet points with key facts, not complete sentences
                key_points = ""
                for sentence in selected_sentences:
                    # Extract key phrases from the sentence
                    words = sentence.split()
                    if len(words) > 8:
                        # For longer sentences, take phrases rather than whole sentence
                        chunks = [' '.join(words[i:i + 4]) for i in range(0, len(words), 4)]
                        chunks_added = False
                        for chunk in chunks:
                            if len(chunk) > 10:  # Only meaningful chunks
                                key_points += f"- Key fact: {chunk}\n"
                                chunks_added = True
                        if not chunks_added and chunks:
                            # If no chunks were long enough, use the first one anyway
                            key_points += f"- Key fact: {chunks[0]}\n"
                    else:
                        # For shorter sentences, use core concept
                        key_points += f"- Main point: {' '.join(words[:min(5, len(words))])}\n"
            
            # Ensure we have some output
            if not key_points.strip():
                key_points = "- Key point: " + answer[:min(50, len(answer))]
                
            return key_points
        except Exception as e:
            self.logger.error(f"Error extracting key concepts: {str(e)}")
            return "- Error extracting key concepts: " + str(e)
        
    def enhance_prompt_with_expected_answer(self, base_prompt: str, expected_answer: str) -> str:
        """
        Enhance a system prompt with guidance from an expected answer.
        
        Args:
            base_prompt (str): The original system prompt
            expected_answer (str): The expected answer to use for guidance
            
        Returns:
            str: Enhanced system prompt with key concepts from the expected answer
        """
        # Validate inputs
        if base_prompt is None:
            self.logger.warning("enhance_prompt_with_expected_answer received None base_prompt")
            base_prompt = "You are a helpful assistant specialized in marketing attribution."
        
        if expected_answer is None:
            self.logger.warning("enhance_prompt_with_expected_answer received None expected_answer")
            return base_prompt
            
        try:
            key_points = self.extract_key_concepts(expected_answer)
            
            enhanced_prompt = f"""{base_prompt}
                IMPORTANT - RESPONSE GUIDANCE:
                I'm providing you with key concepts about this topic. Your task is to:

                1. STRICTLY AVOID any phrasing that matches the reference material
                2. Use only the concepts and facts to inform your response
                3. Write a COMPLETELY ORIGINAL answer in your own words and structure
                4. Include source links from the retrieved context, not these concepts
                5. If these concepts contradict retrieved information, prioritize information from retrieval

                KEY CONCEPTS FROM REFERENCE:
                {key_points}

                CRITICAL INSTRUCTION: Your response must NOT contain any exact phrases from the reference concepts. Rephrase everything completely while preserving the meaning.
                """
                
            return enhanced_prompt
        except Exception as e:
            self.logger.error(f"Error enhancing prompt with expected answer: {str(e)}")
            # Return the base prompt if there was an error
            return base_prompt
    
    def extract_keywords_from_qa(self) -> Dict[str, List[str]]:
        """
        Extract keywords from Q&A pairs to use for tagging content.

        Returns:
            Dictionary mapping keywords to related terms
        """
        # Define key topic areas and related terms
        keyword_map = {
            "causal attribution": ["attribution", "base attribution", "advanced attribution",
                            "self-attribution", "self-attributed", "attribution multiplier",
                            "advanced attribution multiplier", "causal attribution",
                             "multi-touch attribution", "mta", "multi touch", "touchpoints"],
            "incrementality": ["incrementality", "incrementality testing", "geo testing",
                               "holdout test", "scale test", "lift"],
            "market_mix_modeling": ["marketing mix modeling", "mmm", "marketing mix model",
                                    "media mix"],
            "measurement": ["measurement", "metrics", "kpi", "measure",
                            "statistical significance", "minimum detectable lift",
                            "mdl", "confidence"],
            "marketing_funnel": ["funnel", "awareness", "consideration", "conversion",
                                 "retention", "advocacy"],
            "tracking": ["tracking", "cookies", "first-party", "third-party", "pixels"],
            "optimization": ["optimization", "budget allocation", "diminishing returns",
                             "roas", "roi", "cpa", "cac", "icac"],
            "channels": ["facebook", "google", "tiktok", "search", "social", "display"]
        }

        # Extract all questions/answers as a list to build a frequency map
        q_a = list(self.qa_data.items())

        # Count frequency of keywords in questions
        keyword_frequency = {}
        for q, a in q_a:
            q_a_item = q + " " + a
            for category, terms in keyword_map.items():
                for term in terms:
                    if term.lower() in q_a_item.lower():
                        if category not in keyword_frequency:
                            keyword_frequency[category] = 0
                        keyword_frequency[category] += 1

        # Sort by frequency for each category
        sorted_keywords = {k: v for k, v in sorted(
            keyword_frequency.items(), key=lambda item: item[1], reverse=True)}

        self.logger.info(f"Extracted keywords with frequencies: {sorted_keywords}")
        return keyword_map
    
    def create_embedding_prompt(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create an optimized prompt for embedding that highlights attribution terms.

        Args:
            text: Original text to embed
            metadata: Metadata associated with the text

        Returns:
            Enhanced prompt for embedding
        """
        metadata = metadata or {}

        # Add definitions for any technical terms found in the text
        term_definitions = []
        for term, definition in self.technical_definitions.items():
            if term.lower() in text.lower():
                term_definitions.append(f"{term}: {definition}")

        if term_definitions:
            term_context = "\n".join(term_definitions)
            return f"Technical marketing terms context:\n{term_context}\n\n{text}"

        # For tracking-specific Q&A
        if 'special_type' in metadata and metadata['special_type'] == 'tracking_types_examples':
            return f"""
            Context: Web and app tracking methods categorized as first-party and third-party tracking. 
            First-party tracking uses first-party cookies and internal systems.
            Third-party tracking uses third-party cookies and external platforms.

            {text}
            """

        # For attribution-specific texts, add context
        is_attribution_related = any(term in text.lower() for term in [
            "attribution", "incrementality", "MDL", "MMM", "MTA", "CAC", "last click",
            "self-attribution", "self-attributed", "base attribution",
            "advanced attribution", "advanced attribution multiplier"
        ])

        if is_attribution_related:
            return f"Marketing attribution context: {text}"

        return text
    
    def enrich_attribution_metadata(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for attribution terminology and create enhanced metadata.

        Args:
            content: Markdown or text content to analyze

        Returns:
            Dictionary of attribution-related metadata
        """
        # Key attribution terms to identify
        attribution_terms = [
            "attribution", "incrementality", "MDL", "Minimum Detectable Lift",
            "MMM", "marketing mix modeling", "MTA", "multi-touch attribution",
            "CAC", "iCAC", "multiplier", "last click", "geo testing", "holdout test",
            "scale test", "self-attribution", "self-attributed", "base attribution",
            "advanced attribution", "advanced attribution multiplier"
        ]

        metadata: Dict[str, Any] = {"attribution_terms": []}

        content_lower = content.lower()

        # Check for attribution terms
        for term in attribution_terms:
            if term.lower() in content_lower:
                metadata[f"has_{term.replace(' ', '_').lower()}"] = True
                metadata["attribution_terms"].append(term.lower())

        return metadata
    
    def prepare_qa_pairs(self) -> List[Dict[str, Any]]:
        """
        Process Q&A content to preserve question-answer relationships

        Returns:
            List of processed Q&A records
        """
        qa_records = []
        # Split into question-answer pairs
        q_a = list(self.qa_data.items())

        for question, answer in q_a:
            question = question.strip()
            answer = answer.strip()
            self.logger.info(f"Processing Q&A pair: {question} - {answer}")

            if "tracking" in question.lower() and "web and app" in question.lower():
                # Add special metadata for tracking questions
                record = {
                    'title': f"Q&A: {question[:50]}...",
                    'url': '#tracking-types',
                    'markdown': f"Q: {question}\n\nA: {answer}",
                    'type': 'qa_pair',
                    'special_type': 'tracking_types_examples'
                }
                qa_records.append(record)
            else:
                record = {
                    'title': f"Q&A: {question[:50]}...",
                    'url': '#qa',
                    'markdown': f"Q: {question}\n\nA: {answer}",
                    'type': 'qa_pair'
                }
                qa_records.append(record)

        return qa_records
    
    def enhance_records_with_keywords(self, records: List[Dict[str, Any]],
                                   keyword_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Enhance records with keywords based on content analysis.

        Args:
            records: List of content records
            keyword_map: Dictionary of keywords and related terms

        Returns:
            Enhanced records with keywords
        """
        enhanced_records = []

        for record in records:
            # Skip if no markdown content
            if 'markdown' not in record:
                enhanced_records.append(record)
                continue

            content = record['markdown'].lower()
            record_keywords = set()

            # Check for each keyword category in the content
            for category, terms in keyword_map.items():
                for term in terms:
                    if term.lower() in content:
                        record_keywords.add(category)
                        # Add the specific term that matched
                        record_keywords.add(term.lower())

            # Add keywords to record
            if record_keywords:
                record['keywords'] = list(record_keywords)

            enhanced_records.append(record)

        self.logger.info(f"Enhanced {len(enhanced_records)} records with keywords")
        return enhanced_records
    
    def analyze_image_with_llm(self, image_content: bytes, prompt: str, model: str = "gpt-4o") -> Optional[str]:
        """
        Analyze image content using Vision API.
        
        Args:
            image_content: Raw image data
            prompt: The prompt to use for analysis
            model: The model to use
            
        Returns:
            Markdown string with the analysis result
        """
        try:
            # Use the centralized LLMClientManager for image analysis
            return LLMClientManager.analyze_image(
                image_content=image_content,
                prompt=prompt,
                model=model,
                max_tokens=getattr(self.config, 'VISION_MAX_TOKENS', 4000),
                temperature=0
            )
        except Exception as e:
            self.logger.error(f"Error analyzing image with LLM: {str(e)}")
            return None
    
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
        
    async def try_alternative_queries(
        self, 
        original_query: str, 
        process_function: callable, 
        is_adequate_function: callable = None,
        history: List = None,
        max_attempts: int = 3
    ) -> tuple:
        """
        Try alternative query formulations using a processing function.
        
        Args:
            original_query: The original user query
            process_function: Async function that processes each query (e.g., sends to an agent)
            is_adequate_function: Function that determines if a response is adequate
            history: Optional conversation history
            max_attempts: Maximum number of queries to try
            
        Returns:
            Tuple of (best_response, queries_tried)
        """
        self.logger.debug(f"Trying alternative queries for: {original_query}")
        
        # Get alternative queries
        enhanced_query_data = self.enhance_query(original_query, history)
        alt_queries = enhanced_query_data.get("alt_queries", [original_query])
        
        all_responses = []
        queries_tried = []
        
        # Special handling for technical terms
        for term in self.technical_definitions:
            if term.lower() in original_query.lower():
                # For technical terms, try the exact term query first
                technical_query = term
                self.logger.debug(f"Technical term detected, trying exact term: {technical_query}")
                
                technical_response = await process_function(technical_query)
                
                # Use the provided function or default to our own method
                adequacy_check = is_adequate_function or self._is_empty_or_inadequate_response
                
                # If this gives a good result, use it
                if not adequacy_check(technical_response):
                    self.logger.debug(f"Exact technical term query produced good results")
                    return technical_response, [original_query, technical_query]
                else:
                    self.logger.debug(f"Technical term query produced inadequate results, will try alternatives")
        
        # Try the original query first
        original_query = alt_queries[0]
        queries_tried.append(original_query)
        
        self.logger.debug(f"Trying original query: {original_query}")
        original_response = await process_function(original_query)
        all_responses.append((original_response, original_query))
        
        # Use the provided function or default to our own method
        adequacy_check = is_adequate_function or self._is_empty_or_inadequate_response
        
        # If the first response seems good, return it
        if not adequacy_check(original_response):
            self.logger.debug(f"Original query produced good results, no need for alternatives")
            return original_response, queries_tried
        
        # Otherwise, try alternative queries
        self.logger.debug(f"Original query produced inadequate results, trying alternatives")
        
        # Remove the original query as we've already tried it
        alt_queries = alt_queries[1:]
        
        for i, alt_query in enumerate(alt_queries):
            if i >= max_attempts - 1:  # We already tried the original query
                break
                
            queries_tried.append(alt_query)
            self.logger.debug(f"Trying alternative query {i + 1}: {alt_query}")
            
            alt_response = await process_function(alt_query)
            all_responses.append((alt_response, alt_query))
            
            # If this alternative query worked well, return it
            if not adequacy_check(alt_response):
                self.logger.debug(f"Alternative query {i + 1} produced good results")
                return alt_response, queries_tried
            else:
                self.logger.debug(f"Alternative query {i + 1} produced inadequate results")
        
        # If we get here, none of the queries produced good results
        # Return the original response as a fallback
        self.logger.debug(f"No alternative queries produced better results, returning original")
        
        return original_response, queries_tried

    def condense_content_using_llm(self, content: str, max_chars: int = 800) -> str:
        """
        Summarize content using the OpenAI API.
        
        Args:
            content: The content to summarize
            max_chars: Maximum characters for the summary
            
        Returns:
            Summarized content
        """
        try:
            # Use the centralized LLM client
            llm = LLMClientManager.get_chat_llm(
                model=self.config.OPENAI_SUMMARY_MODEL,
                temperature=0
            )
            
            prompt = f"<markdown>\n{content}\n</markdown>\nYou should shorten the above markdown text to MAXIMUM OF {max_chars} characters while making sure ALL THE HEADINGS AND HYPERLINKS are retained so that the users can refer to those links later. In your response, don't include <markdown> tags."
            
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"Error condensing content: {str(e)}")
            return content[:max_chars] + "..." if len(content) > max_chars else content


# Create a singleton instance
enhancement_service = EnhancementService()