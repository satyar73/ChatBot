"""
Service layer for handling question and answer functionality with enhanced prompt enrichment.
"""
import json
from typing import Dict, List, Any, Optional

from app.config.chat_config import ChatConfig
from app.utils.llm_client import LLMClientManager
from app.utils.logging_utils import get_logger
from app.utils.other_utlis import load_json


class QAService:
    """Service for handling question-answering functionality."""

    def __init__(self):
        """Initialize the QA service."""
        self.config = ChatConfig()
        self.logger = get_logger(__name__, "DEBUG")
        self.logger.debug("QAService initialized")
        
        # Initialize the QA data
        self.qa_data = {}
        self.load_qa_data()

    def load_qa_data(self) -> None:
        """Load QA pairs from the configured JSON file and cache them in memory."""
        if hasattr(self.config, 'QA_SOURCE_FILE_JSON') and self.config.QA_SOURCE_FILE_JSON:
            json_file = self.config.QA_SOURCE_FILE_JSON
            self.qa_data = load_json(json_file)
            self.logger.info(f"Loaded {len(self.qa_data)} QA pairs from {json_file}")

    def get_answer(self, question: str) -> Optional[str]:
        """
        Get the expected answer for a question if it exists in the cache.

        Args:
            question (str): The question to look up

        Returns:
            str or None: The expected answer if found, None otherwise
        """
        # Look for exact match
        if question in self.qa_data:
            return self.qa_data[question]

        # Case-insensitive match
        for q, a in self.qa_data.items():
            if q.lower() == question.lower():
                return a

        # No match found
        return None
        
    def extract_key_concepts(self, answer: str) -> str:
        """
        Extract key concepts from an expected answer to guide response generation
        without revealing the exact answer.
        
        Args:
            answer (str): The full expected answer
            
        Returns:
            str: Formatted key concepts suitable for inclusion in a prompt
        """
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
            key_points = "\n".join([f"- Concept: {concept}" for concept in key_concepts if len(concept) > 5])
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
                    for chunk in chunks:
                        if len(chunk) > 10:  # Only meaningful chunks
                            key_points += f"- Key fact: {chunk}\n"
                else:
                    # For shorter sentences, use core concept
                    key_points += f"- Main point: {' '.join(words[:min(5, len(words))])}\n"
                    
        return key_points
        
    def enhance_prompt_with_expected_answer(self, base_prompt: str, expected_answer: str) -> str:
        """
        Enhance a system prompt with guidance from an expected answer.
        
        Args:
            base_prompt (str): The original system prompt
            expected_answer (str): The expected answer to use for guidance
            
        Returns:
            str: Enhanced system prompt with key concepts from the expected answer
        """
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
        for q_a_item in q_a:
            for category, terms in keyword_map.items():
                for term in terms:
                    if term.lower() in q_a_item:
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

        # Check for specific technical terms to highlight
        technical_terms = {
            "advanced attribution multiplier": "A coefficient used in advanced "
                                               "attribution to adjust the credit given "
                                               "to marketing channels based on their true "
                                               "incremental value",
            "attribution multiplier": "A factor used to adjust attribution models "
                                      "to reflect true marketing contribution",
            # Add other technical terms as needed
        }

        # Add definitions for any technical terms found in the text
        term_definitions = []
        for term, definition in technical_terms.items():
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
qa_service = QAService()