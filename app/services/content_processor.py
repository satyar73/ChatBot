"""
Content processor service for document processing and Pinecone indexing.
"""
import os
from typing import List, Dict, Any, Optional

from app.config.chat_config import ChatConfig, chat_config
from app.services.enhancement_service import enhancement_service
from app.utils.logging_utils import get_logger
from app.utils.vectorstore_client import VectorStoreClient


class ContentProcessor:
    """
    Base class for processing and indexing document content to Pinecone.
    Provides common indexing functionality for different content sources.
    """

    def __init__(self, config: Optional[ChatConfig] = None):
        """
        Initialize the content processor with configuration.
        
        Args:
            config: Configuration object with Pinecone parameters
        """
        self.config = config or chat_config
        self.logger = get_logger(__name__, "DEBUG")
        self.logger.debug("ContentProcessor initialized")
        self.enhancement_service = enhancement_service
        
        # Create output directory if needed
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and enhance records before indexing.
        
        Args:
            records: List of content records with title, url, and markdown
            
        Returns:
            Enhanced records with additional metadata
        """
        # Extract keywords from QA content
        keyword_map = self.enhancement_service.extract_keywords_from_qa()
        
        # Enhance records with keywords
        enhanced_records = self.enhancement_service.enhance_records_with_keywords(records, keyword_map)
        
        return enhanced_records

    def index_to_vector_store(self, records: List[Dict[str, Any]]) -> bool:
        """
        Go thru each configured vector store (e.g. Pinecone, Neon, etc) and index the documents
        """
        success = True
        for chat_model_config in chat_config.chat_model_configs.values():
            vector_store_config = chat_model_config.vector_store_config
            vector_store_client: VectorStoreClient = VectorStoreClient.get_vector_store_client(vector_store_config)
            success &= vector_store_client.index_to_vector_store(chat_model_config, records)

        return success
