"""
Content processor service for document processing and vector store indexing.
"""
import os
from typing import List, Dict, Any, Optional

from app.utils.logging_utils import get_logger
from app.config.chat_config import ChatConfig, chat_config
from app.services.enhancement_service import enhancement_service
from app.utils.text_splitters import TokenTextSplitter
from app.utils.vectorstore_client import VectorStoreClient
from langchain.schema import Document


class ContentProcessor:
    """
    Base class for processing and indexing document content to vector stores.
    Provides common indexing functionality for different content sources.
    """

    def __init__(self, config: Optional[ChatConfig] = None):
        """
        Initialize the content processor with configuration.
        
        Args:
            config: Configuration object with vector store parameters
        """
        self.config = config or ChatConfig()
        self.logger = get_logger(__name__, "DEBUG")
        self.logger.debug("ContentProcessor initialized")
        self.enhancement_service = enhancement_service
        
        # Create output directory if needed
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current vector index.
        
        Returns:
            Dictionary containing index information
        """
        try:
            # Get index info from vector store
            vector_store = VectorStoreClient()
            index_info = vector_store.get_index_info()
            
            return {
                "total_documents": index_info.get("total_documents", 0),
                "total_chunks": index_info.get("total_chunks", 0),
                "index_name": index_info.get("index_name", ""),
                "last_updated": index_info.get("last_updated", ""),
                "dimension": index_info.get("dimension", 0)
            }
        except Exception as e:
            self.logger.error(f"Error getting index info: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "index_name": "",
                "last_updated": "",
                "dimension": 0,
                "error": str(e)
            }

    def prepare_documents_for_indexing(self, records: List[Dict[str, Any]]) \
                                                -> List[Document]:
        """
        Prepare documents for indexing by splitting content into chunks and adding metadata.
        
        Args:
            records: List of content records with title, url, and markdown
            
        Returns:
            List of Document objects ready for indexing
        """
        docs = []
        # Extract keywords from QA content once for all chunks
        keyword_map = self.enhancement_service.extract_keywords_from_qa()
        
        for i, record in enumerate(records):
            # Check if record has markdown content
            if 'markdown' not in record:
                self.logger.warning(f"Record {i} missing 'markdown' field: {record}")
                continue  # Skip records without markdown

            # Split content into chunks
            if record.get('type') == 'qa_pair':
                # For Q&A content, don't split questions from answers
                chunks = [record['markdown']]
            else:
                # Define special technical terms to preserve
                special_terms = [
                    "advanced attribution multiplier",
                    "attribution multiplier",
                    "marketing mix modeling",
                    # Add other multi-word technical terms
                ]

                # For technical content, use smaller chunks with more overlap
                if any(term in record['markdown'].lower() for term in special_terms):
                    text_splitter = TokenTextSplitter(
                        chunk_size=self.config.CHUNK_SIZE // 2,  # Smaller chunks for technical content
                        chunk_overlap=self.config.CHUNK_OVERLAP * 2,  # More overlap
                        model_name=self.config.OPENAI_EMBEDDING_MODEL
                    )
                else:
                    text_splitter = TokenTextSplitter(
                        chunk_size=self.config.CHUNK_SIZE,
                        chunk_overlap=self.config.CHUNK_OVERLAP,
                        model_name=self.config.OPENAI_EMBEDDING_MODEL
                    )
                chunks = text_splitter.split_text(record['markdown'])

            # Create documents with metadata
            for j, chunk in enumerate(chunks):
                # Get attribution metadata
                attribution_metadata = self.enhancement_service.enrich_attribution_metadata(chunk)
                
                # Enhance chunk with keywords
                chunk_keywords = self.enhancement_service.enhance_chunk_with_keywords(chunk, keyword_map)

                # Merge with standard metadata
                metadata = {
                    "title": record['title'],
                    "url": record['url'],
                    "chunk": j,
                    "source": f"{record.get('type', 'content')}"
                }
                metadata.update(attribution_metadata)
                if record.get('source') == 'Google Drive':
                    metadata["type"] = record.get('type')
                    metadata["client"] = record.get('client')

                # Add chunk-specific keywords
                if chunk_keywords:
                    metadata["keywords"] = chunk_keywords

                # Create embedding prompt
                optimized_text = self.enhancement_service.create_embedding_prompt(chunk, metadata)

                doc = Document(
                    page_content=optimized_text,
                    metadata=metadata
                )
                docs.append(doc)

        return docs

    def index_to_vector_store(self, docs: List[Document]) -> bool:
        """
        Go through each configured vector store (e.g. Pinecone, Neon, etc.) and index the documents
        """
        success = True
        for chat_model_config in chat_config.chat_model_configs.values():
            vector_store_config = chat_model_config.vector_store_config
            vector_store_client: VectorStoreClient = VectorStoreClient.get_vector_store_client(vector_store_config)
            success &= vector_store_client.index_to_vector_store(chat_model_config, docs)

        return success
