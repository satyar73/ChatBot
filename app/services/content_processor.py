"""
Content processor service for document processing and Pinecone indexing.
"""
import time
import os
from typing import List, Dict, Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from app.config.chat_config import ChatConfig
from app.services.enhancement_service import enhancement_service
from app.utils.logging_utils import get_logger


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
        self.config = config or ChatConfig()
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

    def index_to_pinecone(self, records: List[Dict[str, Any]]) -> bool:
        """
        Index content records to Pinecone vector database.
        
        Args:
            records: List of enhanced content records with title, url, and markdown
            
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            # If no records, return success
            if not records:
                self.logger.warning("No records to index")
                return True

            self.logger.info(f"Indexing {len(records)} records to Pinecone index "
                             f"'{self.config.PINECONE_INDEX_NAME}'")

            # Initialize Pinecone
            pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

            # Check if index exists
            existing_indexes = pc.list_indexes().names()

            # Create index if it doesn't exist
            if self.config.PINECONE_INDEX_NAME not in existing_indexes:
                self.logger.info(f"Creating new Pinecone index: {self.config.PINECONE_INDEX_NAME}")

                pc.create_index(
                    name=self.config.PINECONE_INDEX_NAME,
                    dimension=self.config.PINECONE_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.config.PINECONE_CLOUD,
                        region=self.config.PINECONE_REGION
                    )
                )

                # Wait for index to initialize
                self.logger.info("Waiting for index to initialize...")
                time.sleep(10)
            else:
                self.logger.info(f"Using existing Pinecone index: {self.config.PINECONE_INDEX_NAME}")

            # Prepare documents
            docs = []
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

                    # Create a custom separator pattern that preserves these terms
                    separators = ["\n\n", "\n", ". ", " ", ""]

                    # For technical content, use smaller chunks with more overlap
                    if any(term in record['markdown'].lower() for term in special_terms):
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=self.config.CHUNK_SIZE // 2,  # Smaller chunks for technical content
                            chunk_overlap=self.config.CHUNK_OVERLAP * 2,  # More overlap
                            separators=separators
                        )
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=self.config.CHUNK_SIZE,
                            chunk_overlap=self.config.CHUNK_OVERLAP,
                            separators=separators
                        )
                    chunks = text_splitter.split_text(record['markdown'])

                # Create documents with metadata
                for j, chunk in enumerate(chunks):
                    # Get attribution metadata
                    attribution_metadata = self.enhancement_service.enrich_attribution_metadata(chunk)

                    # Merge with standard metadata
                    metadata = {
                        "title": record['title'],
                        "url": record['url'],
                        "chunk": j,
                        "source": f"{record.get('type', 'content')}"
                    }
                    metadata.update(attribution_metadata)

                    # Add keywords if available
                    if 'keywords' in record:
                        metadata["keywords"] = record['keywords']

                    # Create embedding prompt
                    optimized_text = self.enhancement_service.create_embedding_prompt(chunk, metadata)

                    doc = Document(
                        page_content=optimized_text,
                        metadata=metadata
                    )
                    docs.append(doc)

            # Initialize embeddings
            embeddings = OpenAIEmbeddings(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.OPENAI_EMBEDDING_MODEL,
                embedding_ctx_length=self.config.EMBEDDING_CONTEXT_LENGTH,
                show_progress_bar=True
            )

            # Index documents
            self.logger.info(f"Indexing {len(docs)} document chunks to Pinecone...")

            # Store in Pinecone
            vectorstore = PineconeVectorStore.from_documents(
                docs,
                index_name=self.config.PINECONE_INDEX_NAME,
                pinecone_api_key=self.config.PINECONE_API_KEY,
                embedding=embeddings
            )

            self.logger.info(
                f"Successfully indexed {len(docs)} document chunks to "
                f"Pinecone index '{self.config.PINECONE_INDEX_NAME}'.")
            return True

        except Exception as e:
            self.logger.error(f"Error indexing to Pinecone: {str(e)}")
            return False