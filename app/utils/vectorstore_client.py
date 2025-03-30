
import json
import os
import time
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from app.config.chat_config import chat_config
from app.config.chat_model_config import ChatModelConfig
from app.config.vector_store_config import NeonConfig, PineconeConfig, VectorStoreConfig, VectorStoreType
from app.services.enhancement_service import enhancement_service
from app.utils.logging_utils import get_logger


class VectorStoreClient():
    def __init__(self):
        self.logger = get_logger(__name__, "DEBUG")
        self.config = chat_config
        self.enhancement_service = enhancement_service

    def index_to_vector_store(self, chat_model_config: ChatModelConfig, records: List[Dict[str, Any]]) -> bool:
        pass

    def get_index_info(self) -> Dict:
        pass

    def delete_index(self) -> Dict:
        pass

    def get_vector_count(self) -> int:
        pass


class PineconeClient(VectorStoreClient):
    def __init__(self, pinecone_config: PineconeConfig):
        super().__init__()
        self._pinecone_config: PineconeConfig = pinecone_config

    def index_to_vector_store(self, chat_model_config: ChatModelConfig, records: List[Dict[str, Any]]) -> bool:
        """
        Index content records to Pinecone vector database.
        
        Args:
            records: List of enhanced content records with title, url, and markdown
            
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            pinecone_config: PineconeConfig = chat_model_config.vector_store_config
            # If no records, return success
            if not records:
                self.logger.warning("No records to index")
                return True

            self.logger.info(f"Indexing {len(records)} records to Pinecone index "
                             f"'{pinecone_config.index_name}'")

            # Initialize Pinecone
            pc = Pinecone(api_key=pinecone_config.api_key)

            # Check if index exists
            existing_indexes = pc.list_indexes().names()

            # Create index if it doesn't exist
            if pinecone_config.index_name not in existing_indexes:
                self.logger.info(f"Creating new Pinecone index: {pinecone_config.index_name}")

                pc.create_index(
                    name=pinecone_config.index_name,
                    dimension=pinecone_config.get_embedding_dimensions(chat_model_config.model),
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=pinecone_config.cloud,
                        region=pinecone_config.region
                    )
                )

                # Wait for index to initialize
                self.logger.info("Waiting for index to initialize...")
                time.sleep(10)
            else:
                self.logger.info(f"Using existing Pinecone index: {pinecone_config.index_name}")

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
                api_key=chat_model_config.cloud_api_key,
                model=chat_model_config.model,
                embedding_ctx_length=chat_model_config.embedding_context_length,
                show_progress_bar=True
            )

            # Index documents
            self.logger.info(f"Indexing {len(docs)} document chunks to Pinecone...")

            # Store in Pinecone
            vectorstore = PineconeVectorStore.from_documents(
                docs,
                index_name=pinecone_config.index_name,
                pinecone_api_key=pinecone_config.api_key,
                embedding=embeddings
            )

            self.logger.info(
                f"Successfully indexed {len(docs)} document chunks to "
                f"Pinecone index '{pinecone_config.index_name}'.")
            return True

        except Exception as e:
            self.logger.error(f"Error indexing to Pinecone", e, exc_info=True)
            return False

    def get_index_info(self) -> Dict:
        """Get information about the current vector index from pinecone"""
        pc = Pinecone(api_key=self._pinecone_config.api_key)

        # Check if index exists
        available_indexes = pc.list_indexes().names()

        if self._pinecone_config.index_name in available_indexes:
            # Get index stats
            index = pc.Index(self._pinecone_config.index_name)
            stats_raw = index.describe_index_stats()

            # Convert any non-serializable objects to strings or simple types
            stats = {
                "dimension": stats_raw.dimension,
                "index_fullness": stats_raw.index_fullness,
                "namespaces": {k: {"vector_count": v.vector_count} for k, v in stats_raw.namespaces.items()},
                "total_vector_count": stats_raw.total_vector_count
            }
            
            # Try to load the Shopify content from saved files
            content = []
            try:
                # Check if product and article files exist
                product_path = os.path.join(self.config.OUTPUT_DIR, "products.json")
                article_path = os.path.join(self.config.OUTPUT_DIR, "articles.json")
                
                if os.path.exists(product_path):
                    with open(product_path, "r") as f:
                        products = json.load(f)
                        content.extend(products)
                
                if os.path.exists(article_path):
                    with open(article_path, "r") as f:
                        articles = json.load(f)
                        content.extend(articles)
            except Exception as e:
                self.logger.warning(f"Could not load content files: {str(e)}")
            
            return {
                "status": "success",
                "exists": True,
                "name": self._pinecone_config.index_name,
                "stats": stats,
                "content": content
            }
        else:
            return {
                "status": "success",
                "exists": False,
                "name": self._pinecone_config.index_name,
                "content": []
            }

    def delete_index(self) -> Dict:
        pc = Pinecone(api_key=self._pinecone_config.api_key)

        # Check if index exists
        available_indexes = pc.list_indexes().names()

        if self._pinecone_config.index_name in available_indexes:
            # Delete the index
            pc.delete_index(self._pinecone_config.index_name)
            return {"status": "success", "message": f"Index '{self._pinecone_config.index_name}' deleted successfully"}
        else:
            return {"status": "success", "message": f"Index '{self._pinecone_config.index_name}' does not exist"}

    def get_vector_count(self) -> int:
        pc = Pinecone(api_key=self._pinecone_config.api_key)
        
        # Check if index exists
        available_indexes = pc.list_indexes().names()

        if self._pinecone_config.index_name in available_indexes:
            index = pc.Index(self._pinecone_config.index_name)
            stats = index.describe_index_stats()
            return stats.total_vector_count
        else:
            return 0


class NeonClient(VectorStoreClient):
    def __init__(self, neon_config: NeonConfig):
        super().__init__()
        self._neon_config: NeonConfig = neon_config

    def index_to_vector_store(self, chat_model_config: ChatModelConfig, records: List[Dict[str, Any]]) -> bool:
        #TODO implement later
        pass

    def get_index_info(self) -> Dict:
        """Get information about the current vector index from pinecone"""
        # TODO yet to implement
        
        return {
                "status": "success",
                "exists": False,
                "name": "xxx",
                "content": []
            }

    def delete_index(self) -> Dict:
        #TODO implement later
        pass

    def get_vector_count(self) -> int:
        #TODO implement later
        return 0


def get_vector_store_client(vector_store_config: VectorStoreConfig):
    vector_store_client = None
    if vector_store_config.vector_store_type == VectorStoreType.PINECONE:
        vector_store_client = PineconeClient(vector_store_config)
    elif vector_store_config.vector_store_type == VectorStoreType.NEON:
        vector_store_client = NeonClient(vector_store_config)
    return vector_store_client

