import json
import os
import time
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

from app.config.chat_config import chat_config
from app.config.chat_model_config import ChatModelConfig
from app.config.vector_store_config import NeonConfig, PineconeConfig, VectorStoreConfig
from app.services.common.enhancement_service import enhancement_service
from app.utils.logging_utils import get_logger


class VectorStoreClient:
    """
    VectorStoreClient is an interface class for performing CURL operations on various vector stores such as Pinecone, Neon, etc
    """
    def __init__(self):
        self.logger = get_logger(__name__, "DEBUG")
        self.config = chat_config
        self.enhancement_service = enhancement_service

    def index_to_vector_store(self, chat_model_config: ChatModelConfig, docs: List[Document]) -> bool:
        pass

    def get_index_info(self) -> Dict:
        pass

    def delete_index(self) -> Dict:
        pass

    def get_vector_count(self) -> int:
        pass

    @staticmethod
    def get_vector_store_client(vector_store_config: VectorStoreConfig) \
                                         -> Optional['VectorStoreClient']:
        """
        Factory method to get the appropriate vector store client.
        
        Args:
            vector_store_config: Configuration for the vector store
            
        Returns:
            Appropriate vector store client instance or None if type is not supported
        """
        if isinstance(vector_store_config, PineconeConfig):
            return PineconeClient(vector_store_config)
        elif isinstance(vector_store_config, NeonConfig):
            return NeonClient(vector_store_config)
        return None


class PineconeClient(VectorStoreClient):
    def __init__(self, pinecone_config: PineconeConfig):
        super().__init__()
        self._pinecone_config: PineconeConfig = pinecone_config

    def _clean_metadata_for_pinecone(self, docs: List[Document]) -> List[Document]:
        """
        Clean document metadata to make it compatible with Pinecone requirements.
        Removes null values and ensures all values are strings, numbers, booleans or lists of strings.
        
        Args:
            docs: List of Document objects
            
        Returns:
            Cleaned list of Document objects
        """
        cleaned_docs = []
        
        for doc in docs:
            # Create a new clean metadata dict
            clean_metadata = {}
            
            # Only keep non-null values
            for key, value in doc.metadata.items():
                if value is not None:
                    # Ensure value is a supported type
                    if isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        clean_metadata[key] = value
                    else:
                        # Convert other types to string
                        clean_metadata[key] = str(value)
            
            # Create a new document with the cleaned metadata
            cleaned_doc = Document(page_content=doc.page_content, metadata=clean_metadata)
            cleaned_docs.append(cleaned_doc)
            
        self.logger.debug(f"Cleaned metadata for {len(cleaned_docs)} documents to comply with Pinecone requirements")
        return cleaned_docs
        
    def index_to_vector_store(self, 
                              chat_model_config: ChatModelConfig, 
                              docs: List[Document]) -> bool:
        """
        Index documents to Pinecone vector database.
        
        Args:
            chat_model_config: Configuration for the chat model to use for indexing, including embedding model and dimensionality
            docs: List of Document objects ready for indexing
            
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            # If no documents, return success
            if not docs:
                self.logger.warning("No documents to index")
                return True

            # Clean document metadata to comply with Pinecone requirements
            cleaned_docs = self._clean_metadata_for_pinecone(docs)
            self.logger.debug(f"Cleaned {len(cleaned_docs)} documents for Pinecone compatibility")

            namespace_used = self._pinecone_config.namespace
            self.logger.debug(f"Indexing {len(cleaned_docs)} document chunks to Pinecone index "
                             f"'{self._pinecone_config.index_name}' in namespace '{namespace_used}'")
            self.logger.debug(f"NAMESPACE TRACKING (INDEXING): Using namespace '{namespace_used}'")
            
            pinecone_namespace_env = os.getenv("PINECONE_NAMESPACE", "Not set")
            self.logger.debug(f"NAMESPACE TRACKING (ENV): PINECONE_NAMESPACE environment variable is '{pinecone_namespace_env}'")

            # Initialize Pinecone
            pc = Pinecone(api_key=self._pinecone_config.api_key)
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(
                model=chat_model_config.model,
                dimensions=self._pinecone_config.get_embedding_dimensions(chat_model_config.model)
            )

            # Check if index exists
            existing_indexes = pc.list_indexes().names()

            # Create index if it doesn't exist
            if self._pinecone_config.index_name not in existing_indexes:
                self.logger.info(f"Creating new Pinecone index: {self._pinecone_config.index_name}")

                pc.create_index(
                    name=self._pinecone_config.index_name,
                    dimension=self._pinecone_config.get_embedding_dimensions(chat_model_config.model),
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self._pinecone_config.cloud,
                        region=self._pinecone_config.region
                    )
                )

                # Wait for index to initialize
                self.logger.info("Waiting for index to initialize...")
                time.sleep(10)
            else:
                self.logger.info(f"Using existing Pinecone index: {self._pinecone_config.index_name}")

            # Index documents
            self.logger.info(f"Indexing {len(cleaned_docs)} document chunks to Pinecone...")

            # Store in Pinecone with namespace support
            vectorstore = PineconeVectorStore.from_documents(
                cleaned_docs,
                index_name=self._pinecone_config.index_name,
                pinecone_api_key=self._pinecone_config.api_key,
                embedding=embeddings,
                namespace=self._pinecone_config.namespace
            )

            self.logger.info(
                f"Successfully indexed {len(cleaned_docs)} document chunks to "
                f"Pinecone index '{self._pinecone_config.index_name}' in namespace '{self._pinecone_config.namespace}'.")
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


