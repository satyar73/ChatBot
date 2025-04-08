from enum import Enum
from typing import Dict

from app.config.llm_proxy_config import LlmProxyConfig
from app.config.vector_store_config import VectorStoreConfig


class CloudProvider(Enum):
    OpenAI = 1 #cloud provider
    Ollama = 2 #e.g. downloaded to the server


class ChatModelConfig:
    """Class to manage list of models and its vector store needs to be used configuration settings for the application"""

    def __init__(self, cloud_provider: CloudProvider, embedding_model: str, vector_store_config: VectorStoreConfig, 
                 cloud_api_key: str = None, llm_proxy_config: LlmProxyConfig = None,
                 embedding_context_length: int = 8192):
        self._cloud_provider = cloud_provider
        self._embedding_model = embedding_model
        self._vector_store_config = vector_store_config
        self._cloud_api_key = cloud_api_key
        self._llm_proxy_config = llm_proxy_config
        self._embedding_context_length = embedding_context_length

    @property
    def cloud_provider(self)->CloudProvider:
        return self._cloud_provider

    @property
    def embedding_model(self)->str:
        return self._embedding_model

    @property
    def vector_store_config(self)->VectorStoreConfig:
        return self._vector_store_config

    @property
    def cloud_api_key(self)->str:
        return self._cloud_api_key

    @property
    def llm_proxy_config(self)->LlmProxyConfig:
        return self._llm_proxy_config
    
    @property
    def embedding_context_length(self)->int:
        return self._embedding_context_length
        
    def get_embedding_dimensions(self)->int:
        """
        Get the dimensions for configured embedding model.
        This is a base implementation that can be overridden by specific vector store configs.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Number of dimensions for the embedding model
        """
        # Common OpenAI embedding model dimensions
        model_dimensions: Dict[str, int] = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # Add more models as needed
        }
        return model_dimensions.get(self.embedding_model, 1536)  # Default to 1536 if unknown