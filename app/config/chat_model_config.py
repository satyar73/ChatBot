from enum import Enum

from app.config.llm_proxy_config import LlmProxyConfig
from app.config.vector_store_config import VectorStoreConfig


class CloudProvider(Enum):
    OpenAI = 1 #cloud provider
    Local = 2 #e.g. downloaded to the server


class ChatModelConfig:
    """Class to manage list of models and its vector store needs to be used configuration settings for the application"""

    def __init__(self, cloud_provider: CloudProvider, model: str, vector_store_config: VectorStoreConfig, 
                 cloud_api_key: str = None, llm_proxy_config: LlmProxyConfig = None,
                 embedding_context_length: int = 8192):
        self._cloud_provider = cloud_provider
        self._model = model
        self._vector_store_config = vector_store_config
        self._cloud_api_key = cloud_api_key
        self._llm_proxy_config = llm_proxy_config
        self._embedding_context_length = embedding_context_length

    @property
    def cloud_provider(self)->CloudProvider:
        return self._cloud_provider

    @property
    def model(self)->str:
        return self._model

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
        