from enum import Enum
from typing import Dict

class VectorStoreType(Enum):
    PINECONE = 1 #online paid
    NEON = 2 #opensource local

class VectorStoreConfig:
    """
    VectorStoreConfig is a base class to store different type of vector store client (e.g. PINECONE, NEON)
    """
    def __init__(self, vector_store_type: VectorStoreType):
        self._vector_store_type = VectorStoreType.PINECONE
        self._index_name = ""  # Default empty index name

    @property
    def vector_store_type(self)->VectorStoreType:
        return self._vector_store_type

    @property
    def index_name(self) -> str:
        """
        Get the name of the vector store index.
        
        Returns:
            Name of the index
        """
        return self._index_name

    def get_embedding_dimensions(self, model_name: str) -> int:
        """
        Get the dimensions for a specific embedding model.
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
        return model_dimensions.get(model_name, 1536)  # Default to 1536 if unknown

class PineconeConfig(VectorStoreConfig):
    """
    PineconeConfig stores the configuration for PINECONE
    """
    def __init__(self, api_key: str, index_name: str, cloud: str, region: str, namespace: str = "default"):
        super().__init__(VectorStoreType.PINECONE)
        self._api_key = api_key
        self._index_name = index_name
        self._cloud = cloud
        self._region = region
        self._namespace = namespace

    @property
    def api_key(self)->str:
        return self._api_key

    @property
    def index_name(self)->str:
        return self._index_name
        
    @property
    def namespace(self)->str:
        """
        Get the namespace for this Pinecone configuration.
        
        Returns:
            The namespace for the Pinecone index
        """
        return self._namespace

    @property
    def cloud(self)->str:
        return self._cloud

    @property
    def region(self)->str:
        return self._region

    def get_embedding_dimensions(self, model_name: str) -> int:
        """Get the dimensions for a specific embedding model"""
        # Common OpenAI embedding model dimensions
        model_dimensions: Dict[str, int] = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # Add more models as needed
        }
        return model_dimensions.get(model_name, 1536)  # Default to 1536 if unknown


class NeonConfig(VectorStoreConfig):
    """
    PineconeConfig stores the configuration for PINECONE
    """
    def __init__(self):
        super().__init__(VectorStoreType.NEON)
