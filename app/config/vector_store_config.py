from enum import Enum

class VectorStoreType(Enum):
    PINECONE = 1 #online paid
    NEON = 2 #opensource local

class VectorStoreConfig:
    """
    VectorStoreConfig is a base class to store different type of vector store client (e.g. PINECONE, NEON)
    """
    def __init__(self, vector_store_type: VectorStoreType):
        self._vector_store_type = VectorStoreType.PINECONE

    @property
    def vector_store_type(self)->VectorStoreType:
        return self._vector_store_type

class PineconeConfig(VectorStoreConfig):
    """
    PineconeConfig stores the configuration for PINECONE
    """
    def __init__(self, api_key: str, index_name: str, cloud: str, region: str):
        super().__init__(VectorStoreType.PINECONE)
        self._api_key = api_key
        self._index_name = index_name
        self._cloud = cloud
        self._region = region

    @property
    def api_key(self)->str:
        return self._api_key

    @property
    def index_name(self)->str:
        return self._index_name

    @property
    def cloud(self)->str:
        return self._cloud

    @property
    def region(self)->str:
        return self._region

    def get_embedding_dimensions(self, model_name)->int:
        """Get the dimensions for a specific embedding model"""
        # Common OpenAI embedding model dimensions
        model_dimensions = {
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
