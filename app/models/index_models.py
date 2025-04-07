"""
Data models for the indexing API.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class IndexRequestBase(BaseModel):
    """Base model for index requests."""
    source: str = Field(
        ..., 
        description="Source type to index content from", 
        examples=["shopify", "google_drive"]
    )
    namespace: Optional[str] = Field(
        None, 
        description="Optional namespace to organize indexed content (e.g., client name, project)"
    )

class ShopifyIndexOptions(BaseModel):
    """Options specific to Shopify indexing."""
    store: Optional[str] = Field(
        None,
        description="Shopify store domain (e.g., 'mystore.myshopify.com')",
        examples=["mystore.myshopify.com"]
    )
    summarize: Optional[bool] = Field(
        None,
        description="Whether to summarize content using LLM for improved retrieval",
        examples=[False]
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "store": "mystore.myshopify.com",
                "summarize": False
            }
        }
    }

class GoogleDriveIndexOptions(BaseModel):
    """Options specific to Google Drive indexing."""
    folder_id: Optional[str] = Field(
        None, 
        description="Google Drive folder ID (leave empty to use root folder)",
        examples=["1AbCdEfGhIjKlMnOpQrStUv"]
    )
    recursive: Optional[bool] = Field(
        True, 
        description="Whether to recursively process subfolders and their contents",
        examples=[True]
    )
    summarize: Optional[bool] = Field(
        None, 
        description="Whether to summarize content using LLM for improved retrieval",
        examples=[False]
    )
    enhanced_slides: Optional[bool] = Field(
        None, 
        description="Whether to use enhanced slide processing with GPT-4 Vision for better slide understanding",
        examples=[True]
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "folder_id": "1AbCdEfGhIjKlMnOpQrStUv",
                "recursive": True,
                "enhanced_slides": True,
                "summarize": False
            }
        }
    }

class IndexRequest(IndexRequestBase):
    """
    Model for unified indexing requests. 
    
    This model is used for creating and populating indices from various content sources.
    The unified approach allows consistent handling of different source types through
    a standardized interface.
    """
    options: Dict[str, Any] = Field(
        default_factory=dict, 
        description="""
        Source-specific options for indexing. The available options depend on the source type:
        
        - For 'shopify': store, summarize
        - For 'google_drive': folder_id, recursive, enhanced_slides, summarize
        
        See the ShopifyIndexOptions and GoogleDriveIndexOptions models for details.
        """
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "shopify",
                    "namespace": "mystore",
                    "options": {
                        "store": "mystore.myshopify.com",
                        "summarize": False
                    }
                },
                {
                    "source": "google_drive",
                    "namespace": "client_docs",
                    "options": {
                        "folder_id": "1AbCdEfGhIjKlMnOpQrStUv",
                        "recursive": True,
                        "enhanced_slides": True
                    }
                }
            ]
        }
    }

class IndexSourceStats(BaseModel):
    """
    Model for source-specific indexing statistics.
    
    Provides detailed information about the content indexed from a specific source,
    including document counts, vector counts, and when the indexing was last performed.
    """
    document_count: int = Field(
        0, 
        description="Number of documents indexed from this source",
        examples=[120]
    )
    vector_count: Optional[int] = Field(
        None, 
        description="Number of vectors in the index for this source (each document may create multiple vectors)",
        examples=[450]
    )
    last_indexed: Optional[str] = Field(
        None, 
        description="ISO 8601 timestamp of when the source was last indexed",
        examples=["2023-05-15T14:30:00Z"]
    )

class IndexStatusResponse(BaseModel):
    """
    Model for index status response.
    
    Provides a comprehensive overview of the current state of the vector index,
    including statistics for each source, total vectors, and available namespaces.
    This response is returned when querying the status of indices.
    """
    status: str = Field(
        ..., 
        description="Status of the request: 'success' or 'error'",
        examples=["success"]
    )
    message: Optional[str] = Field(
        None, 
        description="Status message or error details if status is 'error'",
        examples=["Index information retrieved successfully"]
    )
    sources: Dict[str, IndexSourceStats] = Field(
        default_factory=dict, 
        description="Statistics per source type, with each key representing a source ('shopify', 'google_drive', etc.)"
    )
    total_vectors: int = Field(
        0, 
        description="Total number of vectors across all sources and namespaces in the index",
        examples=[650]
    )
    namespaces: List[str] = Field(
        default_factory=list, 
        description="List of all namespaces present in the index",
        examples=["default", "client1", "client2"]
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "message": "Index information retrieved successfully",
                "sources": {
                    "shopify": {
                        "document_count": 120,
                        "vector_count": 450,
                        "last_indexed": "2023-05-15T14:30:00Z"
                    },
                    "google_drive": {
                        "document_count": 45,
                        "vector_count": 200,
                        "last_indexed": "2023-05-16T09:15:00Z"
                    }
                },
                "total_vectors": 650,
                "namespaces": ["default", "client1", "client2"]
            }
        }
    }

class IndexOperationResponse(BaseModel):
    """
    Model for index operation responses.
    
    Used for responses to create, update, and delete operations on indices.
    Contains the operation status, a descriptive message, and optional details
    about the operation results.
    """
    status: str = Field(
        ..., 
        description="Status of the operation: 'success' or 'error'",
        examples=["success"]
    )
    message: str = Field(
        ..., 
        description="Human-readable description of the operation result",
        examples=["Successfully indexed 120 documents from Shopify in namespace 'mystore'"]
    )
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="""
        Additional details about the operation, which may include:
        - document_count: Number of documents processed
        - namespace: The namespace used for the operation
        - processing_time_seconds: Time taken to complete the operation
        - source: The source type that was processed
        """
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "message": "Successfully indexed 120 documents from Shopify in namespace 'mystore'",
                    "details": {
                        "document_count": 120,
                        "chunk_count": 350,
                        "namespace": "mystore",
                        "source": "shopify",
                        "processing_time_seconds": 45
                    }
                },
                {
                    "status": "success",
                    "message": "Successfully deleted all indices for source 'google_drive'",
                    "details": {
                        "indices_deleted": 1,
                        "vectors_removed": 200
                    }
                },
                {
                    "status": "error",
                    "message": "Failed to index from Google Drive: Permission denied",
                    "details": {
                        "error_type": "permission_error",
                        "folder_id": "1AbCdEfGhIjKlMnOpQrStUv"
                    }
                }
            ]
        }
    }