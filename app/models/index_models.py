"""
Data models for the indexing API.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class IndexRequestBase(BaseModel):
    """Base model for index requests."""
    source: str = Field(..., description="Source type: 'shopify' or 'google_drive'")
    namespace: Optional[str] = Field(None, description="Namespace to use for the index")

class ShopifyIndexOptions(BaseModel):
    """Options specific to Shopify indexing."""
    store: Optional[str] = Field(None, description="Shopify store domain")
    summarize: Optional[bool] = Field(None, description="Whether to summarize content")

class GoogleDriveIndexOptions(BaseModel):
    """Options specific to Google Drive indexing."""
    folder_id: Optional[str] = Field(None, description="Google Drive folder ID")
    recursive: Optional[bool] = Field(True, description="Whether to recursively process subfolders")
    summarize: Optional[bool] = Field(None, description="Whether to summarize content")
    enhanced_slides: Optional[bool] = Field(None, description="Whether to use enhanced slide processing")

class IndexRequest(IndexRequestBase):
    """Model for unified indexing requests."""
    options: Dict[str, Any] = Field(default_factory=dict, description="Source-specific options")

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
                        "folder_id": "abc123xyz",
                        "recursive": True,
                        "enhanced_slides": True
                    }
                }
            ]
        }
    }

class IndexSourceStats(BaseModel):
    """Model for source-specific indexing statistics."""
    document_count: int = Field(0, description="Number of documents indexed from this source")
    vector_count: Optional[int] = Field(None, description="Number of vectors in the index for this source")
    last_indexed: Optional[str] = Field(None, description="Timestamp of the last indexing operation")

class IndexStatusResponse(BaseModel):
    """Model for index status response."""
    status: str = Field(..., description="Status of the request: 'success' or 'error'")
    message: Optional[str] = Field(None, description="Status message")
    sources: Dict[str, IndexSourceStats] = Field(default_factory=dict, description="Statistics per source")
    total_vectors: int = Field(0, description="Total number of vectors in the index")
    namespaces: List[str] = Field(default_factory=list, description="List of namespaces in the index")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
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
    """Model for index operation responses."""
    status: str = Field(..., description="Status of the operation: 'success' or 'error'")
    message: str = Field(..., description="Operation result message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details about the operation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "message": "Successfully indexed 120 documents from Shopify in namespace 'mystore'",
                "details": {
                    "document_count": 120,
                    "namespace": "mystore",
                    "processing_time_seconds": 45
                }
            }
        }
    }