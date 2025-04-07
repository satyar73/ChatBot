"""
FastAPI routes for indexing operations.
"""
from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from app.models.index_models import (
    IndexRequest, 
    IndexOperationResponse, 
    IndexStatusResponse
)
from app.services.indexing.index_service import IndexService

# Initialize router
router = APIRouter(prefix="/index", tags=["index"])

# Dependency to get the index service
def get_index_service():
    """Dependency to get an IndexService instance."""
    return IndexService()

# Unified indexing approach
@router.post("/", response_model=IndexOperationResponse, status_code=202,
             description="Create and populate a vector index from the specified source")
async def create_index(
    request: IndexRequest,
    index_service: IndexService = Depends(get_index_service)
):
    """
    Create and populate a vector index from the specified source.
    
    ## Source Types
    - **shopify**: Index content from Shopify store
       - Options:
         - `store`: Shopify store domain (e.g., "mystore.myshopify.com")
         - `summarize`: Whether to summarize content (boolean)
    
    - **google_drive**: Index content from Google Drive
       - Options:
         - `folder_id`: Google Drive folder ID (string)
         - `recursive`: Whether to recursively process subfolders (boolean, defaults to true)
         - `enhanced_slides`: Whether to use enhanced slide processing with GPT-4 Vision (boolean)
         - `summarize`: Whether to summarize content (boolean)
    
    ## Namespace
    The optional namespace parameter lets you organize your indices by client, project,
    or any other category. This makes it possible to have separate indices for different
    use cases and filter content when retrieving or deleting.
    
    ## Examples
    
    ### Indexing Shopify Content
    ```json
    {
        "source": "shopify",
        "namespace": "client1",
        "options": {
            "store": "client1.myshopify.com"
        }
    }
    ```
    
    ### Indexing Google Drive Content
    ```json
    {
        "source": "google_drive",
        "namespace": "marketing_docs",
        "options": {
            "folder_id": "1AbCdEfGhIjKlMnOpQrStUv",
            "recursive": true,
            "enhanced_slides": true
        }
    }
    ```
    
    ## Response
    The response includes the operation status, a message describing the result,
    and details about the operation such as document count, namespace, and processing time.
    """
    try:
        if request.source == "shopify":
            # Extract Shopify-specific options
            store = request.options.get("store")
            summarize = request.options.get("summarize")
            
            # Call the existing Shopify indexing method
            result = await index_service.create_shopify_index(
                store=store, 
                summarize=summarize,
                namespace=request.namespace
            )
            
        elif request.source == "google_drive":
            # Extract Google Drive-specific options
            folder_id = request.options.get("folder_id")
            recursive = request.options.get("recursive", True)
            summarize = request.options.get("summarize")
            enhanced_slides = request.options.get("enhanced_slides")
            
            # Call the existing Google Drive indexing method
            result = await index_service.create_gdrive_index(
                folder_id=folder_id,
                recursive=recursive,
                summarize=summarize,
                enhanced_slides=enhanced_slides,
                namespace=request.namespace
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {request.source}")
        
        return JSONResponse(
            content=result,
            status_code=200 if result.get("status") == "success" else 500
        )
        
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@router.get("/", response_model=IndexStatusResponse,
           description="Get information about the vector index")
async def get_index_info(
    source: Optional[str] = Query(
        None, 
        description="Filter by source type ('shopify' or 'google_drive')",
        example="shopify"
    ),
    namespace: Optional[str] = Query(
        None,
        description="Filter by namespace",
        example="client1"
    ),
    index_service: IndexService = Depends(get_index_service)
):
    """
    Get information about the current vector index.
    
    ## Query Parameters
    - **source**: Optional filter by source type ('shopify' or 'google_drive')
    - **namespace**: Optional filter by namespace
    
    ## Response
    The response includes:
    - Statistics about the index (total vectors, document count)
    - List of available namespaces
    - Content information by source type
    
    ## Examples
    
    ### Get All Index Information
    `GET /api/index/`
    
    ### Get Shopify Index Only
    `GET /api/index/?source=shopify`
    
    ### Get Specific Namespace
    `GET /api/index/?namespace=client1`
    
    ### Get Specific Source in Specific Namespace
    `GET /api/index/?source=shopify&namespace=client1`
    """
    try:
        result = await index_service.get_index_info(source=source, namespace=namespace)
        return JSONResponse(
            content=result,
            status_code=200 if result.get("status") == "success" else 500
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@router.delete("/", response_model=IndexOperationResponse,
              description="Delete the vector index")
async def delete_index(
    source: Optional[str] = Query(
        None, 
        description="Filter deletion by source type ('shopify' or 'google_drive')",
        example="shopify"
    ),
    namespace: Optional[str] = Query(
        None,
        description="Filter deletion by namespace",
        example="client1"
    ),
    index_service: IndexService = Depends(get_index_service)
):
    """
    Delete the vector index.
    
    ## Query Parameters
    - **source**: Optional filter deletion by source type ('shopify' or 'google_drive')
    - **namespace**: Optional filter deletion by namespace
    
    The parameters let you control the scope of deletion:
    
    - If source and namespace are both specified, only delete content from that source in that namespace
    - If only source is specified, delete all content from that source (across all namespaces)
    - If only namespace is specified, delete all content in that namespace (from all sources)
    - If neither is specified, delete all indices
    
    ## Examples
    
    ### Delete All Indices
    `DELETE /api/index/`
    
    ### Delete All Shopify Content
    `DELETE /api/index/?source=shopify`
    
    ### Delete Specific Namespace
    `DELETE /api/index/?namespace=client1`
    
    ### Delete Specific Source in Specific Namespace
    `DELETE /api/index/?source=shopify&namespace=client1`
    
    ## Response
    The response includes a status, message describing what was deleted, and details about the deletion operation.
    """
    try:
        result = await index_service.delete_index(source=source, namespace=namespace)
        return JSONResponse(
            content=result,
            status_code=200 if result.get("status") == "success" else 500
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@router.get("/files", 
           description="Get list of indexed files from a specific source")
async def get_indexed_files(
    source: str = Query(
        ..., 
        description="Source type ('shopify' or 'google_drive')",
        example="google_drive"
    ),
    namespace: Optional[str] = Query(
        None,
        description="Filter by namespace",
        example="marketing_docs"
    ),
    index_service: IndexService = Depends(get_index_service)
):
    """
    Get list of indexed files from the specified source.
    
    ## Query Parameters
    - **source**: Required source type ('shopify' or 'google_drive')
    - **namespace**: Optional filter by namespace
    
    ## Response Format
    
    ### For Google Drive
    ```json
    {
        "status": "success",
        "files": [
            {
                "id": "file_id",
                "name": "Document Title.pdf",
                "mimeType": "application/pdf",
                "lastModified": "2023-05-15T14:30:00Z"
            }
        ]
    }
    ```
    
    ### For Shopify
    ```json
    {
        "status": "success",
        "content": [
            {
                "id": "item_id",
                "title": "Product Name",
                "type": "product",
                "url": "https://store.shopify.com/products/product-name"
            }
        ]
    }
    ```
    
    ## Examples
    
    ### Get All Google Drive Files
    `GET /api/index/files?source=google_drive`
    
    ### Get Google Drive Files in Specific Namespace
    `GET /api/index/files?source=google_drive&namespace=marketing_docs`
    """
    try:
        if source == "google_drive":
            # Pass the namespace parameter to filter files
            result = await index_service.get_google_drive_files(namespace=namespace)
        elif source == "shopify":
            # Placeholder for Shopify content listing
            result = {"status": "success", "message": "Shopify content listing not implemented yet"}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {source}")
            
        return JSONResponse(
            content=result,
            status_code=200 if result.get("status") == "success" else 500
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )