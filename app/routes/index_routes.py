"""
FastAPI routes for indexing operations.
"""
from fastapi import APIRouter, Query, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any

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
    - **google_drive**: Index content from Google Drive
    
    ## Common Parameters
    - **source**: The content source type ('shopify' or 'google_drive')
    - **namespace**: Optional namespace to use for this index
    
    ## Shopify-specific Options
    - **store**: Optional Shopify store domain
    - **summarize**: Whether to summarize content
    
    ## Google Drive-specific Options
    - **folder_id**: Optional Google Drive folder ID
    - **recursive**: Whether to recursively process subfolders
    - **summarize**: Whether to summarize content
    - **enhanced_slides**: Whether to use enhanced slide processing
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

@router.get("/", description="Get information about the current vector index")
async def get_index_info(
    source: Optional[str] = Query(None, description="Filter results by source type ('shopify' or 'google_drive')"),
    index_service: IndexService = Depends(get_index_service)
):
    """
    Get information about the current vector index.
    
    Optionally filter results by source type.
    """
    try:
        result = await index_service.get_index_info(source=source)
        return JSONResponse(
            content=result,
            status_code=200 if result.get("status") == "success" else 500
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@router.delete("/", description="Delete the vector index")
async def delete_index(
    source: Optional[str] = Query(None, description="Source type to delete ('shopify' or 'google_drive')"),
    namespace: Optional[str] = Query(None, description="Namespace to delete"),
    index_service: IndexService = Depends(get_index_service)
):
    """
    Delete the vector index.
    
    - If source is specified, only delete content from that source
    - If namespace is specified, only delete content in that namespace
    - If both are specified, delete content from that source in that namespace
    - If neither is specified, delete all indices
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

@router.get("/files", description="Get list of indexed files")
async def get_indexed_files(
    source: str = Query(..., description="Source type ('shopify' or 'google_drive')"),
    index_service: IndexService = Depends(get_index_service)
):
    """
    Get list of indexed files from the specified source.
    """
    try:
        if source == "google_drive":
            result = await index_service.get_google_drive_files()
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