"""
FastAPI routes for chat and testing functionality.
"""
from fastapi import (
    APIRouter,
    HTTPException,
    Response,
    Depends,
    Query,
    UploadFile,
    File,
    Form,
    Path,
)
from typing import Optional, List
from typing import Dict, Any
from datetime import datetime

from app.models.session_models import SessionListResponse, SessionMetadata, SessionFilterOptions
import os
import tempfile

from app.models.chat_models import Message, ResponseMessage
from app.utils.logging_utils import get_logger

# Create logger
logger = get_logger(__name__)
from app.models.chat_test_models import (
    ChatTestRequest,
    ChatTestResponse,
    ChatBatchTestResponse,
)
from app.services.chat.chat_service import ChatService
from app.services.chat.chat_evaluation_service import ChatEvaluationService
from app.services.chat.chat_cache_service import chat_cache

# Initialize router
router = APIRouter(prefix="/chat", tags=["chat"])  # Using single primary tag

# Dependency to get the chat service
def get_chat_service():
    """Dependency to get a ChatService instance."""
    return ChatService()

# Dependency to get the test service
def get_test_service():
    """Dependency to get a TestService instance."""
    # You could load configuration from environment variables here
    chatbot_api_url = os.getenv("CHATBOT_API_URL", "http://localhost:8005")
    return ChatEvaluationService(chatbot_api_url)


# Chat routes
@router.post("/", response_model=ResponseMessage)
async def chat(data: Message, chat_service: ChatService = Depends(get_chat_service)) -> ResponseMessage:
    """
    Process a chat message and return an AI response with optional sources.
    
    ## Request Parameters
    - **message**: The user's question or message
    - **session_id**: Unique ID for maintaining conversation context
    - **mode**: Response mode
        - `rag` (default): Knowledge-enhanced responses with sources
        - `no_rag`: General knowledge only without sources
    - **system_prompt**: Optional custom prompt to override default behavior
    - **prompt_style**: Response style preference
        - `default`: Standard balanced response
        - `detailed`: Comprehensive, thorough response
        - `concise`: Brief, to-the-point response
    - **metadata**: Additional context parameters
        - `client_name`: Retrieve client-specific information (e.g., "LaserAway")
        - `content_type`: Filter by content type ("article", "blog", "product")
        - `topic`: Topic-specific information ("attribution", "facebook", "geo_testing")
    
    ## Response Fields
    - **response.output**: The AI's answer
    - **response.no_rag_output**: Non-RAG response (if mode is "compare")
    - **sources**: Referenced documents with title, URL, and content excerpt
    """
    return await chat_service.chat(data)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, chat_service: ChatService = Depends(get_chat_service)):
    """
    Delete chat history for a session.

    Args:
        session_id: Session ID to delete, or "ALL_CHATS" to clear all
        chat_service: ChatService instance from dependency

    Returns:
        204 No Content on success, 404 Not Found if session doesn't exist
    """
    if chat_service.delete_chat(session_id):
        return Response(status_code=204)
    raise HTTPException(status_code=404, detail="session_id does not exist")


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, description="Page number (1-indexed)", ge=1),
    page_size: int = Query(20, description="Number of sessions per page", ge=1, le=100),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    List available chat sessions with pagination.
    
    ## Response Format
    - **sessions**: List of session metadata
    - **total_count**: Total number of available sessions
    - **page**: Current page number
    - **page_size**: Number of sessions per page
    """
    # Calculate offset from page and page_size
    offset = (page - 1) * page_size
    
    # Get sessions from service
    result = chat_service.list_sessions(limit=page_size, offset=offset)
    
    # Return formatted result
    return {
        "sessions": result["sessions"],
        "total_count": result["total_count"],
        "page": page,
        "page_size": page_size
    }

@router.get("/sessions/filter", response_model=Dict[str, Any])
async def filter_sessions(
    tags: Optional[List[str]] = Query(None, description="Filter by tags (comma-separated)"),
    mode: Optional[str] = Query(None, description="Filter by mode (rag, no_rag, needl, etc.)"),
    client: Optional[str] = Query(None, description="Filter by client name"),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Filter sessions by various criteria.
    
    ## Filter Parameters
    - **tags**: Optional list of tags to filter by (comma-separated)
    - **mode**: Optional mode to filter by (e.g., 'rag', 'no_rag', 'needl')
    - **client**: Optional client name to filter by
    
    ## Response Format
    - **session_ids**: List of matching session IDs
    - **count**: Number of matching sessions
    - **filters_applied**: Summary of filters that were applied
    """
    matching_sessions = []
    filters_applied = []
    
    # Apply tag filter if provided
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            result = chat_service.find_sessions_by_tag(tag)
            matching_sessions.extend(result["session_ids"])
            filters_applied.append(f"tag:{tag}")
    
    # Apply mode filter if provided
    if mode:
        result = chat_service.find_sessions_by_mode(mode)
        # If we already have results from tags, filter them
        if matching_sessions:
            matching_sessions = [s for s in matching_sessions if s in result["session_ids"]]
        else:
            matching_sessions = result["session_ids"]
        filters_applied.append(f"mode:{mode}")
    
    # Remove duplicates
    matching_sessions = list(set(matching_sessions))
    
    return {
        "session_ids": matching_sessions,
        "count": len(matching_sessions),
        "filters_applied": filters_applied
    }

@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(
    session_id: str = Path(..., description="The session ID to retrieve"),
    mode: Optional[str] = Query("all", description="Filter messages by mode (rag, no_rag, standard, needl, compare, all)"),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Get detailed information about a specific chat session.
    
    ## Path Parameters
    - **session_id**: Unique identifier for the chat session
    
    ## Query Parameters
    - **mode**: Optional mode to filter messages by (rag, no_rag, standard, needl, compare)
    
    ## Response Format
    - **session_id**: The session ID
    - **metadata**: Session metadata including creation time, tags, etc.
    - **messages**: List of messages in the session
    """
    result = chat_service.get_session_by_id(session_id, mode=mode)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
        
    return result



# Test routes
@router.post("/test", response_model=ChatTestResponse)
async def run_test(
        request: ChatTestRequest,
        test_service: ChatEvaluationService = Depends(get_test_service)
):
    """
    Run a test on a prompt/expected result pair through the testing workflow.

    Args:
        request: Test request with prompt and expected response
        test_service: TestService instance from dependency

    Returns:
        Test response with results
    """
    try:
        response = await test_service.run_test(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")


@router.post("/batch-test", response_model=ChatBatchTestResponse)
async def run_batch_test(
        similarity_threshold: float = Query(0.7, description="Default threshold for similarity comparison"),
        csv_file: UploadFile = File(..., description="CSV file with test cases"),
        test_service: ChatEvaluationService = Depends(get_test_service)
):
    """
    Run tests from a CSV file containing prompts and expected results.

    Args:
        similarity_threshold: Default threshold for similarity comparison
        csv_file: Uploaded CSV file with test cases
        test_service: TestService instance from dependency

    Returns:
        Batch test response with results and summary
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_file_path = temp_file.name
            content = await csv_file.read()
            temp_file.write(content)

        # Run batch test
        response = await test_service.run_batch_test(
            csv_file=temp_file_path,
            similarity_threshold=similarity_threshold
        )

        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        return response
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Batch test execution failed: {str(e)}"
        )

# Session management routes
@router.post("/sessions/{session_id}/tags/{tag}")
async def add_tag_to_session(
    session_id: str = Path(..., description="The session ID to tag"),
    tag: str = Path(..., description="The tag to add"),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Add a tag to a specific session.
    
    ## Path Parameters
    - **session_id**: The session ID to tag
    - **tag**: The tag to add
    
    ## Response
    - HTTP 200 OK with success message
    - HTTP 404 Not Found if session doesn't exist
    """
    try:
        # Get the session
        session = chat_service.session_manager.get_session(session_id)
        
        # Add the tag
        session.add_tag(tag)
        
        # Save the session
        chat_service.session_manager.save_session(session)
        
        return {"message": f"Tag '{tag}' added to session '{session_id}'"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error adding tag: {str(e)}")

@router.delete("/sessions/{session_id}/tags/{tag}")
async def remove_tag_from_session(
    session_id: str = Path(..., description="The session ID to modify"),
    tag: str = Path(..., description="The tag to remove"),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Remove a tag from a specific session.
    
    ## Path Parameters
    - **session_id**: The session ID to modify
    - **tag**: The tag to remove
    
    ## Response
    - HTTP 200 OK with success message
    - HTTP 404 Not Found if session or tag doesn't exist
    """
    try:
        # Get the session
        session = chat_service.session_manager.get_session(session_id)
        
        # Remove the tag
        success = session.remove_tag(tag)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Tag '{tag}' not found in session '{session_id}'")
        
        # Save the session
        chat_service.session_manager.save_session(session)
        
        return {"message": f"Tag '{tag}' removed from session '{session_id}'"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error removing tag: {str(e)}")

@router.post("/sessions/backup", response_model=Dict[str, Any])
async def create_backup(
    custom_path: Optional[str] = Query(None, description="Optional custom backup path"),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Create a backup of all sessions.
    
    ## Query Parameters
    - **custom_path**: Optional custom backup path
    
    ## Response
    - HTTP 200 OK with backup path information
    - HTTP 500 Internal Server Error if backup creation fails
    """
    backup_path = chat_service.session_manager.create_backup(custom_path)
    
    if backup_path:
        return {
            "message": "Backup created successfully",
            "backup_path": backup_path,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create backup")

@router.post("/sessions/restore", response_model=Dict[str, Any])
async def restore_backup(
    backup_path: str = Query(..., description="Path to the backup file to restore"),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Restore sessions from a backup.
    
    ## Query Parameters
    - **backup_path**: Path to the backup file to restore
    
    ## Response
    - HTTP 200 OK with restoration information
    - HTTP 400 Bad Request if the backup file doesn't exist
    - HTTP 500 Internal Server Error if restoration fails
    """
    if not os.path.exists(backup_path):
        raise HTTPException(status_code=400, detail=f"Backup file not found: {backup_path}")
    
    session_count = chat_service.session_manager.restore_backup(backup_path)
    
    if session_count > 0:
        return {
            "message": "Backup restored successfully",
            "sessions_restored": session_count,
            "backup_path": backup_path,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to restore backup or no sessions in backup")

# Cache management routes
@router.get("/cache/stats", tags=["cache"])
async def get_cache_stats():
    """
    Get statistics about the cache usage.
    
    Returns:
        Dictionary with cache statistics
    """
    return chat_cache.get_cache_stats()
    
@router.get("/test-slides-connection", tags=["slides"])
async def test_slides_connection():
    """
    Test endpoint to verify connectivity to the slides service.
    
    Returns:
        Simple message confirming connectivity
    """
    return {"status": "ok", "message": "Connection to slides service is working"}

@router.post("/test-slides-upload", tags=["slides"])
async def test_slides_upload(
    file: UploadFile = File(...),
):
    """
    Test endpoint to verify file upload functionality.
    
    Args:
        file: Test file to upload
        
    Returns:
        Information about the uploaded file
    """
    return {
        "status": "ok", 
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(await file.read())
    }


@router.delete("/cache", response_model=Dict[str, Any], tags=["cache"])
async def clear_cache(older_than_days: Optional[int] = None):
    """
    Clear cache entries.
    
    Args:
        older_than_days: Optional, only clear entries older than this many days
        
    Returns:
        Number of entries cleared
    """
    from app.config.cache_config import DEFAULT_CACHE_CLEANUP_DAYS
    
    # If older_than_days is not provided, but we have a default, use it
    if older_than_days is None and DEFAULT_CACHE_CLEANUP_DAYS > 0:
        # Log that we're using the default
        chat_cache.logger.info(f"Using default cache cleanup age: {DEFAULT_CACHE_CLEANUP_DAYS} days")
        entries_cleared = chat_cache.clear_cache(DEFAULT_CACHE_CLEANUP_DAYS)
        return {
            "message": f"Cleared {entries_cleared} cache entries older than {DEFAULT_CACHE_CLEANUP_DAYS} days",
            "older_than_days": DEFAULT_CACHE_CLEANUP_DAYS,
            "entries_cleared": entries_cleared
        }
    
    # Otherwise, clear as specified (or all if older_than_days is None)
    entries_cleared = chat_cache.clear_cache(older_than_days)
    
    if older_than_days is None:
        return {
            "message": f"Cleared all cache entries ({entries_cleared} entries)",
            "entries_cleared": entries_cleared
        }
    else:
        return {
            "message": f"Cleared {entries_cleared} cache entries older than {older_than_days} days",
            "older_than_days": older_than_days,
            "entries_cleared": entries_cleared
        }

@router.post("/create-slides", deprecated=True, tags=["slides"])
async def create_slides(
    file: UploadFile = File(...),
    title: str = Form("Q&A Presentation"),
    owner_email: Optional[str] = Form(None, description="Email address to share the presentation with"),
    author_name: Optional[str] = Form(None, description="Name of the author to display on the title slide")
):
    """
    Legacy endpoint for creating slides (maintained for backward compatibility).
    Please use /create-document endpoint instead.
    
    Creates a Google Slides presentation from a CSV file containing questions and format templates.
    
    Args:
        file: CSV file with questions and format templates
        title: Title for the presentation
        owner_email: Email address to share with
        author_name: Name of the author for the title slide
        
    Returns:
        dict: Contains the presentation ID and URL
    """
    # Forward to the create_document endpoint with slides document_type
    return await create_document(
        file=file,
        title=title,
        owner_email=owner_email,
        author_name=author_name,
        document_type="slides"
    )

@router.post("/create-document", tags=["document"])
async def create_document(
    file: UploadFile = File(...),
    title: str = Form("Generated Document"),
    owner_email: Optional[str] = Form(None, description="Email address to share the document with"),
    author_name: Optional[str] = Form(None, description="Name of the author (for slides title page)"),
    document_type: str = Form("slides", description="Type of document to create: 'slides' or 'docs'")
):
    """
    Create a Google document (Slides or Docs) from a CSV file containing questions and format templates.
    Content will be generated using the RAG system with the specified format templates.
    
    The CSV file should contain two columns:
    - 'question': The question to ask
    - 'format': The format template to use
        For slides: "Title: Overview\nBody: [bullets]"
        For docs: "# Main Title\n## Section\n- Bullet points"
    
    Args:
        file: CSV file with questions and format templates
        title: Title for the document
        owner_email: Email address to share the document with
        author_name: Name of the author (for slides title page)
        document_type: Type of document to create ('slides' or 'docs')
        
    Returns:
        dict: Contains the document ID and URL
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    if document_type not in ["slides", "docs"]:
        raise HTTPException(status_code=400, detail="document_type must be 'slides' or 'docs'")
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        document_id = ""
        document_url = ""
        
        # Create the appropriate document type
        if document_type == "slides":
            # Create slides
            from app.services.output.generators.slides_service import SlidesService
            service = SlidesService()
            document_id = await service.create_presentation_from_csv(
                csv_path=temp_file_path,
                title=title,
                author_name=author_name,
                owner_email=owner_email  # Sharing handled in the service
            )
            document_url = f"https://docs.google.com/presentation/d/{document_id}"
            document_type_name = "presentation"
        else:
            # Create docs
            from app.services.output.generators.docs_service import DocsService
            service = DocsService()
            document_id = await service.create_document_from_csv(
                csv_path=temp_file_path,
                title=title,
                owner_email=owner_email  # Sharing handled in the service
            )
            document_url = f"https://docs.google.com/document/d/{document_id}"
            document_type_name = "document"
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return {
            "document_id": document_id,
            "document_type": document_type_name,
            "url": document_url,
            "shared_with": owner_email if owner_email else None,
            "title": title,
            "author": author_name if author_name and document_type == "slides" else None,
            "note": f"The {document_type_name} is shared with anyone who has the link. If an email was provided, they've been granted editor access."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))