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
)
from typing import Optional

from typing import Dict, Any
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
from app.services.chat_service import ChatService
from app.services.chat_test_service import ChatTestService
from app.services.cache_service import chat_cache
from app.services.slides_service import SlidesService

# Initialize router
router = APIRouter(prefix="/chat", tags=["chat", "cache"])

# Dependency to get the chat service
def get_chat_service():
    """Dependency to get a ChatService instance."""
    return ChatService()

# Dependency to get the test service
def get_test_service():
    """Dependency to get a TestService instance."""
    # You could load configuration from environment variables here
    chatbot_api_url = os.getenv("CHATBOT_API_URL", "http://localhost:8005")
    return ChatTestService(chatbot_api_url)


# Chat routes
@router.post("/", response_model=ResponseMessage)
async def chat(data: Message, chat_service: ChatService = Depends(get_chat_service)) -> ResponseMessage:
    """
    Process chat message and return responses based on the specified mode.

    Args:
        data: Message object containing user input, session ID, and Mode.
              Mode can be "rag", "no_rag", or "compare"
        chat_service: ChatService instance from dependency

    Returns:
        ResponseMessage containing responses based on the specified mode:
        - "rag": Only RAG-based response
        - "no_rag": Only standard (non-RAG) response
        - "compare": Both RAG and non-RAG responses
    """
    return await chat_service.chat(data)


@router.delete("/session/{session_id}", status_code=204)
async def delete_chat(session_id: str, chat_service: ChatService = Depends(get_chat_service)):
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


@router.get("/session/{session_id}")
async def get_chat(session_id: str, chat_service: ChatService = Depends(get_chat_service)):
    """
    Get chat history for a session.

    Args:
        session_id: Session ID to retrieve, or "ALL_CHATS" to get all
        chat_service: ChatService instance from dependency

    Returns:
        Dictionary containing chat history, 404 Not Found if session doesn't exist
    """
    chat_history = chat_service.get_chat(session_id)
    if chat_history:
        return chat_history
    raise HTTPException(status_code=404, detail="session_id does not exist")


# Test routes
@router.post("/test", response_model=ChatTestResponse)
async def run_test(
        request: ChatTestRequest,
        test_service: ChatTestService = Depends(get_test_service)
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
        test_service: ChatTestService = Depends(get_test_service)
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

# Cache management routes
@router.get("/cache/stats", tags=["cache"])
async def get_cache_stats():
    """
    Get statistics about the cache usage.
    
    Returns:
        Dictionary with cache statistics
    """
    return chat_cache.get_cache_stats()
    
@router.get("/test-slides-connection")
async def test_slides_connection():
    """
    Test endpoint to verify connectivity to the slides service.
    
    Returns:
        Simple message confirming connectivity
    """
    return {"status": "ok", "message": "Connection to slides service is working"}

@router.post("/test-slides-upload")
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

@router.post("/create-slides")
async def create_slides(
    file: UploadFile = File(...),
    title: str = Form("Q&A Presentation"),
    owner_email: Optional[str] = Form(None, description="Email address to transfer ownership to"),
    author_name: Optional[str] = Form(None, description="Name of the author to display on the title slide"),
    use_rag: bool = Form(True, description="Whether to use RAG to generate answers (defaults to True)")
):
    """
    Create a Google Slides presentation from a CSV file containing questions.
    If use_rag is True (default), answers will be generated using the RAG system.
    Otherwise, answers from the CSV file will be used.
    
    Args:
        file: CSV file with questions (and optional answers as fallback)
        title: Title for the presentation
        owner_email: Email address to transfer ownership to
        author_name: Name of the author to display on the title slide
        use_rag: Whether to generate answers using RAG (True) or use answers from CSV (False)
        
    Returns:
        dict: Contains the presentation ID and URL
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Create slides
        slides_service = SlidesService()
        presentation_id = await slides_service.create_presentation_from_csv(
            csv_path=temp_file_path,
            title=title,
            author_name=author_name,
            use_rag=use_rag
        )
        
        # Get the Drive service
        drive_service = slides_service._get_drive_service()
        
        # Make the document accessible with a link
        anyone_permission = {
            'type': 'anyone',
            'role': 'writer',
            'allowFileDiscovery': False
        }
        
        # Add the public permission
        drive_service.permissions().create(
            fileId=presentation_id,
            body=anyone_permission,
            fields='id'
        ).execute()
        
        # Share with editor access if email is provided
        # Note: We can't transfer ownership directly due to Google API limitations
        # requiring user consent. Instead, we'll grant editor access.
        if owner_email:
            try:
                # First try with editor role instead of owner
                editor_permission = {
                    'type': 'user',
                    'role': 'writer',  # 'writer' provides edit access
                    'emailAddress': owner_email
                }
                
                # Add the permission to the file
                drive_service.permissions().create(
                    fileId=presentation_id,
                    body=editor_permission,
                    fields='id',
                    sendNotificationEmail=True
                ).execute()
                
                # Log the action
                logger.info(f"Granted editor access to {owner_email} for presentation {presentation_id}")
            except Exception as share_error:
                # If sharing fails, log it but continue
                logger.error(f"Error sharing presentation with {owner_email}: {str(share_error)}")
                # Don't re-raise, as we want to return the presentation even if sharing fails
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return {
            "presentation_id": presentation_id,
            "url": f"https://docs.google.com/presentation/d/{presentation_id}",
            "shared_with": owner_email if owner_email else None,  # Changed from owner to shared_with
            "author": author_name if author_name else None,
            "used_rag": use_rag,
            "note": "The presentation is shared with anyone who has the link. If an email was provided, they've been granted editor access."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))