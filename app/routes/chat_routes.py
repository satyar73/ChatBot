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
)

from typing import Optional, Dict, Any
import os
import tempfile

from app.models.chat_models import Message, ResponseMessage
from app.models.chat_test_models import (
    ChatTestRequest,
    ChatTestResponse,
    ChatBatchTestRequest,
    ChatBatchTestResponse,
)
from app.services.chat_service import ChatService
from app.services.chat_test_service import ChatTestService

# Initialize router
router = APIRouter(prefix="/chat", tags=["chat"])

# Dependency to get the chat service
def get_chat_service():
    """Dependency to get a ChatService instance."""
    return ChatService()

# Dependency to get the test service
def get_test_service():
    """Dependency to get a TestService instance."""
    # You could load configuration from environment variables here
    msquared_api_url = os.getenv("MSQUARED_API_URL", "http://localhost:8005")
    return ChatTestService(msquared_api_url)


# Chat routes
@router.post("/", response_model=ResponseMessage)
async def chat(data: Message, chat_service: ChatService = Depends(get_chat_service)) -> ResponseMessage:
    """
    Process chat message and return both RAG and non-RAG responses.

    Args:
        data: Message object containing user input and session ID
        chat_service: ChatService instance from dependency

    Returns:
        ResponseMessage containing both RAG and non-RAG responses with sources
    """
    return await chat_service.chat(data)


@router.delete("/{session_id}", status_code=204)
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


@router.get("/{session_id}")
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