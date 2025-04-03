"""
FastAPI routes for testing functionality with support for long-running operations.
Also includes basic service health and diagnostic endpoints.
"""
from fastapi import (
    APIRouter,
    HTTPException,
    BackgroundTasks,
    Depends,
    Query,
    UploadFile,
    File,
)

# Import needed modules
import pandas as pd
import time
import asyncio

from typing import Dict, Any, List
import os
import tempfile

from app.models.chat_test_models import (
    ChatTestRequest,
    ChatTestResponse
)
from app.services.chat_test_service import ChatTestService
from app.services.background_jobs import (
    start_background_job,
    get_job_status,
    get_all_jobs,
    update_job_progress
)

# Initialize router
router = APIRouter(prefix="/test", tags=["testing", "diagnostics"])

# Dependency to get the test service
def get_test_service():
    """Dependency to get a TestService instance."""
    # You could load configuration from environment variables here
    chatbot_api_url = os.getenv("CHATBOT_API_URL", "http://localhost:8005")
    return ChatTestService(chatbot_api_url)


# Regular test route - for single, quick tests
@router.post("/single", response_model=ChatTestResponse)
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


# Asynchronous batch test route with background processing
@router.post("/batch/start", response_model=Dict[str, Any])
async def start_batch_test(
        background_tasks: BackgroundTasks,
        similarity_threshold: float = Query(0.7, description="Default threshold for similarity comparison"),
        csv_file: UploadFile = File(..., description="CSV file with test cases"),
        test_service: ChatTestService = Depends(get_test_service)
):
    """
    Start a batch test job in the background and return a job ID.

    Args:
        background_tasks: FastAPI BackgroundTasks
        similarity_threshold: Default threshold for similarity comparison
        csv_file: Uploaded CSV file with test cases
        test_service: TestService instance from dependency

    Returns:
        Job ID and status information
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_file_path = temp_file.name
            content = await csv_file.read()
            temp_file.write(content)

        # Define the background task function
        async def run_batch_test_job(file_path: str, threshold: float):
            try:

                # Setup progress tracking variables
                start_time = time.time()
                last_progress_time = start_time
                current_progress = 0
                
                # Setup progress update timer
                async def progress_timer():
                    nonlocal current_progress, last_progress_time
                    while current_progress < 90:
                        # If no progress update in 5 seconds, increment by 1%
                        current_time = time.time()
                        if current_time - last_progress_time > 5 and current_progress < 90:
                            current_progress += 1
                            last_progress_time = current_time
                            update_job_progress(job_id, current_progress, 
                                f"Still working... ({current_progress}% complete)")
                        await asyncio.sleep(3)
                
                # Start the timer in the background
                timer_task = asyncio.create_task(progress_timer())
                
                # Update job progress - reading file
                current_progress = 5
                last_progress_time = time.time()
                update_job_progress(job_id, current_progress, "Reading test file")
                
                # Read the CSV file to get total count
                df = pd.read_csv(file_path)
                total_tests = len(df)
                
                # Update job progress - preparing tests
                current_progress = 10
                last_progress_time = time.time()
                update_job_progress(job_id, current_progress, f"Starting batch test with {total_tests} test cases")
                
                # Create a wrapper function to track progress during batch test execution
                original_run_test = test_service.run_test
                test_count = 0
                
                async def run_test_with_progress(request):
                    nonlocal test_count, current_progress, last_progress_time
                    result = await original_run_test(request)
                    test_count += 1
                    
                    # Calculate progress percentage (10-90% range for tests)
                    # Ensure minimum 10% progress and always show some movement
                    progress = max(10, 10 + int(80 * (test_count / total_tests)))
                    
                    # Always ensure progress increases from the current value
                    progress = max(progress, current_progress + 1)
                    
                    # Cap at 90% (the remaining 10% is for finalizing)
                    progress = min(progress, 90)
                    
                    # Update our tracking variables
                    current_progress = progress
                    last_progress_time = time.time()
                    
                    # Get the prompt content from the request (truncate if too long)
                    prompt_text = request.prompt[:50] + "..." if len(request.prompt) > 50 else request.prompt
                    
                    # Create a more informative status message with the test content
                    status_message = (
                        f"Test {test_count}/{total_tests} ({progress}% complete)\n"
                        f"Current test: \"{prompt_text}\""
                    )
                    
                    # Add console output for debugging
                    print(f"Updating progress: {progress}% - Test {test_count}/{total_tests}")
                    print(f"Running test prompt: {prompt_text}")
                    
                    # Update job progress with the enhanced message
                    update_job_progress(job_id, progress, status_message)
                    
                    return result
                
                # Temporarily replace the run_test method with our progress tracking version
                test_service.run_test = run_test_with_progress
                
                # Run batch test
                response = await test_service.run_batch_test(
                    csv_file=file_path,
                    similarity_threshold=threshold
                )
                
                # Restore original method
                test_service.run_test = original_run_test
                
                # Update job progress - finalizing
                update_job_progress(job_id, 95, "Finalizing results")
                
                # Cancel the progress timer task
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass
                
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.unlink(file_path)
                
                # Get file paths for status message
                result_files = []
                if hasattr(response, 'output_file') and response.output_file:
                    result_files.append(f"Results saved to: {response.output_file}")
                if hasattr(response, 'rag_report_file') and response.rag_report_file:
                    result_files.append(f"RAG comparison report saved to: {response.rag_report_file}")
                
                # Include file paths in final status message
                status_message = f"Completed {total_tests} tests: {response.passed} passed, {response.failed} failed"
                
                # Add more details about storage location for debugging
                if 'results_dir' in locals():
                    status_message += f"\n\nResults stored in: {results_dir}"

                if result_files:
                    # Add each file path on a new line for better UI display
                    status_message += f"\n\n" + "\n".join(result_files)
                
                # Final update - make sure to set 100% to indicate completion
                update_job_progress(job_id, 100, status_message)
                
                return response.dict()
            except Exception as e:
                # Cancel the timer task if it existsZ
                if 'timer_task' in locals():
                    # pylint: disable=used-before-assignment
                    timer_task.cancel()
                    try:
                        await timer_task
                    except (asyncio.CancelledError, Exception):
                        pass
                
                # Clean up temporary file in case of error
                if os.path.exists(file_path):
                    os.unlink(file_path)
                
                # Log the error
                print(f"Error in batch test: {str(e)}")
                raise e

        # Start the background job
        job_id = start_background_job(
            background_tasks,
            "batch_test",
            run_batch_test_job,
            f"Batch test from {csv_file.filename} with threshold {similarity_threshold}",
            temp_file_path,
            similarity_threshold
        )

        # Return the job ID
        return {
            "job_id": job_id,
            "message": "Batch test job started",
            "status": "pending"
        }
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch test: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_test_job(job_id: str):
    """
    Get status and results for a specific test job.

    Args:
        job_id: The ID of the job to check

    Returns:
        Job status and results if available
    """
    job_status = get_job_status(job_id)
    
    if job_status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status


@router.get("/jobs", response_model=List[Dict[str, Any]])
async def list_test_jobs():
    """
    List all test jobs.

    Returns:
        List of all test jobs and their status
    """
    return get_all_jobs()


# Diagnostic endpoints
@router.get("/status", tags=["diagnostics"])
async def test_backend_status():
    """
    Check if the backend service is running correctly.
    
    Returns:
        Basic status confirmation message
    """
    import platform
    import sys
    
    return {
        "status": "ok", 
        "message": "Backend is up and running",
        "version": {
            "python": sys.version,
            "platform": platform.platform(),
        }
    }

@router.post("/upload", tags=["diagnostics"])
async def test_file_upload(
    file: UploadFile = File(...),
):
    """
    Test endpoint to verify file upload functionality.
    
    Args:
        file: Test file to upload
        
    Returns:
        Information about the uploaded file
    """
    file_content = await file.read()
    file_size = len(file_content)
    
    # Read the first few bytes of the file content for type verification
    content_preview = file_content[:100].decode('utf-8', errors='replace') if file_size > 0 else ""
    
    return {
        "status": "ok", 
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_size,
        "content_preview": content_preview if len(content_preview) < 100 else content_preview[:100] + "..."
    }