"""
FastAPI routes for testing functionality with support for long-running operations.
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

from typing import Optional, Dict, Any, List
import os
import tempfile

from app.models.chat_test_models import (
    ChatTestRequest,
    ChatTestResponse,
    ChatBatchTestResponse,
)
from app.services.chat_test_service import ChatTestService
from app.services.background_jobs import (
    start_background_job,
    get_job_status,
    get_all_jobs,
    update_job_progress
)

# Initialize router
router = APIRouter(prefix="/test", tags=["testing"])

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
                # Count total tests in CSV for progress tracking
                import pandas as pd
                
                # Update job progress - reading file
                update_job_progress(job_id, 5, "Reading test file")
                
                # Read the CSV file to get total count
                df = pd.read_csv(file_path)
                total_tests = len(df)
                
                # Update job progress - preparing tests
                update_job_progress(job_id, 10, f"Starting batch test with {total_tests} test cases")
                
                # Create a wrapper function to track progress during batch test execution
                original_run_test = test_service.run_test
                test_count = 0
                
                async def run_test_with_progress(request):
                    nonlocal test_count
                    result = await original_run_test(request)
                    test_count += 1
                    
                    # Calculate progress percentage (10-90% range for tests)
                    progress = 10 + int(80 * (test_count / total_tests))
                    
                    # Update job progress
                    update_job_progress(
                        job_id, 
                        progress, 
                        f"Running test {test_count}/{total_tests} ({progress}% complete)"
                    )
                    
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
                
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.unlink(file_path)
                
                # Get file paths for status message
                result_files = []
                if hasattr(response, 'output_file') and response.output_file:
                    result_files.append(f"Results saved to: {response.output_file}")
                
                # Include file paths in final status message
                status_message = f"Completed {total_tests} tests: {response.passed} passed, {response.failed} failed"
                if result_files:
                    status_message += f"\n{' '.join(result_files)}"
                
                # Final update
                update_job_progress(job_id, 100, status_message)
                
                return response.dict()
            except Exception as e:
                # Clean up temporary file in case of error
                if os.path.exists(file_path):
                    os.unlink(file_path)
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