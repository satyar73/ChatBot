"""
Background job management for long-running tasks.

This module provides utilities for running tasks in the background
and tracking their status, results, and progress.
"""

import asyncio
import uuid
from typing import Dict, Any, Callable, Awaitable, Optional, List
import logging
import time
from datetime import datetime, timedelta
import traceback
import json
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks

# Set up logging
logger = logging.getLogger(__name__)

# In-memory storage for job status and results
# For a production system, you'd want to use a database or Redis
JOBS: Dict[str, Dict[str, Any]] = {}

# How long to keep completed jobs in memory (seconds)
JOB_RETENTION_SECONDS = 24 * 60 * 60  # 24 hours

# Auto-cleanup for old jobs
async def cleanup_old_jobs():
    """Remove old completed jobs to prevent memory leaks"""
    while True:
        try:
            now = datetime.now()
            jobs_to_remove = []
            
            for job_id, job_data in JOBS.items():
                if job_data["status"] in ["completed", "failed"]:
                    completed_at = job_data.get("completed_at")
                    if completed_at and (now - completed_at).total_seconds() > JOB_RETENTION_SECONDS:
                        jobs_to_remove.append(job_id)
            
            # Remove old jobs
            for job_id in jobs_to_remove:
                del JOBS[job_id]
                logger.info(f"Cleaned up old job {job_id}")
                
            # Sleep for a while before the next cleanup
            await asyncio.sleep(3600)  # Check once per hour
        except Exception as e:
            logger.error(f"Error in job cleanup: {str(e)}")
            await asyncio.sleep(3600)  # Try again in an hour

# Start the cleanup task
cleanup_task = None

@asynccontextmanager
async def lifespan(app):
    """Context manager for FastAPI app startup/shutdown"""
    # Start the cleanup task
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_old_jobs())
    
    yield
    
    # Cancel the cleanup task on shutdown
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get the status of a background job"""
    if job_id not in JOBS:
        return {"status": "not_found", "message": "Job not found"}
    
    # Return a copy to avoid modifications
    job_data = JOBS[job_id].copy()
    
    # Remove internal data that shouldn't be exposed
    job_data.pop("task", None)
    job_data.pop("background_task", None)
    
    return job_data

def get_all_jobs() -> List[Dict[str, Any]]:
    """Get status for all jobs"""
    result = []
    for job_id, job_data in JOBS.items():
        # Create a copy without internal fields
        job_info = job_data.copy()
        job_info.pop("task", None)
        job_info.pop("background_task", None)
        job_info["job_id"] = job_id
        result.append(job_info)
    
    return result

def create_job(job_type: str, description: str = None) -> str:
    """Create a new background job and return its ID"""
    job_id = str(uuid.uuid4())
    
    JOBS[job_id] = {
        "job_id": job_id,
        "type": job_type,
        "description": description or f"{job_type} job",
        "status": "pending",
        "created_at": datetime.now(),
        "progress": 0,
        "message": "Job created, waiting to start",
        "result": None,
        "error": None
    }
    
    return job_id

async def _run_job(job_id: str, task_func: Callable[..., Awaitable[Any]], *args, **kwargs):
    """Internal function to run a job and update its status"""
    if job_id not in JOBS:
        logger.error(f"Job {job_id} not found")
        return
    
    # Update job to running state
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = datetime.now()
    JOBS[job_id]["message"] = "Job is running"
    
    try:
        # Run the task
        start_time = time.time()
        result = await task_func(*args, **kwargs)
        end_time = time.time()
        
        # Update job with success result
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["completed_at"] = datetime.now()
        JOBS[job_id]["duration_seconds"] = end_time - start_time
        JOBS[job_id]["result"] = result
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Job completed successfully"
        
        logger.info(f"Job {job_id} completed successfully in {end_time - start_time:.2f} seconds")
        return result
    
    except Exception as e:
        # Update job with error information
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["completed_at"] = datetime.now()
        JOBS[job_id]["error"] = {
            "message": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        JOBS[job_id]["message"] = f"Job failed: {str(e)}"
        
        logger.error(f"Job {job_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def update_job_progress(job_id: str, progress: int, message: Optional[str] = None):
    """Update a job's progress percentage (0-100) and message"""
    if job_id not in JOBS:
        logger.warning(f"Attempted to update non-existent job {job_id}")
        return False
    
    JOBS[job_id]["progress"] = min(max(0, progress), 100)
    
    if message:
        JOBS[job_id]["message"] = message
        
    return True

def start_background_job(
    background_tasks: BackgroundTasks, 
    job_type: str,
    task_func: Callable[..., Awaitable[Any]], 
    description: Optional[str] = None,
    *args, **kwargs
) -> str:
    """
    Create and start a new background job
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
        job_type: Type of job (for categorization)
        task_func: Async function to run as the job
        description: Optional human-readable description
        *args, **kwargs: Arguments to pass to the task function
        
    Returns:
        job_id: ID of the created job
    """
    job_id = create_job(job_type, description)
    
    # Store the task and start it
    background_tasks.add_task(_run_job, job_id, task_func, *args, **kwargs)
    
    # Also store a reference to the background_tasks to prevent it from being garbage collected
    JOBS[job_id]["background_task"] = background_tasks
    
    logger.info(f"Started background job {job_id} of type {job_type}")
    return job_id