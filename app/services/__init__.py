"""
Services module for the ChatBot application.
"""
# Import from chat
from app.services.chat import ChatService, ChatEvaluationService, ChatCacheService, AgentService

# Import from indexing
from app.services.indexing import IndexService, ContentProcessor
from app.services.indexing.providers import GoogleDriveIndexer, ShopifyIndexer

# Import from output
from app.services.output import DocumentService
from app.services.output.generators import DocsService, SlidesService

# Import from common
from app.services.common import EnhancementService
from app.services.common.background_jobs import (
    start_background_job,
    get_job_status,
    get_all_jobs,
    update_job_progress,
    lifespan
)

__all__ = [
    # Chat services
    "ChatService",
    "AgentService",
    "ChatEvaluationService",
    "ChatCacheService",
    
    # Indexing services
    "IndexService",
    "ContentProcessor",
    "GoogleDriveIndexer",
    "ShopifyIndexer",
    
    # Output services
    "DocumentService",
    "DocsService",
    "SlidesService",
    
    # Common services
    "EnhancementService",
    
    # Background job functions
    "start_background_job",
    "get_job_status",
    "get_all_jobs",
    "update_job_progress",
    "lifespan"
]