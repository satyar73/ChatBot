import os
import sys
from pathlib import Path
from fastapi import APIRouter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path if needed to enable both running from app dir or project root
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
print(f"Project root path: {project_root}")
print(f"Current sys.path: {sys.path}")

# Set default import paths
import_base = "app"
routes_path = f"{import_base}.routes"

# Try to determine if we're running from the app directory or project root
if os.path.exists(os.path.join(os.getcwd(), "routes")) and os.getcwd().endswith("app"):
    # We're running from the app directory
    import_base = ""
    routes_path = "routes"
    print("Running from app directory, using relative imports")
else:
    print("Running from project root, using absolute imports")

# Import with the determined path
if import_base:
    from app.routes import chat_routes, index_routes, test_routes
    from app.utils.logging_utils import configure_logging, update_logger_levels, get_logger
    from app.config.logging_config import (
        SERVICE_LOG_LEVELS,
        DEVELOPMENT_LOG_LEVELS,
        PRODUCTION_LOG_LEVELS
    )
    from app.services.common.background_jobs import lifespan
else:
    from routes import chat_routes, index_routes, test_routes
    from utils.logging_utils import configure_logging, update_logger_levels, get_logger
    from config.logging_config import (
        SERVICE_LOG_LEVELS,
        DEVELOPMENT_LOG_LEVELS,
        PRODUCTION_LOG_LEVELS
    )
    from app.services.common.background_jobs import lifespan

# Initialize logging first
configure_logging()

# Get a logger for the main module - do this right after configuring
logger = get_logger("app.main")

# Determine environment and set appropriate log levels
env = os.environ.get("ENVIRONMENT", "development").lower()
if env == "production":
    update_logger_levels(PRODUCTION_LOG_LEVELS)
    log_config = PRODUCTION_LOG_LEVELS
else:
    update_logger_levels(DEVELOPMENT_LOG_LEVELS)
    log_config = DEVELOPMENT_LOG_LEVELS

# Apply service-specific log levels (these override environment settings)
update_logger_levels(SERVICE_LOG_LEVELS)

# Log environment information right after configuring everything
logger.info(f"Starting AttributionGPT Backend in {env} environment")
logger.info(f"Logging configured with environment: {env}")

# Explicitly make sure this logger is at INFO level or lower
logger.setLevel("INFO")

# Create FastAPI app
app = FastAPI(
    title="AttributionGPT Backend",
    version="1.0",
    description="""
    Backend for AttributionGPT with chat, indexing, and document generation capabilities.
    
    ## Key Endpoints:
    - `/api/chat/`: Send questions and receive AI responses with sources
    - `/api/index/`: Manage content indexing (Google Drive & Shopify)
    - `/api/create-document`: Generate Google Docs or Slides from chat responses
    
    For detailed usage examples, refer to the schema sections below.
    """,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers. All routers under the API prefix
api_router = APIRouter(prefix="/api")

api_router.include_router(chat_routes.router)
api_router.include_router(index_routes.router)
api_router.include_router(test_routes.router)

app.include_router(api_router)

@app.get("/health")
async def health_endpoint():
    logger.debug("Health endpoint called")
    return {"message": "Up and running!"}
    
@app.get("/debug/check-session-response-types")
async def check_session_response_types():
    """Debug endpoint to check response type integrity across all sessions."""
    logger.info("Running session response type integrity check...")
    from app.services.chat.session_adapter import session_adapter
    
    try:
        # Run the check - will output detailed logs
        session_adapter.check_session_response_types()
        return {"status": "Success", "message": "Session response type integrity check complete. See logs for details."}
    except Exception as e:
        logger.error(f"Error during session response type check: {e}")
        return {"status": "Error", "message": str(e)}

@app.get("/debug/check-session/{session_id}")
async def check_specific_session(session_id: str):
    """Debug endpoint to check a specific session's message types."""
    logger.info(f"Checking session {session_id}...")
    from app.services.chat.session_service import session_manager
    
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        
        # Summarize message info
        message_info = []
        for msg in session.messages:
            if msg.get("role") == "assistant":
                msg_info = {
                    "id": msg.get("id", "unknown"),
                    "role": msg.get("role", "unknown"),
                    "response_type": msg.get("response_type", "missing"),
                    "originalResponseType": msg.get("originalResponseType", "missing"),
                    "has_additional_data": "additional_data" in msg
                }
                
                # Check additional_data if available
                if "additional_data" in msg and isinstance(msg["additional_data"], dict):
                    add_data = msg["additional_data"]
                    if "originalResponseType" in add_data:
                        msg_info["additional_data_originalResponseType"] = add_data["originalResponseType"]
                    if "originalMode" in add_data:
                        msg_info["originalMode"] = add_data["originalMode"]
                        
                message_info.append(msg_info)
                
        return {
            "session_id": session_id,
            "message_count": len(session.messages),
            "assistant_messages": message_info,
            "metadata": session.metadata
        }
    except Exception as e:
        logger.error(f"Error checking session {session_id}: {e}")
        return {"status": "Error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn

    # Log again for good measure
    logger.info(f"Starting Uvicorn server on port 8005")

    # Start the Uvicorn server
    uvicorn.run(
        "app.main:app",  # Make sure this matches your actual module path
        host="0.0.0.0",
        port=8005,
        log_level=log_config.get("uvicorn", "info").lower(),
        reload=(env != "production")  # Auto-reload in development mode
    )