import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat_routes, index_routes
from app.utils.logging_utils import configure_logging, update_logger_levels, get_logger
from app.config.logging_config import (
    SERVICE_LOG_LEVELS,
    DEVELOPMENT_LOG_LEVELS,
    PRODUCTION_LOG_LEVELS
)

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
    title="MSquared AttributionGPT Backend",
    version="1.0",
    description="Backend for AttributionGPT with chat and index management",
)

# Allow all origins (replace "*" with the specific origin of your Angular app if possible)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_routes.router)
app.include_router(index_routes.router)


@app.get("/health")
async def health_endpoint():
    logger.debug("Health endpoint called")
    return {"message": "Up and running!"}


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