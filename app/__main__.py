"""
This module allows running the app with `python -m app`
"""

import os
import uvicorn
from app.config.logging_config import DEVELOPMENT_LOG_LEVELS

if __name__ == "__main__":
    # Get the log level from config
    env = os.environ.get("ENVIRONMENT", "development").lower()
    log_level = DEVELOPMENT_LOG_LEVELS.get("uvicorn", "info").lower()
    
    # Start the Uvicorn server
    uvicorn.run(
        "app:app",  # Use the app export from __init__.py
        host="0.0.0.0",
        port=8005,
        log_level=log_level,
        reload=(env != "production")  # Auto-reload in development mode
    )