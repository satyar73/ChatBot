#!/usr/bin/env python3
"""
Convenience script to run the ChatBot application.
Run this from any directory with `python /path/to/run.py`
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to sys.path 
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import config after setting up path
from app.config.logging_config import DEVELOPMENT_LOG_LEVELS, PRODUCTION_LOG_LEVELS

if __name__ == "__main__":
    # Get the environment and log level
    env = os.environ.get("ENVIRONMENT", "development").lower()
    
    if env == "production":
        log_config = PRODUCTION_LOG_LEVELS
    else:
        log_config = DEVELOPMENT_LOG_LEVELS
    
    log_level = log_config.get("uvicorn", "info").lower()
    
    print(f"Starting ChatBot server in {env} environment")
    print(f"Access the API at http://localhost:8005")
    print(f"Press Ctrl+C to stop the server")
    
    # Start the Uvicorn server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8005,
        log_level=log_level,
        reload=(env != "production")
    )