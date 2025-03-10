"""
ChatBot Example package.
This __init__.py file makes it possible to run the application from anywhere.

Example usages:
    # Run directly
    python -m app

    # Run with uvicorn
    uvicorn app:app
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and expose the application
from app.main import app

__all__ = ['app']