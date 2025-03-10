#!/usr/bin/env python3
"""
Local runner script for when you're inside the app directory.
Usage: python run_local.py
"""

import uvicorn

if __name__ == "__main__":
    print("Starting ChatBot API server...")
    print("API will be available at http://127.0.0.1:8005")
    
    uvicorn.run(
        "main:app",  # Using the local main.py file
        host="127.0.0.1",
        port=8005,
        reload=True
    )