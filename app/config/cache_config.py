"""
Configuration for the chat response caching system.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# Create necessary directories
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Cache database settings
CACHE_DB_PATH = CACHE_DIR / "chat_cache.db"

# Cache behavior settings
CACHE_ENABLED = True  # Can be toggled to quickly disable caching
CACHE_TTL = 86400  # Time to live for cache entries in seconds (default: 24 hours)
CACHE_SIZE_LIMIT = 10000  # Maximum number of entries in the cache

# Cache cleanup settings
DEFAULT_CACHE_CLEANUP_DAYS = 30  # Default age in days for automatic cleanup

# Logging settings
CACHE_LOG_ENABLED = True
CACHE_LOG_PATH = CACHE_DIR / "cache_stats.log"

# Query hashing settings
CONSIDER_SESSION_IN_HASH = False  # Whether to include session ID in query hash
MAX_HISTORY_FOR_HASH = 3  # How many previous exchanges to consider for context