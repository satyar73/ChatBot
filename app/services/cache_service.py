"""
Cache service for storing and retrieving chat responses.
"""
import json
import time
import hashlib
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os

from app.config import cache_config
from app.utils.logging_utils import get_logger

class ChatCacheService:
    """Service for caching chat responses to avoid redundant API calls."""

    def __init__(self):
        """Initialize the cache service."""
        self.logger = get_logger(f"{__name__}.ChatCacheService")
        self.logger.info("Initializing ChatCacheService")
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the SQLite database for the cache."""
        try:
            conn = sqlite3.connect(str(cache_config.CACHE_DB_PATH))
            cursor = conn.cursor()
            
            # Create cache table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_cache (
                query_hash TEXT PRIMARY KEY,
                user_input TEXT,
                rag_response TEXT,
                no_rag_response TEXT,
                sources TEXT,
                system_prompt TEXT,
                timestamp REAL,
                hit_count INTEGER DEFAULT 1
            )
            ''')
            
            # Create logs table for statistics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                session_id TEXT,
                user_input TEXT,
                cache_hit BOOLEAN,
                query_hash TEXT,
                response_time REAL
            )
            ''')
            
            # Perform database migrations
            self._run_migrations(conn)
            
            conn.commit()
            conn.close()
            self.logger.info(f"Cache database initialized at {cache_config.CACHE_DB_PATH}")
        except Exception as e:
            self.logger.error(f"Failed to initialize cache database: {e}")
            raise
            
    def _run_migrations(self, conn):
        """Run database migrations to update schema when needed.
        
        Args:
            conn: SQLite connection
        """
        cursor = conn.cursor()
        try:
            # Check if system_prompt column exists in chat_cache table
            cursor.execute("PRAGMA table_info(chat_cache)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add system_prompt column if it doesn't exist
            if "system_prompt" not in columns:
                self.logger.info("Adding system_prompt column to chat_cache table")
                cursor.execute("ALTER TABLE chat_cache ADD COLUMN system_prompt TEXT")
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error during database migration: {e}")
            # Continue execution - don't let migration failure break the application
    
    def generate_query_hash(self, query: str, history: List = None, session_id: str = None, system_prompt: str = None) -> str:
        """
        Generate a hash to uniquely identify a query with its context.
        
        Args:
            query: The user's query text
            history: Optional chat history
            session_id: Optional session ID
            system_prompt: Optional custom system prompt
            
        Returns:
            String hash that uniquely identifies this query in its context
        """
        # Start with the query itself
        hash_content = query.strip().lower()
        
        # Add limited history if provided
        if history and cache_config.MAX_HISTORY_FOR_HASH > 0:
            recent_history = history[-cache_config.MAX_HISTORY_FOR_HASH:]
            for msg in recent_history:
                content = getattr(msg, 'content', str(msg))
                if isinstance(content, str):
                    hash_content += content.strip().lower()
        
        # Add session ID if configured
        if session_id and cache_config.CONSIDER_SESSION_IN_HASH:
            hash_content += session_id
            
        # Add system prompt if provided (this is critical for proper caching)
        if system_prompt:
            hash_content += "system_prompt:" + system_prompt.strip()
            
        # Generate hash
        query_hash = hashlib.md5(hash_content.encode('utf-8')).hexdigest()
        return query_hash
    
    def get_cached_response(self, 
                           query_hash: str) -> Tuple[Optional[Dict], bool]:
        """
        Retrieve a cached response for the given query hash.
        
        Args:
            query_hash: Hash identifying the query
            
        Returns:
            Tuple containing (cached_response, cache_hit_bool)
        """
        if not cache_config.CACHE_ENABLED:
            return None, False
            
        try:
            conn = sqlite3.connect(str(cache_config.CACHE_DB_PATH))
            cursor = conn.cursor()
            
            # Get cached response
            cursor.execute(
                "SELECT rag_response, no_rag_response, sources, timestamp, hit_count, system_prompt FROM chat_cache WHERE query_hash = ?", 
                (query_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # Handle the case where the result might not have system_prompt (older rows)
                if len(result) >= 6:
                    rag_response, no_rag_response, sources_json, timestamp, hit_count, system_prompt = result
                else:
                    rag_response, no_rag_response, sources_json, timestamp, hit_count = result
                    system_prompt = None
                
                # Check if cache entry has expired
                # Check if cache entry has expired
                age_in_seconds = time.time() - timestamp
                if age_in_seconds > cache_config.CACHE_TTL:
                    self.logger.info(f"Cache hit but expired for {query_hash} (age: {age_in_seconds/3600:.1f} hours)")
                    cursor.execute("DELETE FROM chat_cache WHERE query_hash = ?", (query_hash,))
                    conn.commit()
                    conn.close()
                    return None, False
                
                # Update hit count
                cursor.execute(
                    "UPDATE chat_cache SET hit_count = hit_count + 1 WHERE query_hash = ?", 
                    (query_hash,)
                )
                conn.commit()
                
                # Parse cached data
                sources = json.loads(sources_json) if sources_json else []
                
                cached_response = {
                    "rag_response": rag_response,
                    "no_rag_response": no_rag_response,
                    "sources": sources,
                    "system_prompt": system_prompt
                }
                
                self.logger.info(f"Cache hit for {query_hash}, hit count: {hit_count + 1}")
                conn.close()
                return cached_response, True
            
            conn.close()
            self.logger.info(f"Cache miss for {query_hash}")
            return None, False
            
        except Exception as e:
            self.logger.error(f"Error retrieving from cache: {e}")
            return None, False
    
    def cache_response(self, 
                      query_hash: str, 
                      user_input: str, 
                      rag_response: str, 
                      no_rag_response: str, 
                      sources: List = None,
                      system_prompt: str = None) -> bool:
        """
        Cache a response for future retrieval.
        
        Args:
            query_hash: Hash identifying the query
            user_input: Original user input
            rag_response: Response with RAG
            no_rag_response: Response without RAG
            sources: Optional list of sources
            system_prompt: Optional custom system prompt
            
        Returns:
            Boolean indicating success/failure
        """
        if not cache_config.CACHE_ENABLED:
            return False
            
        try:
            conn = sqlite3.connect(str(cache_config.CACHE_DB_PATH))
            cursor = conn.cursor()
            
            # Convert sources to JSON string
            sources_json = json.dumps(sources) if sources else None
            
            # Check if we have the system_prompt column before trying to use it
            cursor.execute("PRAGMA table_info(chat_cache)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if "system_prompt" in columns:
                # Store response in cache with system_prompt
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chat_cache 
                    (query_hash, user_input, rag_response, no_rag_response, sources, system_prompt, timestamp, hit_count) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """, 
                    (query_hash, user_input, rag_response, no_rag_response, sources_json, system_prompt, time.time())
                )
            else:
                # Handle case where system_prompt column doesn't exist (fallback)
                self.logger.warning("system_prompt column not found, attempting to insert without it")
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chat_cache 
                    (query_hash, user_input, rag_response, no_rag_response, sources, timestamp, hit_count) 
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """, 
                    (query_hash, user_input, rag_response, no_rag_response, sources_json, time.time())
                )
            
            # Ensure cache size doesn't exceed limit
            if cache_config.CACHE_SIZE_LIMIT > 0:
                cursor.execute(
                    """
                    DELETE FROM chat_cache 
                    WHERE query_hash IN (
                        SELECT query_hash FROM chat_cache 
                        ORDER BY timestamp ASC 
                        LIMIT max(0, (SELECT COUNT(*) FROM chat_cache) - ?)
                    )
                    """, 
                    (cache_config.CACHE_SIZE_LIMIT,)
                )
            
            conn.commit()
            conn.close()
            self.logger.info(f"Cached response for {query_hash}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
            return False
    
    def log_cache_access(self, 
                        session_id: str, 
                        user_input: str, 
                        query_hash: str, 
                        cache_hit: bool, 
                        response_time: float) -> None:
        """
        Log cache access statistics.
        
        Args:
            session_id: Session identifier
            user_input: User query
            query_hash: Hash of the query
            cache_hit: Whether the query was found in cache
            response_time: Time taken to process the query
        """
        if not cache_config.CACHE_LOG_ENABLED:
            return
            
        try:
            # Truncate user input if too long
            if len(user_input) > 200:
                user_input = user_input[:197] + "..."
                
            conn = sqlite3.connect(str(cache_config.CACHE_DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO cache_logs 
                (timestamp, session_id, user_input, cache_hit, query_hash, response_time) 
                VALUES (?, ?, ?, ?, ?, ?)
                """, 
                (time.time(), session_id, user_input, cache_hit, query_hash, response_time)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging cache access: {e}")
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            conn = sqlite3.connect(str(cache_config.CACHE_DB_PATH))
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute("SELECT COUNT(*) FROM chat_cache")
            total_entries = cursor.fetchone()[0]
            
            # Total cache hits
            cursor.execute("SELECT SUM(hit_count) FROM chat_cache")
            total_hits = cursor.fetchone()[0] or 0
            
            # Cache hit rate
            cursor.execute("SELECT COUNT(*) FROM cache_logs WHERE cache_hit = 1")
            log_hits = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM cache_logs")
            log_total = cursor.fetchone()[0]
            
            hit_rate = (log_hits / log_total) * 100 if log_total > 0 else 0
            
            # Average response time
            cursor.execute("SELECT AVG(response_time) FROM cache_logs WHERE cache_hit = 1")
            avg_hit_time = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT AVG(response_time) FROM cache_logs WHERE cache_hit = 0")
            avg_miss_time = cursor.fetchone()[0] or 0
            
            # Most frequent queries
            cursor.execute(
                """
                SELECT user_input, COUNT(*) 
                FROM cache_logs 
                GROUP BY user_input 
                ORDER BY COUNT(*) DESC 
                LIMIT 5
                """
            )
            frequent_queries = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_entries": total_entries,
                "total_hits": total_hits,
                "hit_rate_percent": round(hit_rate, 2),
                "avg_hit_time_ms": round(avg_hit_time * 1000, 2),
                "avg_miss_time_ms": round(avg_miss_time * 1000, 2),
                "time_saved_ms": round((avg_miss_time - avg_hit_time) * total_hits * 1000, 2),
                "frequent_queries": frequent_queries,
                "cache_size_limit": cache_config.CACHE_SIZE_LIMIT,
                "cache_ttl_hours": round(cache_config.CACHE_TTL / 3600, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            older_than_days: Optional, only clear entries older than this many days
            
        Returns:
            Number of entries cleared
        """
        try:
            conn = sqlite3.connect(str(cache_config.CACHE_DB_PATH))
            cursor = conn.cursor()
            
            if older_than_days is not None:
                # Convert days to seconds and calculate threshold time
                threshold_seconds = older_than_days * 86400
                threshold_time = time.time() - threshold_seconds
                
                # Get information about what will be deleted for logging
                cursor.execute(
                    "SELECT COUNT(*), MIN(time() - timestamp) / 3600, MAX(time() - timestamp) / 3600, AVG(hit_count) " +
                    "FROM chat_cache WHERE timestamp < ?", 
                    (threshold_time,)
                )
                count_result = cursor.fetchone()
                if count_result:
                    entries_to_delete, min_age_hours, max_age_hours, avg_hits = count_result
                    # Handle None values by providing defaults
                    entries_to_delete = entries_to_delete or 0
                    min_age_hours = 0 if min_age_hours is None else min_age_hours
                    max_age_hours = 0 if max_age_hours is None else max_age_hours
                    avg_hits = 0 if avg_hits is None else avg_hits
                    
                    self.logger.info(
                        f"Clearing {entries_to_delete} cache entries older than {older_than_days} days " +
                        f"(age range: {min_age_hours:.1f} to {max_age_hours:.1f} hours, avg hits: {avg_hits:.1f})"
                    )
                else:
                    entries_to_delete = 0
                
                # Perform the deletion
                cursor.execute("DELETE FROM chat_cache WHERE timestamp < ?", (threshold_time,))
            else:
                # Clearing all entries
                cursor.execute(
                    "SELECT COUNT(*), AVG(time() - timestamp) / 3600, AVG(hit_count) FROM chat_cache"
                )
                count_result = cursor.fetchone()
                if count_result:
                    entries_to_delete, avg_age_hours, avg_hits = count_result
                    # Handle None values by providing defaults
                    entries_to_delete = entries_to_delete or 0
                    avg_age_hours = 0 if avg_age_hours is None else avg_age_hours
                    avg_hits = 0 if avg_hits is None else avg_hits
                    
                    self.logger.info(
                        f"Clearing ALL {entries_to_delete} cache entries " +
                        f"(avg age: {avg_age_hours:.1f} hours, avg hits: {avg_hits:.1f})"
                    )
                else:
                    entries_to_delete = 0
                
                # Perform the deletion
                cursor.execute("DELETE FROM chat_cache")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Successfully cleared {entries_to_delete} cache entries")
            return entries_to_delete
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return 0

# Create singleton instance
chat_cache = ChatCacheService()