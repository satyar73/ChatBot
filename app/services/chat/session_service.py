"""
Service for managing persistent chat sessions.
Handles loading, saving, and querying sessions with multiple storage backends.
"""
import os
import json
import gzip
import sqlite3
import time
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.models.session_models import ChatSession
from app.utils.logging_utils import get_logger


class SessionManager:
    """
    Service for managing persistent chat sessions.
    Handles loading, saving, and querying sessions with multiple storage backends.
    """
    
    def __init__(self, 
                storage_type: str = "file", 
                storage_path: str = "./sessions",
                cache_size: int = 100,
                compression: bool = False,
                enable_backup: bool = True,
                backup_interval_hours: int = 24):
        """
        Initialize the session manager.
        
        Args:
            storage_type: Type of storage ("file", "sqlite", "memory")
            storage_path: Path to storage directory
            cache_size: Maximum number of sessions to keep in memory
            compression: Whether to compress stored sessions
            enable_backup: Whether to enable automatic backups
            backup_interval_hours: Hours between automatic backups
        """
        self.storage_type = storage_type
        self.storage_path = Path(storage_path)
        self.cache_size = cache_size
        self.compression = compression
        self.enable_backup = enable_backup
        self.backup_interval_hours = backup_interval_hours
        
        # Set up logging
        self.logger = get_logger(__name__)
        
        # Initialize in-memory cache with LRU tracking
        self.active_sessions: Dict[str, ChatSession] = {}
        self.session_last_accessed: Dict[str, float] = {}  # Maps session_id -> timestamp
        
        # Mutex for thread safety
        self.mutex = threading.RLock()
        
        # Initialize storage
        if storage_type == "file":
            os.makedirs(storage_path, exist_ok=True)
            os.makedirs(f"{storage_path}/backups", exist_ok=True)
        elif storage_type == "sqlite":
            self._init_db()
            
        # Set up automatic backup if enabled
        if self.enable_backup and storage_type != "memory":
            self._schedule_backup()
            
        self.logger.info(f"SessionManager initialized with {storage_type} storage at {storage_path}")
    
    def _init_db(self) -> None:
        """Initialize SQLite database with schema."""
        os.makedirs(self.storage_path, exist_ok=True)
        db_path = self.storage_path / "sessions.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            compressed INTEGER DEFAULT 0,
            size INTEGER,
            message_count INTEGER DEFAULT 0,
            prompt_count INTEGER DEFAULT 0
        )
        ''')
        
        # Create tags table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            UNIQUE(session_id, tag)
        )
        ''')
        
        # Create prompt modes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompt_modes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            prompt_index INTEGER NOT NULL,
            mode TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            UNIQUE(session_id, prompt_id)
        )
        ''')
        
        # Create indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_last_updated ON sessions(last_updated)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_tags_tag ON session_tags(tag)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompt_modes_mode ON prompt_modes(mode)')
        
        conn.commit()
        conn.close()
        
        self.logger.info("SQLite database initialized with schema")
    
    def get_session(self, session_id: str) -> ChatSession:
        """
        Get or create a session by ID with proper cache management.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            ChatSession instance
        """
        with self.mutex:
            # Check if session is already loaded in memory
            if session_id in self.active_sessions:
                # Update last accessed time for LRU tracking
                self.session_last_accessed[session_id] = time.time()
                return self.active_sessions[session_id]
            
            # Try to load from persistent storage
            session_data = self._load_session(session_id)
            if session_data:
                session = ChatSession.from_dict(session_data)
            else:
                # Create new session if not found
                session = ChatSession(session_id)
                self.logger.info(f"Created new session: {session_id}")
            
            # Add to in-memory cache
            self._add_to_cache(session_id, session)
            return session
    
    def _add_to_cache(self, session_id: str, session: ChatSession) -> None:
        """
        Add a session to the in-memory cache with LRU management.
        
        Args:
            session_id: Session identifier
            session: ChatSession instance
        """
        # Set last accessed time
        self.session_last_accessed[session_id] = time.time()
        
        # Add to active sessions
        self.active_sessions[session_id] = session
        
        # Check if we need to evict sessions to maintain cache size
        if len(self.active_sessions) > self.cache_size:
            self._evict_least_recently_used()
    
    def _evict_least_recently_used(self) -> None:
        """Evict the least recently used session from cache."""
        if not self.active_sessions:
            return
            
        # Find the least recently used session
        oldest_session_id = min(
            self.session_last_accessed.items(), 
            key=lambda x: x[1]
        )[0]
        
        # Save the session before evicting
        session = self.active_sessions[oldest_session_id]
        self.save_session(session)
        
        # Remove from cache and tracking
        del self.active_sessions[oldest_session_id]
        del self.session_last_accessed[oldest_session_id]
        
        self.logger.debug(f"Evicted least recently used session: {oldest_session_id}")
    
    def save_session(self, session: ChatSession) -> bool:
        """
        Save session to persistent storage.
        
        Args:
            session: ChatSession to save
            
        Returns:
            Boolean indicating success
        """
        with self.mutex:
            try:
                # Convert session to dictionary
                session_data = session.to_dict()
                
                # Update cache timestamp
                self.session_last_accessed[session.session_id] = time.time()
                
                # Save to storage with appropriate method
                if self.storage_type == "file":
                    return self._save_to_file(session.session_id, session_data)
                elif self.storage_type == "sqlite":
                    return self._save_to_sqlite(session.session_id, session_data)
                elif self.storage_type == "memory":
                    # Just keep in memory, nothing to do
                    return True
                else:
                    self.logger.error(f"Unknown storage type: {self.storage_type}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error saving session {session.session_id}: {e}")
                return False
    
    def _save_to_file(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Save session data to file with optional compression.
        
        Args:
            session_id: Session identifier
            session_data: Dictionary of session data
            
        Returns:
            Boolean indicating success
        """
        try:
            file_path = self.storage_path / f"{session_id}.json"
            
            # Convert to JSON
            json_data = json.dumps(session_data)
            
            if self.compression:
                # Compress with gzip
                compressed_path = self.storage_path / f"{session_id}.json.gz"
                with gzip.open(str(compressed_path), 'wt', encoding='utf-8') as f:
                    f.write(json_data)
                
                # Remove uncompressed file if it exists
                if file_path.exists():
                    os.remove(file_path)
                    
                self.logger.debug(f"Saved compressed session to {compressed_path}")
            else:
                # Save without compression
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
                
                # Remove compressed file if it exists    
                compressed_path = self.storage_path / f"{session_id}.json.gz"
                if compressed_path.exists():
                    os.remove(compressed_path)
                    
                self.logger.debug(f"Saved session to {file_path}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session {session_id} to file: {e}")
            return False
    
    def _save_to_sqlite(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Save session data to SQLite with indexing.
        
        Args:
            session_id: Session identifier
            session_data: Dictionary of session data
            
        Returns:
            Boolean indicating success
        """
        try:
            db_path = self.storage_path / "sessions.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Convert to JSON
            json_data = json.dumps(session_data)
            
            # Compress if enabled
            is_compressed = 0
            if self.compression:
                json_data = json_data.encode('utf-8')
                json_data = gzip.compress(json_data)
                is_compressed = 1
            
            # Extract metadata
            created_at = session_data["metadata"]["created_at"]
            last_updated = session_data["metadata"]["last_updated"]
            message_count = session_data["metadata"].get("message_count", len(session_data["messages"]))
            prompt_count = session_data["metadata"].get("prompt_count", len(session_data["prompts_and_responses"]))
            
            # Calculate size
            size = len(json_data)
            
            # Upsert session
            cursor.execute(
                """
                INSERT INTO sessions 
                (session_id, data, created_at, last_updated, compressed, size, message_count, prompt_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                data=excluded.data, 
                last_updated=excluded.last_updated,
                compressed=excluded.compressed,
                size=excluded.size,
                message_count=excluded.message_count,
                prompt_count=excluded.prompt_count
                """,
                (session_id, json_data, created_at, last_updated, is_compressed, size, message_count, prompt_count)
            )
            
            # Update tags
            self._update_tags(cursor, session_id, session_data["metadata"].get("tags", []))
            
            # Update prompt modes index
            self._update_prompt_modes(cursor, session_id, session_data["prompts_and_responses"])
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Saved session to SQLite: {session_id}, size: {size} bytes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session {session_id} to SQLite: {e}")
            return False
    
    def _update_tags(self, cursor: sqlite3.Cursor, session_id: str, tags: List[str]) -> None:
        """
        Update the tags for a session in the database.
        
        Args:
            cursor: SQLite cursor
            session_id: Session identifier
            tags: List of tags
        """
        # Clear existing tags
        cursor.execute("DELETE FROM session_tags WHERE session_id = ?", (session_id,))
        
        # Add new tags
        for tag in tags:
            cursor.execute(
                "INSERT INTO session_tags (session_id, tag) VALUES (?, ?)",
                (session_id, tag)
            )
    
    def _update_prompt_modes(self, cursor: sqlite3.Cursor, session_id: str, 
                           prompt_responses: List[Dict[str, Any]]) -> None:
        """
        Update the prompt modes index in the database.
        
        Args:
            cursor: SQLite cursor
            session_id: Session identifier
            prompt_responses: List of prompt-response pairs
        """
        # Clear existing prompt modes
        cursor.execute("DELETE FROM prompt_modes WHERE session_id = ?", (session_id,))
        
        # Add new prompt modes
        for idx, pair in enumerate(prompt_responses):
            prompt = pair["prompt"]
            prompt_id = prompt.get("id", str(uuid.uuid4()))
            mode = prompt["attributes"].get("mode", "unknown")
            timestamp = prompt["timestamp"]
            
            cursor.execute(
                "INSERT INTO prompt_modes (session_id, prompt_id, prompt_index, mode, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, prompt_id, idx, mode, timestamp)
            )
    
    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        try:
            if self.storage_type == "file":
                return self._load_from_file(session_id)
            elif self.storage_type == "sqlite":
                return self._load_from_sqlite(session_id)
            elif self.storage_type == "memory":
                # Memory-only storage doesn't persist
                return None
            else:
                self.logger.error(f"Unknown storage type: {self.storage_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def _load_from_file(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from file with compression support.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        # Try compressed file first
        compressed_path = self.storage_path / f"{session_id}.json.gz"
        if compressed_path.exists():
            try:
                with gzip.open(str(compressed_path), 'rt', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading compressed session {session_id}: {e}")
                
        # Try uncompressed file
        file_path = self.storage_path / f"{session_id}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading session {session_id}: {e}")
                
        return None
    
    def _load_from_sqlite(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from SQLite with compression support.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        db_path = self.storage_path / "sessions.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        try:
            # Get session data with compression flag
            cursor.execute(
                "SELECT data, compressed FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return None
                
            data, is_compressed = result
            
            # Decompress if needed
            if is_compressed:
                try:
                    if isinstance(data, str):
                        data = data.encode('utf-8')
                    data = gzip.decompress(data)
                    data = data.decode('utf-8')
                except Exception as e:
                    self.logger.error(f"Error decompressing session {session_id}: {e}")
                    conn.close()
                    return None
            
            # Parse JSON
            if isinstance(data, str):
                session_data = json.loads(data)
            else:
                session_data = json.loads(data.decode('utf-8'))
                
            conn.close()
            return session_data
            
        except Exception as e:
            self.logger.error(f"Error loading session {session_id} from SQLite: {e}")
            conn.close()
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from storage and cache.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Boolean indicating success
        """
        with self.mutex:
            try:
                # Remove from memory if present
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                if session_id in self.session_last_accessed:
                    del self.session_last_accessed[session_id]
                
                # Remove from storage
                if self.storage_type == "file":
                    success = self._delete_from_file(session_id)
                elif self.storage_type == "sqlite":
                    success = self._delete_from_sqlite(session_id)
                elif self.storage_type == "memory":
                    # Nothing to do for memory-only storage
                    success = True
                else:
                    self.logger.error(f"Unknown storage type: {self.storage_type}")
                    success = False
                    
                if success:
                    self.logger.info(f"Deleted session: {session_id}")
                    
                return success
                    
            except Exception as e:
                self.logger.error(f"Error deleting session {session_id}: {e}")
                return False
    
    def _delete_from_file(self, session_id: str) -> bool:
        """
        Delete session files from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Boolean indicating success
        """
        success = True
        
        # Try to delete both regular and compressed files
        file_path = self.storage_path / f"{session_id}.json"
        compressed_path = self.storage_path / f"{session_id}.json.gz"
        
        if file_path.exists():
            try:
                os.remove(file_path)
            except Exception as e:
                self.logger.error(f"Error deleting session file {file_path}: {e}")
                success = False
                
        if compressed_path.exists():
            try:
                os.remove(compressed_path)
            except Exception as e:
                self.logger.error(f"Error deleting compressed session file {compressed_path}: {e}")
                success = False
                
        return success
    
    def _delete_from_sqlite(self, session_id: str) -> bool:
        """
        Delete session from SQLite database.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Boolean indicating success
        """
        try:
            db_path = self.storage_path / "sessions.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA foreign_keys = ON")  # Enable cascade deletion
            cursor = conn.cursor()
            
            # Delete from sessions (will cascade to other tables)
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting session {session_id} from SQLite: {e}")
            return False
    
    def delete_all_sessions(self) -> int:
        """
        Delete all sessions.
        
        Returns:
            Number of sessions deleted
        """
        with self.mutex:
            try:
                # Count how many sessions we have
                session_count = len(self.active_sessions)
                
                if self.storage_type == "file":
                    # Count sessions on disk
                    regular_files = list(self.storage_path.glob("*.json"))
                    compressed_files = list(self.storage_path.glob("*.json.gz"))
                    session_count = len(regular_files) + len(compressed_files)
                    
                    # Delete all session files
                    for file_path in regular_files + compressed_files:
                        os.remove(file_path)
                        
                elif self.storage_type == "sqlite":
                    # Count sessions in database
                    db_path = self.storage_path / "sessions.db"
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT COUNT(*) FROM sessions")
                    session_count = cursor.fetchone()[0]
                    
                    # Delete all sessions
                    cursor.execute("DELETE FROM sessions where 1 != 0")
                    conn.commit()
                    conn.close()
                    
                # Clear memory cache
                self.active_sessions.clear()
                self.session_last_accessed.clear()
                
                self.logger.info(f"Deleted all sessions ({session_count} sessions)")
                return session_count
                
            except Exception as e:
                self.logger.error(f"Error deleting all sessions: {e}")
                return 0
    
    def list_sessions(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List available sessions with metadata and pagination.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Starting offset
            
        Returns:
            List of session metadata dictionaries
        """
        if self.storage_type == "file":
            return self._list_sessions_file(limit, offset)
        elif self.storage_type == "sqlite":
            return self._list_sessions_sqlite(limit, offset)
        elif self.storage_type == "memory":
            return self._list_sessions_memory(limit, offset)
        else:
            self.logger.error(f"Unknown storage type: {self.storage_type}")
            return []
    
    def _list_sessions_file(self, limit: Optional[int], offset: int) -> List[Dict[str, Any]]:
        """List sessions from file storage."""
        session_files = []
        
        # Collect both regular and compressed files
        regular_files = list(self.storage_path.glob("*.json"))
        compressed_files = list(self.storage_path.glob("*.json.gz"))
        
        # Extract session IDs from filenames
        for file_path in regular_files:
            session_id = file_path.stem
            session_files.append((session_id, file_path))
            
        for file_path in compressed_files:
            # Remove both .json and .gz extensions
            session_id = file_path.stem.rsplit('.', 1)[0]
            # Skip if already added from regular files
            if session_id not in [s[0] for s in session_files]:
                session_files.append((session_id, file_path))
        
        # Sort by modification time (most recent first)
        session_files.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
        
        # Apply pagination
        session_files = session_files[offset:]
        if limit is not None:
            session_files = session_files[:limit]
            
        sessions = []
        
        # Load minimal metadata for each session
        for session_id, file_path in session_files:
            try:
                is_compressed = file_path.suffix == ".gz"
                    
                if is_compressed:
                    with gzip.open(str(file_path), 'rt', encoding='utf-8') as f:
                        # Just read enough to get metadata
                        data = json.loads(f.read(4096))
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.loads(f.read(4096))
                
                # Extract key metadata
                metadata = data.get("metadata", {})
                sessions.append({
                    "session_id": session_id,
                    "created_at": metadata.get("created_at", ""),
                    "last_updated": metadata.get("last_updated", ""),
                    "message_count": metadata.get("message_count", 0),
                    "prompt_count": metadata.get("prompt_count", 0),
                    "tags": metadata.get("tags", []),
                    "file_size": file_path.stat().st_size
                })
            except Exception as e:
                self.logger.error(f"Error reading metadata for session {session_id}: {e}")
            
        return sessions
    
    def _list_sessions_sqlite(self, limit: Optional[int], offset: int) -> List[Dict[str, Any]]:
        """List sessions from SQLite storage with efficient querying."""
        try:
            db_path = self.storage_path / "sessions.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Query sessions with pagination
            query = """
            SELECT 
                s.session_id, s.created_at, s.last_updated, 
                s.message_count, s.prompt_count, s.size,
                GROUP_CONCAT(t.tag, ',') as tags
            FROM 
                sessions s
            LEFT JOIN 
                session_tags t ON s.session_id = t.session_id
            GROUP BY 
                s.session_id
            ORDER BY 
                s.last_updated DESC
            """
            
            # Add pagination
            if offset > 0:
                if limit is not None:
                    query += f" LIMIT {limit} OFFSET {offset}"
                else:
                    query += f" LIMIT -1 OFFSET {offset}"  # Use -1 or a very large number to get all records
            elif limit is not None:
                query += f" LIMIT {limit}"
                
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Format results
            sessions = []
            for row in results:
                session_id, created_at, last_updated, message_count, prompt_count, size, tags_str = row
                
                tags = tags_str.split(',') if tags_str else []
                if tags and tags[0] == '':
                    tags = []
                    
                sessions.append({
                    "session_id": session_id,
                    "created_at": created_at,
                    "last_updated": last_updated,
                    "message_count": message_count,
                    "prompt_count": prompt_count,
                    "tags": tags,
                    "file_size": size
                })
                
            conn.close()
            return sessions
            
        except Exception as e:
            self.logger.error(f"Error listing sessions from SQLite: {e}")
            return []
    
    def _list_sessions_memory(self, limit: Optional[int], offset: int) -> List[Dict[str, Any]]:
        """List sessions from memory cache."""
        # Sort sessions by last accessed time (most recent first)
        sorted_sessions = sorted(
            self.active_sessions.items(),
            key=lambda x: self.session_last_accessed.get(x[0], 0),
            reverse=True
        )
        
        # Apply pagination
        sorted_sessions = sorted_sessions[offset:]
        if limit is not None:
            sorted_sessions = sorted_sessions[:limit]
            
        sessions = []
        
        # Collect metadata
        for session_id, session in sorted_sessions:
            metadata = session.metadata
            sessions.append({
                "session_id": session_id,
                "created_at": metadata.get("created_at", ""),
                "last_updated": metadata.get("last_updated", ""),
                "message_count": metadata.get("message_count", 0),
                "prompt_count": metadata.get("prompt_count", 0),
                "tags": metadata.get("tags", []),
                "in_memory": True
            })
            
        return sessions
    
    def find_sessions_by_tag(self, tag: str) -> List[str]:
        """
        Find all sessions with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching session IDs
        """
        if self.storage_type == "sqlite":
            try:
                db_path = self.storage_path / "sessions.db"
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT session_id FROM session_tags WHERE tag = ?",
                    (tag,)
                )
                
                session_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                return session_ids
                
            except Exception as e:
                self.logger.error(f"Error finding sessions by tag: {e}")
                return []
        else:
            # For file and memory storage, check each session
            session_ids = []
            
            # Get all session IDs
            all_ids = set()
            
            # Add IDs from memory
            for session_id in self.active_sessions.keys():
                all_ids.add(session_id)
                
            # Add IDs from files if using file storage
            if self.storage_type == "file":
                regular_files = list(self.storage_path.glob("*.json"))
                compressed_files = list(self.storage_path.glob("*.json.gz"))
                
                for file_path in regular_files:
                    all_ids.add(file_path.stem)
                    
                for file_path in compressed_files:
                    session_id = file_path.stem.rsplit('.', 1)[0]
                    all_ids.add(session_id)
            
            # Check each session for the tag
            for session_id in all_ids:
                try:
                    # Try in-memory cache first
                    if session_id in self.active_sessions:
                        session = self.active_sessions[session_id]
                        if tag in session.metadata.get("tags", []):
                            session_ids.append(session_id)
                    else:
                        # Load minimal metadata
                        data = self._load_session(session_id)
                        if data and tag in data.get("metadata", {}).get("tags", []):
                            session_ids.append(session_id)
                except Exception as e:
                    self.logger.error(f"Error checking tags for session {session_id}: {e}")
            
            return session_ids
    
    def find_sessions_by_mode(self, mode: str) -> List[str]:
        """
        Find all sessions containing prompts with a specific mode.
        
        Args:
            mode: Mode to search for (rag, standard, needl, compare)
            
        Returns:
            List of matching session IDs
        """
        mode_lower = mode.lower()
        
        if self.storage_type == "sqlite":
            try:
                db_path = self.storage_path / "sessions.db"
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT DISTINCT session_id FROM prompt_modes WHERE mode = ?",
                    (mode_lower,)
                )
                
                session_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                return session_ids
                
            except Exception as e:
                self.logger.error(f"Error finding sessions by mode: {e}")
                return []
        else:
            # For file and memory storage, check each session
            session_ids = []
            
            # Get all session IDs
            all_ids = set()
            
            # Add IDs from memory
            for session_id in self.active_sessions.keys():
                all_ids.add(session_id)
                
            # Add IDs from files if using file storage
            if self.storage_type == "file":
                regular_files = list(self.storage_path.glob("*.json"))
                compressed_files = list(self.storage_path.glob("*.json.gz"))
                
                for file_path in regular_files:
                    all_ids.add(file_path.stem)
                    
                for file_path in compressed_files:
                    session_id = file_path.stem.rsplit('.', 1)[0]
                    all_ids.add(session_id)
            
            # Check each session for the mode
            for session_id in all_ids:
                try:
                    session = self.get_session(session_id)
                    
                    # Check if any prompt has this mode
                    mode_found = False
                    for pair in session.prompts_and_responses:
                        prompt_mode = pair["prompt"]["attributes"].get("mode", "").lower()
                        if prompt_mode == mode_lower:
                            mode_found = True
                            break
                            
                    if mode_found:
                        session_ids.append(session_id)
                        
                except Exception as e:
                    self.logger.error(f"Error checking modes for session {session_id}: {e}")
            
            return session_ids
    
    def create_backup(self, backup_path: Optional[str] = None) -> Optional[str]:
        """
        Create a backup of all sessions.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to the created backup or None on failure
        """
        with self.mutex:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if backup_path is None:
                backup_path = str(self.storage_path / "backups" / f"sessions_backup_{timestamp}")
                
            try:
                if self.storage_type == "file":
                    # Create zip archive of all files
                    backup_file = f"{backup_path}.zip"
                    
                    # Ensure all sessions are saved
                    for session_id, session in self.active_sessions.items():
                        self.save_session(session)
                    
                    # Create archive of regular files
                    regular_files = list(self.storage_path.glob("*.json"))
                    compressed_files = list(self.storage_path.glob("*.json.gz"))
                    
                    import zipfile
                    with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file_path in regular_files + compressed_files:
                            zipf.write(
                                file_path, 
                                arcname=file_path.name
                            )
                    
                    self.logger.info(f"Created backup at {backup_file}")
                    return backup_file
                    
                elif self.storage_type == "sqlite":
                    # Create backup of SQLite database
                    backup_file = f"{backup_path}.db"
                    
                    db_path = self.storage_path / "sessions.db"
                    
                    # Copy database file
                    shutil.copy2(db_path, backup_file)
                    
                    self.logger.info(f"Created backup at {backup_file}")
                    return backup_file
                    
                elif self.storage_type == "memory":
                    # Export all in-memory sessions to JSON
                    backup_file = f"{backup_path}.json"
                    
                    # Convert all sessions to dicts
                    sessions_data = {}
                    for session_id, session in self.active_sessions.items():
                        sessions_data[session_id] = session.to_dict()
                    
                    # Save to file
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(sessions_data, f, indent=2)
                    
                    self.logger.info(f"Created backup at {backup_file}")
                    return backup_file
                    
                else:
                    self.logger.error(f"Backup not supported for storage type: {self.storage_type}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Error creating backup: {e}")
                return None
    
    def restore_backup(self, backup_path: str) -> int:
        """
        Restore sessions from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            Number of sessions restored
        """
        with self.mutex:
            try:
                backup_path = Path(backup_path)
                
                if not backup_path.exists():
                    self.logger.error(f"Backup file does not exist: {backup_path}")
                    return 0
                
                # Determine backup type
                if backup_path.suffix == ".zip":
                    return self._restore_file_backup(backup_path)
                elif backup_path.suffix == ".db":
                    return self._restore_sqlite_backup(backup_path)
                elif backup_path.suffix == ".json":
                    return self._restore_json_backup(backup_path)
                else:
                    self.logger.error(f"Unknown backup format: {backup_path.suffix}")
                    return 0
                    
            except Exception as e:
                self.logger.error(f"Error restoring backup: {e}")
                return 0
    
    def _restore_file_backup(self, backup_path: Path) -> int:
        """Restore from a file-based ZIP backup."""
        if self.storage_type != "file":
            self.logger.error("Cannot restore file backup to non-file storage")
            return 0
            
        try:
            import zipfile
            
            # Create a temporary directory for extraction
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract all files
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    zipf.extractall(temp_path)
                
                # Move files to sessions directory
                session_count = 0
                for file_path in temp_path.glob("*"):
                    shutil.copy2(file_path, self.storage_path / file_path.name)
                    session_count += 1
                
                # Clear memory cache to reflect restored sessions
                self.active_sessions.clear()
                self.session_last_accessed.clear()
                
                self.logger.info(f"Restored {session_count} sessions from {backup_path}")
                return session_count
                
        except Exception as e:
            self.logger.error(f"Error restoring file backup: {e}")
            return 0
    
    def _restore_sqlite_backup(self, backup_path: Path) -> int:
        """Restore from a SQLite database backup."""
        if self.storage_type != "sqlite":
            self.logger.error("Cannot restore SQLite backup to non-SQLite storage")
            return 0
            
        try:
            # Connect to back-up database to get session count
            backup_conn = sqlite3.connect(str(backup_path))
            backup_cursor = backup_conn.cursor()
            
            backup_cursor.execute("SELECT COUNT(*) FROM sessions")
            session_count = backup_cursor.fetchone()[0]
            backup_conn.close()
            
            # Copy backup to the current database
            db_path = self.storage_path / "sessions.db"
            
            # Close any existing connections
            self.active_sessions.clear()
            self.session_last_accessed.clear()
            
            # Replace database file
            shutil.copy2(backup_path, db_path)
            
            self.logger.info(f"Restored {session_count} sessions from {backup_path}")
            return session_count
            
        except Exception as e:
            self.logger.error(f"Error restoring SQLite backup: {e}")
            return 0
    
    def _restore_json_backup(self, backup_path: Path) -> int:
        """Restore from a JSON export backup."""
        try:
            # Load the JSON data
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
                
            if not isinstance(backup_data, dict):
                self.logger.error("Invalid backup data format")
                return 0
                
            # Import all sessions
            session_count = 0
            for session_id, session_data in backup_data.items():
                try:
                    session = ChatSession.from_dict(session_data)
                    self.save_session(session)
                    session_count += 1
                except Exception as e:
                    self.logger.error(f"Error restoring session {session_id}: {e}")
            
            self.logger.info(f"Restored {session_count} sessions from {backup_path}")
            return session_count
            
        except Exception as e:
            self.logger.error(f"Error restoring JSON backup: {e}")
            return 0
    
    def _schedule_backup(self) -> None:
        """Schedule a periodic backup task."""
        if not self.enable_backup:
            return
            
        # Create an automatic backup thread
        def backup_task():
            while self.enable_backup:
                # Sleep for the specified interval
                time.sleep(self.backup_interval_hours * 3600)
                
                try:
                    # Create automatic backup
                    self.create_backup()
                except Exception as e:
                    self.logger.error(f"Error in automatic backup: {e}")
        
        # Start the backup thread
        backup_thread = threading.Thread(target=backup_task, daemon=True)
        backup_thread.start()
        
        self.logger.info(f"Automatic backups scheduled every {self.backup_interval_hours} hours")


# Create a singleton instance of SessionManager
import uuid
session_manager = SessionManager(
    storage_type="sqlite",  # Changed from "file" to "sqlite"
    storage_path="./data/sessions",
    cache_size=100,
    compression=True,
    enable_backup=True
)