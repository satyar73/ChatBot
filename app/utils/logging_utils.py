# File: app/utils/logging_utils.py
import logging  # Make sure this import is at the top of the file
import os
import sys
from typing import Optional, Dict, Any
import logging.config  # Also import logging.config at the top

def configure_logging(log_level: Optional[str] = None) -> None:
    """
    Configure application-wide logging settings for FastAPI application.

    Args:
        log_level: Optional override for log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  If not provided, will try to get from environment variable LOG_LEVEL,
                  defaulting to INFO if not set.
    """
    # Determine the log level
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()

    numeric_level = getattr(logging, log_level, logging.INFO)
    
    # Create centralized logs directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up log file paths
    app_log_file = os.path.join(logs_dir, "app.log")
    error_log_file = os.path.join(logs_dir, "error.log")

    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
        },
        "handlers": {
            "console": {
                "level": "DEBUG",  # Set handler to DEBUG to allow all messages
                "class": "logging.StreamHandler",
                "formatter": "detailed",
                "stream": sys.stderr  # Use stderr for better Docker compatibility
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": app_log_file,
                "maxBytes": 1073741824,  # 1GB
                "backupCount": 5,
                "encoding": "utf-8"
            },
            "error_file": {
                "level": "ERROR",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": error_log_file,
                "maxBytes": 1073741824,  # 1GB
                "backupCount": 5,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            # FastAPI and Uvicorn loggers
            "uvicorn": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
            "uvicorn.error": {"level": "INFO", "handlers": ["console", "file", "error_file"], "propagate": False},
            "uvicorn.access": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
            "fastapi": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},

            # Application loggers
            "app": {"level": log_level, "handlers": ["console", "file"], "propagate": False},
            "app.services": {"level": log_level, "handlers": ["console", "file"], "propagate": False},
            "app.services.chat_service": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False},
            "app.services.chat_test_service": {"level": log_level, "handlers": ["console", "file"], "propagate": False},
            "prompt_capture": {"level": "DEBUG", "handlers": ["console"], "propagate": False},

            # Third-party libraries
            "urllib3": {"level": "WARNING", "handlers": ["console", "file"], "propagate": False},
            "httpx": {"level": "WARNING", "handlers": ["console", "file"], "propagate": False},
        },
        "root": {"level": log_level, "handlers": ["console", "file", "error_file"]},
    }

    # Apply the configuration
    logging.config.dictConfig(logging_config)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")

def diagnose_logger(name: str) -> None:
    """Print diagnostic information about a logger."""
    import logging
    logger = logging.getLogger(name)

    print(f"\n=== LOGGER DIAGNOSIS FOR '{name}' ===")
    print(f"Logger level: {logger.level}")
    print(f"Logger effective level: {logger.getEffectiveLevel()}")
    print(f"Logger disabled: {logger.disabled}")
    print(f"Logger propagate: {logger.propagate}")
    print(f"Logger handlers: {logger.handlers}")

    # Check parent loggers (which might be controlling the effective level)
    parts = name.split('.')
    for i in range(len(parts)):
        parent_name = '.'.join(parts[:i]) or 'root'
        parent = logging.getLogger(parent_name)
        print(f"Parent '{parent_name}' level: {parent.level}")

    # The root logger is the ultimate parent
    root = logging.getLogger()
    print(f"Root logger level: {root.level}")
    print(f"Root logger handlers: {root.handlers}")
    print("===================================\n")

def ensure_debug_logging(name: str = None) -> None:
    """
    Force DEBUG level logging to work for a specific logger and its ancestors.

    Args:
        name: Logger name, or None to fix all loggers
    """

    # Set root logger to DEBUG if name is None
    if name is None:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        print(f"Set root logger to DEBUG level")

        # Ensure root has a handler
        if not root.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            root.addHandler(handler)
            print("Added handler to root logger")

        return

    # Get the specified logger
    logger = logging.getLogger(name)

    # Fix this logger
    logger.setLevel(logging.DEBUG)
    print(f"Set {name} logger to DEBUG level")

    # Also fix parent loggers that might block propagation
    parts = name.split('.')
    for i in range(len(parts)):
        parent_name = '.'.join(parts[:i]) or 'root'
        parent = logging.getLogger(parent_name)
        if parent.level > logging.DEBUG:
            parent.setLevel(logging.DEBUG)
            print(f"Also set {parent_name} logger to DEBUG level")

    # If this logger has no handlers and doesn't propagate, add a handler
    if not logger.handlers and not logger.propagate:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        print(f"Added handler to {name} logger")

# Enhance the get_logger function to be more robust and support rotating files
def get_logger(name: str, log_level: str = None, use_rotating_file: bool = True, 
               log_file: str = None) -> logging.Logger:
    """
    Get a configured logger with proper debug support.

    Args:
        name: Logger name (usually __name__)
        log_level: Optional log level as string ('DEBUG', 'INFO', etc.)
        use_rotating_file: Whether to add a rotating file handler
        log_file: Optional specific log file path (defaults to logs/module_name.log)

    Returns:
        Configured logger
    """
    import logging
    from logging.handlers import RotatingFileHandler

    # Get the logger
    logger = logging.getLogger(name)

    # Set level if specified
    if log_level is not None:
        # Handle both string and numeric levels
        if isinstance(log_level, str):
            numeric_level = getattr(logging, log_level.upper(), None)
            if numeric_level is None:
                # Default to DEBUG if invalid level
                numeric_level = logging.DEBUG
                print(f"Warning: Invalid log level '{log_level}', defaulting to DEBUG")
        else:
            numeric_level = log_level

        logger.setLevel(numeric_level)

    # Ensure logger works by checking some conditions
    if logger.getEffectiveLevel() > logging.DEBUG and log_level in ('DEBUG', logging.DEBUG):
        # This shouldn't happen if we just set it to DEBUG, so diagnose
        print(f"Warning: Logger {name} effective level {logger.getEffectiveLevel()} > DEBUG after setting to DEBUG")
        diagnose_logger(name)

        # Try to fix it
        ensure_debug_logging(name)

    # If using rotating file and no file handlers present
    if use_rotating_file and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        try:
            # Set up log file path
            if log_file is None:
                # Create centralized logs directory
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                logs_dir = os.path.join(project_root, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                
                # Create a log file name based on the module name
                module_part = name.split('.')[-1]
                log_file = os.path.join(logs_dir, f"{module_part}.log")
            
            # Add rotating file handler (1GB max size, keep 5 backup files)
            handler = RotatingFileHandler(
                log_file,
                maxBytes=1073741824,  # 1GB
                backupCount=5,
                encoding='utf-8'
            )
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            print(f"Added rotating file handler to {name} logger, logging to {log_file}")
        except Exception as e:
            print(f"Warning: Failed to set up rotating file handler: {e}")

    # If no handlers in logger hierarchy, add one
    has_handlers = logger.handlers
    if not has_handlers and not logger.propagate:
        # No handlers and not propagating - add a handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def update_logger_levels(logger_levels: Dict[str, str]) -> None:
    """
    Update log levels for specific loggers at runtime.

    Args:
        logger_levels: Dictionary mapping logger names to log levels
    """
    for logger_name, level in logger_levels.items():
        numeric_level = getattr(logging, level.upper(), None)
        if numeric_level is not None:
            logging.getLogger(logger_name).setLevel(numeric_level)