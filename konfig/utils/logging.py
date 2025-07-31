"""
Logging utilities for Konfig.

This module provides structured logging capabilities with support for
different output formats and integration with external monitoring systems.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from rich.logging import RichHandler

from konfig.config.settings import get_settings


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    This function sets up logging with the following features:
    - Structured JSON logging for production
    - Rich console logging for development
    - Automatic log rotation
    - Integration with external monitoring systems
    """
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_correlation_id,
            _add_service_info,
            structlog.processors.JSONRenderer() if settings.logging.structured else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format="%(message)s",
        handlers=_get_handlers(settings),
    )
    
    # Set up logger for specific modules
    _configure_module_loggers(settings)


def _get_handlers(settings) -> list:
    """Get logging handlers based on configuration."""
    handlers = []
    
    if settings.is_development() or not settings.logging.structured:
        # Use Rich handler for development
        handlers.append(
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=True,
                markup=True,
            )
        )
    else:
        # Use standard stream handler for production
        handlers.append(logging.StreamHandler(sys.stdout))
    
    # Add file handler if specified
    if settings.logging.file:
        log_file = Path(settings.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
                if settings.logging.structured
                else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        handlers.append(file_handler)
    
    return handlers


def _configure_module_loggers(settings) -> None:
    """Configure loggers for specific modules."""
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # Set specific levels for Konfig modules in development
    if settings.is_development():
        logging.getLogger("konfig.orchestrator").setLevel(logging.DEBUG)
        logging.getLogger("konfig.modules").setLevel(logging.DEBUG)
        logging.getLogger("konfig.tools").setLevel(logging.DEBUG)


def _add_correlation_id(logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add correlation ID to log entries for request tracing."""
    # This will be implemented when we add request context
    # For now, we'll use a placeholder
    correlation_id = getattr(logger, '_correlation_id', None)
    if correlation_id:
        event_dict['correlation_id'] = correlation_id
    return event_dict


def _add_service_info(logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add service information to log entries."""
    event_dict.update({
        'service': 'konfig',
        'version': '0.1.0',
        'component': getattr(logger, '_component', 'unknown'),
    })
    return event_dict


def get_logger(name: str, component: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a configured logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        component: Component name for categorization
        
    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger(name)
    if component:
        logger = logger.bind(component=component)
    return logger


def set_correlation_id(logger: structlog.BoundLogger, correlation_id: str) -> structlog.BoundLogger:
    """
    Set correlation ID for request tracing.
    
    Args:
        logger: Logger instance
        correlation_id: Unique identifier for the request/job
        
    Returns:
        Logger bound with correlation ID
    """
    return logger.bind(correlation_id=correlation_id)


class LoggingMixin:
    """
    Mixin class to add logging capabilities to any class.
    
    Usage:
        class MyClass(LoggingMixin):
            def __init__(self):
                super().__init__()
                self.setup_logging("my_component")
            
            def do_something(self):
                self.logger.info("Doing something", extra_data="value")
    """
    
    def setup_logging(self, component: str, correlation_id: Optional[str] = None) -> None:
        """
        Set up logging for the class instance.
        
        Args:
            component: Component name for log categorization
            correlation_id: Optional correlation ID for request tracing
        """
        self.logger = get_logger(self.__class__.__module__, component)
        if correlation_id:
            self.logger = set_correlation_id(self.logger, correlation_id)
    
    def log_method_call(self, method_name: str, **kwargs) -> None:
        """
        Log a method call with parameters.
        
        Args:
            method_name: Name of the method being called
            **kwargs: Method parameters to log
        """
        if hasattr(self, 'logger'):
            # Filter out sensitive information
            safe_kwargs = {
                k: v if not _is_sensitive_key(k) else "[REDACTED]"
                for k, v in kwargs.items()
            }
            self.logger.debug(
                f"Calling {method_name}",
                method=method_name,
                parameters=safe_kwargs
            )
    
    def log_method_result(self, method_name: str, result: Any = None, duration_ms: Optional[float] = None) -> None:
        """
        Log a method result.
        
        Args:
            method_name: Name of the method that completed
            result: Method result (will be serialized safely)
            duration_ms: Method execution duration in milliseconds
        """
        if hasattr(self, 'logger'):
            log_data = {"method": method_name}
            
            if duration_ms is not None:
                log_data["duration_ms"] = duration_ms
            
            if result is not None:
                # Safely serialize result
                try:
                    if isinstance(result, (str, int, float, bool, list, dict)):
                        log_data["result"] = result
                    else:
                        log_data["result_type"] = type(result).__name__
                except Exception:
                    log_data["result_type"] = "unserializable"
            
            self.logger.debug(f"Completed {method_name}", **log_data)
    
    def log_error(self, method_name: str, error: Exception, **context) -> None:
        """
        Log an error with context.
        
        Args:
            method_name: Name of the method where error occurred
            error: The exception that occurred
            **context: Additional context information
        """
        if hasattr(self, 'logger'):
            self.logger.error(
                f"Error in {method_name}: {str(error)}",
                method=method_name,
                error_type=type(error).__name__,
                error_message=str(error),
                **context,
                exc_info=True
            )


def _is_sensitive_key(key: str) -> bool:
    """Check if a key contains potentially sensitive information."""
    sensitive_keywords = {
        'password', 'token', 'key', 'secret', 'auth', 'credential',
        'api_key', 'access_token', 'refresh_token', 'jwt', 'bearer'
    }
    key_lower = key.lower()
    return any(keyword in key_lower for keyword in sensitive_keywords)


# Context manager for timing operations
class LogTimer:
    """Context manager for timing and logging operations."""
    
    def __init__(self, logger: structlog.BoundLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}", **self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                duration_ms=round(duration_ms, 2),
                **self.context
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                duration_ms=round(duration_ms, 2),
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
                **self.context
            )


# Initialize logging when module is imported
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"Failed to setup structured logging: {e}")