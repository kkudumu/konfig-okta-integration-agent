"""
Comprehensive execution tracing and observability utilities.

This module provides distributed tracing capabilities following OpenTelemetry
standards for complete observability of the agent's execution.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from konfig.modules.memory.memory_module import MemoryModule
from konfig.utils.logging import get_logger

logger = get_logger(__name__, "tracing")


@dataclass
class TraceSpan:
    """Represents a single trace span."""
    
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"  # running, success, error
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def finish(self, status: str = "success", error: Optional[str] = None):
        """Mark the span as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        if error:
            self.error = error
            self.status = "error"
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for storage."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
            "error": self.error
        }


class TraceContext:
    """Thread-local trace context."""
    
    def __init__(self):
        self.current_trace_id: Optional[str] = None
        self.current_span: Optional[TraceSpan] = None
        self.active_spans: List[TraceSpan] = []
        self.job_id: Optional[str] = None
    
    def start_trace(self, trace_id: str, job_id: Optional[str] = None):
        """Start a new trace context."""
        self.current_trace_id = trace_id
        self.job_id = job_id
        self.active_spans = []
    
    def push_span(self, span: TraceSpan):
        """Push a new span onto the context stack."""
        span.trace_id = self.current_trace_id or span.span_id
        if self.current_span:
            span.parent_span_id = self.current_span.span_id
        
        self.current_span = span
        self.active_spans.append(span)
    
    def pop_span(self) -> Optional[TraceSpan]:
        """Pop the current span from the context stack."""
        if not self.active_spans:
            return None
        
        span = self.active_spans.pop()
        self.current_span = self.active_spans[-1] if self.active_spans else None
        return span


# Global trace context (in a real implementation, this would be thread-local)
_trace_context = TraceContext()


class ExecutionTracer:
    """Main execution tracer for the Konfig system."""
    
    def __init__(self):
        self.memory_module = MemoryModule()
        self.active_traces: Dict[str, List[TraceSpan]] = {}
        
    def start_trace(self, job_id: str, operation_name: str = "integration") -> str:
        """Start a new trace for a job."""
        trace_id = str(uuid.uuid4())
        _trace_context.start_trace(trace_id, job_id)
        
        # Create root span
        root_span = TraceSpan(
            span_id=trace_id,
            trace_id=trace_id,
            operation_name=operation_name
        )
        root_span.add_tag("job_id", job_id)
        root_span.add_tag("service", "konfig")
        root_span.add_tag("component", "orchestrator")
        
        _trace_context.push_span(root_span)
        self.active_traces[trace_id] = [root_span]
        
        logger.info("Trace started", trace_id=trace_id, job_id=job_id)
        return trace_id
    
    async def finish_trace(self, trace_id: str):
        """Finish a trace and store all spans."""
        if trace_id not in self.active_traces:
            logger.warning("Attempted to finish unknown trace", trace_id=trace_id)
            return
        
        spans = self.active_traces.pop(trace_id)
        
        # Finish any remaining active spans
        for span in spans:
            if span.status == "running":
                span.finish()
        
        # Store traces in memory
        job_id = _trace_context.job_id
        if job_id:
            try:
                job_uuid = uuid.UUID(job_id)
                for span in spans:
                    await self.memory_module.store_trace(
                        job_id=job_uuid,
                        trace_type="execution_span",
                        content=span.to_dict(),
                        status="success" if span.status != "error" else "failure",
                        duration_ms=int(span.duration_ms) if span.duration_ms else None,
                        error_details=span.error
                    )
                
                logger.info("Trace finished and stored", trace_id=trace_id, num_spans=len(spans))
            except Exception as e:
                logger.error("Failed to store trace", trace_id=trace_id, error=str(e))
    
    @contextmanager
    def span(self, operation_name: str, **tags):
        """Context manager for creating spans."""
        span = TraceSpan(operation_name=operation_name)
        
        # Add provided tags
        for key, value in tags.items():
            span.add_tag(key, value)
        
        _trace_context.push_span(span)
        
        try:
            yield span
            span.finish("success")
        except Exception as e:
            span.finish("error", str(e))
            raise
        finally:
            _trace_context.pop_span()
            
            # Add to active traces
            if _trace_context.current_trace_id in self.active_traces:
                self.active_traces[_trace_context.current_trace_id].append(span)
    
    @asynccontextmanager
    async def async_span(self, operation_name: str, **tags):
        """Async context manager for creating spans."""
        span = TraceSpan(operation_name=operation_name)
        
        # Add provided tags
        for key, value in tags.items():
            span.add_tag(key, value)
        
        _trace_context.push_span(span)
        
        try:
            yield span
            span.finish("success")
        except Exception as e:
            span.finish("error", str(e))
            raise
        finally:
            _trace_context.pop_span()
            
            # Add to active traces
            if _trace_context.current_trace_id in self.active_traces:
                self.active_traces[_trace_context.current_trace_id].append(span)
    
    def current_span(self) -> Optional[TraceSpan]:
        """Get the current active span."""
        return _trace_context.current_span
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the current span."""
        if _trace_context.current_span:
            _trace_context.current_span.add_tag(key, value)
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log to the current span."""
        if _trace_context.current_span:
            _trace_context.current_span.add_log(message, level, **kwargs)


# Global tracer instance
tracer = ExecutionTracer()


# Convenience functions
def start_trace(job_id: str, operation_name: str = "integration") -> str:
    """Start a new trace for a job."""
    return tracer.start_trace(job_id, operation_name)


async def finish_trace(trace_id: str):
    """Finish a trace and store all spans."""
    await tracer.finish_trace(trace_id)


def span(operation_name: str, **tags):
    """Create a span context manager."""
    return tracer.span(operation_name, **tags)


def async_span(operation_name: str, **tags):
    """Create an async span context manager."""
    return tracer.async_span(operation_name, **tags)


def current_span() -> Optional[TraceSpan]:
    """Get the current active span."""
    return tracer.current_span()


def add_tag(key: str, value: Any):
    """Add a tag to the current span."""
    tracer.add_tag(key, value)


def add_log(message: str, level: str = "info", **kwargs):
    """Add a log to the current span."""
    tracer.add_log(message, level, **kwargs)


# Decorators for automatic tracing
def trace_method(operation_name: Optional[str] = None):
    """Decorator to automatically trace method calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with span(op_name):
                # Add method arguments as tags (excluding sensitive data)
                safe_kwargs = {}
                for k, v in kwargs.items():
                    if not any(sensitive in k.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                        safe_kwargs[k] = str(v)[:100]  # Limit length
                
                add_tag("method", func.__name__)
                add_tag("module", func.__module__)
                if safe_kwargs:
                    add_tag("arguments", safe_kwargs)
                
                try:
                    result = func(*args, **kwargs)
                    add_tag("success", True)
                    return result
                except Exception as e:
                    add_tag("success", False)
                    add_tag("error", str(e))
                    raise
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def trace_async_method(operation_name: Optional[str] = None):
    """Decorator to automatically trace async method calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            async with async_span(op_name):
                # Add method arguments as tags (excluding sensitive data)
                safe_kwargs = {}
                for k, v in kwargs.items():
                    if not any(sensitive in k.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                        safe_kwargs[k] = str(v)[:100]  # Limit length
                
                add_tag("method", func.__name__)
                add_tag("module", func.__module__)
                if safe_kwargs:
                    add_tag("arguments", safe_kwargs)
                
                try:
                    result = await func(*args, **kwargs)
                    add_tag("success", True)
                    return result
                except Exception as e:
                    add_tag("success", False)
                    add_tag("error", str(e))
                    raise
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


class TracingMixin:
    """Mixin to add tracing capabilities to any class."""
    
    def trace_method_call(self, method_name: str, **kwargs):
        """Trace a method call with parameters."""
        span_name = f"{self.__class__.__name__}.{method_name}"
        
        with span(span_name):
            add_tag("class", self.__class__.__name__)
            add_tag("method", method_name)
            
            # Add safe parameters
            safe_params = {}
            for k, v in kwargs.items():
                if not any(sensitive in k.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                    safe_params[k] = str(v)[:100] if v is not None else None
            
            if safe_params:
                add_tag("parameters", safe_params)
    
    async def trace_async_method_call(self, method_name: str, **kwargs):
        """Trace an async method call with parameters."""
        span_name = f"{self.__class__.__name__}.{method_name}"
        
        async with async_span(span_name):
            add_tag("class", self.__class__.__name__)
            add_tag("method", method_name)
            
            # Add safe parameters
            safe_params = {}
            for k, v in kwargs.items():
                if not any(sensitive in k.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                    safe_params[k] = str(v)[:100] if v is not None else None
            
            if safe_params:
                add_tag("parameters", safe_params)
    
    def trace_result(self, result: Any, duration_ms: Optional[float] = None):
        """Trace method result."""
        add_tag("result_type", type(result).__name__)
        
        if duration_ms:
            add_tag("duration_ms", duration_ms)
        
        # Add result summary (avoid logging large objects)
        if isinstance(result, (dict, list)):
            add_tag("result_size", len(result))
        elif isinstance(result, str):
            add_tag("result_length", len(result))
    
    def trace_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Trace an error with context."""
        add_tag("error_type", type(error).__name__)
        add_tag("error_message", str(error))
        
        if context:
            add_tag("error_context", context)
        
        add_log(f"Error occurred: {error}", level="error")


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance metrics during execution."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record_duration(self, operation: str, duration_ms: float):
        """Record operation duration."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration_ms)
        
        # Also add to current span if active
        add_tag(f"{operation}_duration_ms", duration_ms)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = self.metrics[operation]
        return {
            "count": len(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations)
        }
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.metrics.clear()


# Global performance monitor
performance_monitor = PerformanceMonitor()


@contextmanager
def monitor_performance(operation: str):
    """Context manager to monitor operation performance."""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        performance_monitor.record_duration(operation, duration_ms)