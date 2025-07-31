"""
Metrics collection and monitoring utilities for Konfig.

This module provides comprehensive metrics collection following Prometheus
standards for monitoring agent performance and health.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from konfig.config.settings import get_settings
from konfig.utils.logging import get_logger

logger = get_logger(__name__, "metrics")


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Represents a metric value with timestamp and labels."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Base metric class."""
    name: str
    metric_type: MetricType
    help_text: str
    labels: Set[str] = field(default_factory=set)
    values: List[MetricValue] = field(default_factory=list)
    
    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a value to this metric."""
        metric_value = MetricValue(value=value, labels=labels or {})
        self.values.append(metric_value)
        
        # Keep only recent values to prevent memory growth
        if len(self.values) > 1000:
            self.values = self.values[-500:]  # Keep last 500 values


class Counter(Metric):
    """Counter metric that only increases."""
    
    def __init__(self, name: str, help_text: str):
        super().__init__(name, MetricType.COUNTER, help_text)
        self._value = 0.0
        self._lock = Lock()
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the counter."""
        with self._lock:
            self._value += amount
            self.add_value(self._value, labels)
    
    def get_value(self) -> float:
        """Get current counter value."""
        return self._value


class Gauge(Metric):
    """Gauge metric that can go up and down."""
    
    def __init__(self, name: str, help_text: str):
        super().__init__(name, MetricType.GAUGE, help_text)
        self._value = 0.0
        self._lock = Lock()
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set the gauge value."""
        with self._lock:
            self._value = value
            self.add_value(self._value, labels)
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the gauge."""
        with self._lock:
            self._value += amount
            self.add_value(self._value, labels)
    
    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount
            self.add_value(self._value, labels)
    
    def get_value(self) -> float:
        """Get current gauge value."""
        return self._value


class Histogram(Metric):
    """Histogram metric for measuring distributions."""
    
    def __init__(self, name: str, help_text: str, buckets: Optional[List[float]] = None):
        super().__init__(name, MetricType.HISTOGRAM, help_text)
        self.buckets = buckets or [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        self._observations: List[float] = []
        self._lock = Lock()
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        with self._lock:
            self._observations.append(value)
            self.add_value(value, labels)
            
            # Keep only recent observations
            if len(self._observations) > 10000:
                self._observations = self._observations[-5000:]
    
    def get_bucket_counts(self) -> Dict[float, int]:
        """Get counts for each bucket."""
        counts = {}
        for bucket in self.buckets:
            counts[bucket] = sum(1 for obs in self._observations if obs <= bucket)
        counts[float('inf')] = len(self._observations)
        return counts
    
    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        if not self._observations:
            return {}
        
        sorted_obs = sorted(self._observations)
        n = len(sorted_obs)
        
        return {
            "count": n,
            "sum": sum(sorted_obs),
            "avg": sum(sorted_obs) / n,
            "min": sorted_obs[0],
            "max": sorted_obs[-1],
            "p50": sorted_obs[int(n * 0.5)],
            "p95": sorted_obs[int(n * 0.95)],
            "p99": sorted_obs[int(n * 0.99)]
        }


class MetricsRegistry:
    """Registry for all metrics in the system."""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self._lock = Lock()
        self.settings = get_settings()
        
        # Initialize default metrics
        self._init_default_metrics()
    
    def _init_default_metrics(self):
        """Initialize default system metrics."""
        # Job metrics
        self.register_counter("konfig_jobs_total", "Total number of integration jobs")
        self.register_counter("konfig_jobs_successful", "Number of successful jobs")
        self.register_counter("konfig_jobs_failed", "Number of failed jobs")
        self.register_gauge("konfig_jobs_active", "Number of currently active jobs")
        self.register_histogram("konfig_job_duration_seconds", "Job execution duration")
        
        # Tool metrics
        self.register_counter("konfig_tool_calls_total", "Total number of tool calls")
        self.register_counter("konfig_tool_errors_total", "Total number of tool errors")
        self.register_histogram("konfig_tool_duration_seconds", "Tool execution duration")
        
        # API metrics
        self.register_counter("konfig_okta_api_calls_total", "Total Okta API calls")
        self.register_counter("konfig_okta_api_errors_total", "Okta API errors")
        self.register_histogram("konfig_okta_api_duration_seconds", "Okta API call duration")
        
        # Browser metrics
        self.register_counter("konfig_browser_actions_total", "Total browser actions")
        self.register_counter("konfig_browser_errors_total", "Browser action errors")
        self.register_histogram("konfig_browser_action_duration_seconds", "Browser action duration")
        
        # LLM metrics
        self.register_counter("konfig_llm_requests_total", "Total LLM requests")
        self.register_counter("konfig_llm_tokens_total", "Total LLM tokens used")
        self.register_histogram("konfig_llm_duration_seconds", "LLM request duration")
        
        # System metrics
        self.register_gauge("konfig_memory_usage_bytes", "Memory usage in bytes")
        self.register_gauge("konfig_database_connections", "Active database connections")
        self.register_counter("konfig_errors_total", "Total system errors")
        
        # HITL metrics
        self.register_counter("konfig_hitl_requests_total", "Total HITL requests")
        self.register_histogram("konfig_hitl_response_time_seconds", "HITL response time")
        
        logger.info("Default metrics initialized")
    
    def register_counter(self, name: str, help_text: str) -> Counter:
        """Register a new counter metric."""
        with self._lock:
            if name in self.metrics:
                if not isinstance(self.metrics[name], Counter):
                    raise ValueError(f"Metric {name} already exists with different type")
                return self.metrics[name]
            
            counter = Counter(name, help_text)
            self.metrics[name] = counter
            return counter
    
    def register_gauge(self, name: str, help_text: str) -> Gauge:
        """Register a new gauge metric."""
        with self._lock:
            if name in self.metrics:
                if not isinstance(self.metrics[name], Gauge):
                    raise ValueError(f"Metric {name} already exists with different type")
                return self.metrics[name]
            
            gauge = Gauge(name, help_text)
            self.metrics[name] = gauge
            return gauge
    
    def register_histogram(self, name: str, help_text: str, buckets: Optional[List[float]] = None) -> Histogram:
        """Register a new histogram metric."""
        with self._lock:
            if name in self.metrics:
                if not isinstance(self.metrics[name], Histogram):
                    raise ValueError(f"Metric {name} already exists with different type")
                return self.metrics[name]
            
            histogram = Histogram(name, help_text, buckets)
            self.metrics[name] = histogram
            return histogram
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        return self.metrics.copy()
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric in self.metrics.values():
            # Add help text
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")
            
            if isinstance(metric, (Counter, Gauge)):
                value = metric.get_value()
                lines.append(f"{metric.name} {value}")
            
            elif isinstance(metric, Histogram):
                stats = metric.get_stats()
                if stats:
                    # Export histogram buckets
                    bucket_counts = metric.get_bucket_counts()
                    for bucket, count in bucket_counts.items():
                        bucket_label = "+Inf" if bucket == float('inf') else str(bucket)
                        lines.append(f"{metric.name}_bucket{{le=\"{bucket_label}\"}} {count}")
                    
                    # Export sum and count
                    lines.append(f"{metric.name}_sum {stats['sum']}")
                    lines.append(f"{metric.name}_count {stats['count']}")
            
            lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)
    
    def export_json(self) -> Dict[str, Any]:
        """Export metrics in JSON format."""
        result = {}
        
        for name, metric in self.metrics.items():
            if isinstance(metric, (Counter, Gauge)):
                result[name] = {
                    "type": metric.metric_type.value,
                    "value": metric.get_value(),
                    "help": metric.help_text
                }
            elif isinstance(metric, Histogram):
                stats = metric.get_stats()
                result[name] = {
                    "type": metric.metric_type.value,
                    "stats": stats,
                    "buckets": metric.get_bucket_counts(),
                    "help": metric.help_text
                }
        
        return result


# Global metrics registry
metrics_registry = MetricsRegistry()


# Convenience functions for common metrics
def job_started(job_id: str, vendor: str = ""):
    """Record that a job has started."""
    metrics_registry.get_metric("konfig_jobs_total").inc(labels={"vendor": vendor})
    metrics_registry.get_metric("konfig_jobs_active").inc()
    logger.debug("Job started metric recorded", job_id=job_id, vendor=vendor)


def job_completed(job_id: str, success: bool, duration_seconds: float, vendor: str = ""):
    """Record that a job has completed."""
    labels = {"vendor": vendor, "status": "success" if success else "failure"}
    
    if success:
        metrics_registry.get_metric("konfig_jobs_successful").inc(labels=labels)
    else:
        metrics_registry.get_metric("konfig_jobs_failed").inc(labels=labels)
    
    metrics_registry.get_metric("konfig_jobs_active").dec()
    metrics_registry.get_metric("konfig_job_duration_seconds").observe(duration_seconds, labels)
    
    logger.debug("Job completed metric recorded", job_id=job_id, success=success, duration=duration_seconds)


def tool_call_started(tool_name: str, action: str):
    """Record that a tool call has started."""
    labels = {"tool": tool_name, "action": action}
    metrics_registry.get_metric("konfig_tool_calls_total").inc(labels=labels)


def tool_call_completed(tool_name: str, action: str, success: bool, duration_seconds: float):
    """Record that a tool call has completed."""
    labels = {"tool": tool_name, "action": action, "status": "success" if success else "error"}
    
    if not success:
        metrics_registry.get_metric("konfig_tool_errors_total").inc(labels=labels)
    
    metrics_registry.get_metric("konfig_tool_duration_seconds").observe(duration_seconds, labels)


def okta_api_call(endpoint: str, method: str, status_code: int, duration_seconds: float):
    """Record an Okta API call."""
    labels = {"endpoint": endpoint, "method": method, "status_code": str(status_code)}
    
    metrics_registry.get_metric("konfig_okta_api_calls_total").inc(labels=labels)
    
    if status_code >= 400:
        metrics_registry.get_metric("konfig_okta_api_errors_total").inc(labels=labels)
    
    metrics_registry.get_metric("konfig_okta_api_duration_seconds").observe(duration_seconds, labels)


def browser_action(action_type: str, success: bool, duration_seconds: float):
    """Record a browser action."""
    labels = {"action": action_type, "status": "success" if success else "error"}
    
    metrics_registry.get_metric("konfig_browser_actions_total").inc(labels=labels)
    
    if not success:
        metrics_registry.get_metric("konfig_browser_errors_total").inc(labels=labels)
    
    metrics_registry.get_metric("konfig_browser_action_duration_seconds").observe(duration_seconds, labels)


def llm_request(provider: str, model: str, tokens_used: int, duration_seconds: float):
    """Record an LLM request."""
    labels = {"provider": provider, "model": model}
    
    metrics_registry.get_metric("konfig_llm_requests_total").inc(labels=labels)
    metrics_registry.get_metric("konfig_llm_tokens_total").inc(amount=tokens_used, labels=labels)
    metrics_registry.get_metric("konfig_llm_duration_seconds").observe(duration_seconds, labels)


def hitl_request(reason: str, response_time_seconds: Optional[float] = None):
    """Record a HITL request."""
    labels = {"reason": reason}
    metrics_registry.get_metric("konfig_hitl_requests_total").inc(labels=labels)
    
    if response_time_seconds is not None:
        metrics_registry.get_metric("konfig_hitl_response_time_seconds").observe(response_time_seconds, labels)


def system_error(component: str, error_type: str):
    """Record a system error."""
    labels = {"component": component, "error_type": error_type}
    metrics_registry.get_metric("konfig_errors_total").inc(labels=labels)


def update_system_gauge(metric_name: str, value: float):
    """Update a system gauge metric."""
    gauge = metrics_registry.get_metric(metric_name)
    if gauge and isinstance(gauge, Gauge):
        gauge.set(value)


class MetricsCollector:
    """Background metrics collector for system stats."""
    
    def __init__(self):
        self.running = False
    
    async def start_collection(self):
        """Start collecting system metrics."""
        self.running = True
        logger.info("Metrics collection started")
        
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error("Error collecting metrics", error=str(e))
                await asyncio.sleep(10)
    
    def stop_collection(self):
        """Stop collecting metrics."""
        self.running = False
        logger.info("Metrics collection stopped")
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        import psutil
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        update_system_gauge("konfig_memory_usage_bytes", memory_info.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        update_system_gauge("konfig_cpu_usage_percent", cpu_percent)
        
        # Database connections (would need actual implementation)
        # update_system_gauge("konfig_database_connections", db_connection_count)