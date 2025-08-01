"""
Configuration management for Konfig.

This module provides centralized configuration management using Pydantic
for validation and type safety.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pydantic.networks import AnyHttpUrl, PostgresDsn


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: PostgresDsn = Field(
        default="postgresql://konfig:konfig_dev_password@localhost:5432/konfig",
        env="DATABASE_URL",
        description="PostgreSQL database URL"
    )
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DATABASE_ECHO")

    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    
    class Config:
        env_prefix = "REDIS_"


class OktaSettings(BaseSettings):
    """Okta API configuration settings."""
    
    domain: Optional[str] = Field(default=None, env="OKTA_DOMAIN", description="Okta domain")
    api_token: Optional[str] = Field(default=None, env="OKTA_API_TOKEN", description="Okta API token")
    rate_limit_max_requests: int = Field(default=1000, env="OKTA_RATE_LIMIT_MAX_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, env="OKTA_RATE_LIMIT_WINDOW_SECONDS")
    
    @validator("domain")
    def validate_domain(cls, v: Optional[str]) -> Optional[str]:
        """Validate Okta domain format."""
        if v is not None and not v.endswith(".okta.com") and not v.endswith(".oktapreview.com"):
            raise ValueError("Okta domain must end with .okta.com or .oktapreview.com")
        return v
    
    class Config:
        env_prefix = "OKTA_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # Google Gemini settings
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_max_tokens: int = Field(default=4096, env="GEMINI_MAX_TOKENS")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    
    # LangSmith settings
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_endpoint: Optional[AnyHttpUrl] = Field(default=None, env="LANGCHAIN_ENDPOINT")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="konfig-development", env="LANGCHAIN_PROJECT")
    
    # Note: LLM API key validation is handled at runtime when needed
    # to avoid issues during configuration initialization
    
    class Config:
        env_prefix = ""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class GoogleSettings(BaseSettings):
    """Google Workspace admin configuration settings."""
    
    admin_username: Optional[str] = Field(default=None, env="GOOGLE_ADMIN_USERNAME")
    admin_password: Optional[str] = Field(default=None, env="GOOGLE_ADMIN_PASSWORD")
    admin_domain: Optional[str] = Field(default=None, env="GOOGLE_ADMIN_DOMAIN")
    
    class Config:
        env_prefix = "GOOGLE_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: Optional[str] = Field(default=None, env="SECRET_KEY", min_length=32)
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY", min_length=32)
    jwt_secret_key: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    class Config:
        env_prefix = "SECURITY_"


class VaultSettings(BaseSettings):
    """HashiCorp Vault configuration settings."""
    
    url: AnyHttpUrl = Field(default="http://localhost:8200", env="VAULT_URL")
    token: Optional[str] = Field(default=None, env="VAULT_TOKEN")
    mount_point: str = Field(default="secret", env="VAULT_MOUNT_POINT")
    
    class Config:
        env_prefix = "VAULT_"


class WebSettings(BaseSettings):
    """Web interface configuration settings."""
    
    host: str = Field(default="0.0.0.0", env="WEB_HOST")
    port: int = Field(default=8000, env="WEB_PORT")
    reload: bool = Field(default=False, env="WEB_RELOAD")
    workers: int = Field(default=1, env="WEB_WORKERS")
    
    class Config:
        env_prefix = "WEB_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    file: Optional[Path] = Field(default=None, env="LOG_FILE")
    structured: bool = Field(default=True, env="STRUCTURED_LOGGING")
    
    @validator("level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"


class BrowserSettings(BaseSettings):
    """Browser automation configuration settings."""
    
    headless: bool = Field(default=True, env="BROWSER_HEADLESS")
    timeout_ms: int = Field(default=30000, env="BROWSER_TIMEOUT_MS")
    viewport_width: int = Field(default=1920, env="BROWSER_VIEWPORT_WIDTH")
    viewport_height: int = Field(default=1080, env="BROWSER_VIEWPORT_HEIGHT")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        env="BROWSER_USER_AGENT"
    )
    
    class Config:
        env_prefix = "BROWSER_"


class AgentSettings(BaseSettings):
    """Agent behavior configuration settings."""
    
    max_iterations: int = Field(default=50, env="AGENT_MAX_ITERATIONS")
    timeout_seconds: int = Field(default=1800, env="AGENT_TIMEOUT_SECONDS")
    memory_window: int = Field(default=10, env="AGENT_MEMORY_WINDOW")
    self_healing_enabled: bool = Field(default=True, env="AGENT_SELF_HEALING_ENABLED")
    hitl_enabled: bool = Field(default=True, env="AGENT_HITL_ENABLED")
    
    class Config:
        env_prefix = "AGENT_"


class NotificationSettings(BaseSettings):
    """Notification configuration settings."""
    
    slack_webhook_url: Optional[AnyHttpUrl] = Field(default=None, env="SLACK_WEBHOOK_URL")
    email_smtp_host: str = Field(default="smtp.gmail.com", env="EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(default=587, env="EMAIL_SMTP_PORT")
    email_username: Optional[str] = Field(default=None, env="EMAIL_USERNAME")
    email_password: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")
    email_from_address: str = Field(default="noreply@konfig.ai", env="EMAIL_FROM_ADDRESS")
    
    class Config:
        env_prefix = "NOTIFICATION_"


class PerformanceSettings(BaseSettings):
    """Performance configuration settings."""
    
    max_concurrent_jobs: int = Field(default=5, env="MAX_CONCURRENT_JOBS")
    task_queue_backend: str = Field(default="redis", env="TASK_QUEUE_BACKEND")
    celery_broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    class Config:
        env_prefix = "PERFORMANCE_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration settings."""
    
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8080, env="PROMETHEUS_PORT")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    health_check_enabled: bool = Field(default=True, env="HEALTH_CHECK_ENABLED")
    
    class Config:
        env_prefix = "MONITORING_"


class FeatureFlags(BaseSettings):
    """Feature flags for enabling/disabling functionality."""
    
    learning_module: bool = Field(default=True, env="FEATURE_LEARNING_MODULE")
    self_healing: bool = Field(default=True, env="FEATURE_SELF_HEALING")
    advanced_planning: bool = Field(default=True, env="FEATURE_ADVANCED_PLANNING")
    multi_tenant: bool = Field(default=False, env="FEATURE_MULTI_TENANT")
    
    class Config:
        env_prefix = "FEATURE_"


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Configuration sections
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    okta: OktaSettings = OktaSettings()
    llm: LLMSettings = LLMSettings()
    google: GoogleSettings = GoogleSettings()
    security: SecuritySettings = SecuritySettings()
    vault: VaultSettings = VaultSettings()
    web: WebSettings = WebSettings()
    logging: LoggingSettings = LoggingSettings()
    browser: BrowserSettings = BrowserSettings()
    agent: AgentSettings = AgentSettings()
    notifications: NotificationSettings = NotificationSettings()
    performance: PerformanceSettings = PerformanceSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    features: FeatureFlags = FeatureFlags()
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_envs = {"development", "testing", "staging", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing" or self.testing
    
    class Config:
        env_file = [".env", "konfig/.env"]
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: The application configuration
    """
    return Settings()


# Export commonly used settings
settings = get_settings()