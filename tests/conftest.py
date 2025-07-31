"""
Pytest configuration and fixtures for Konfig tests.

This module provides shared fixtures and configuration for all test types:
unit, integration, and end-to-end tests.
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "postgresql://konfig:konfig_dev_password@localhost:5432/konfig_test"

from konfig.config.settings import get_settings
from konfig.database.connection import DatabaseManager
from konfig.database.models import Base
from konfig.modules.memory.memory_module import MemoryModule
from konfig.tools.web_interactor import WebInteractor
from konfig.tools.okta_api_client import OktaAPIClient
from konfig.modules.cognition.cognition_module import CognitionModule
from konfig.modules.perception.perception_module import PerceptionModule
from konfig.orchestrator.react_orchestrator import ReActOrchestrator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings():
    """Get test settings."""
    return get_settings()


@pytest.fixture(scope="session")
async def test_database():
    """Set up test database."""
    # Create test database engine
    settings = get_settings()
    db_url = str(settings.database.url).replace("konfig", "konfig_test")
    
    # Create sync engine for setup
    sync_engine = create_engine(db_url.replace("postgresql+asyncpg://", "postgresql://"))
    
    # Create all tables
    Base.metadata.create_all(sync_engine)
    
    yield db_url
    
    # Cleanup
    Base.metadata.drop_all(sync_engine)
    sync_engine.dispose()


@pytest.fixture
async def db_manager(test_database):
    """Database manager for tests."""
    manager = DatabaseManager()
    # Override URL for test database
    manager.settings.database.url = test_database
    await manager.initialize_database()
    yield manager
    await manager.close()


@pytest.fixture
async def memory_module(db_manager):
    """Memory module instance for tests."""
    memory = MemoryModule()
    yield memory


@pytest.fixture
def sample_job_id():
    """Generate a sample job ID for tests."""
    return uuid.uuid4()


@pytest.fixture
def sample_integration_data():
    """Sample integration data for tests."""
    return {
        "vendor_name": "TestVendor",
        "documentation_url": "https://testvendor.com/saml-setup",
        "okta_domain": "test.okta.com",
        "app_name": "TestVendor SAML App"
    }


@pytest.fixture
async def web_interactor():
    """WebInteractor instance for tests."""
    # Use headless mode for tests
    interactor = WebInteractor()
    interactor.headless = True
    yield interactor
    
    # Cleanup
    if interactor._session_active:
        await interactor.close_browser()


@pytest.fixture
def mock_okta_client():
    """Mock Okta API client for tests."""
    class MockOktaAPIClient:
        def __init__(self, *args, **kwargs):
            self.okta_domain = "test.okta.com"
            self.api_token = "mock_token"
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
        
        async def create_saml_app(self, label, settings=None):
            return {
                "id": "mock_app_123",
                "label": label,
                "status": "ACTIVE",
                "settings": {
                    "signOn": {
                        "ssoAcsUrl": None,
                        "audience": None
                    }
                }
            }
        
        async def get_app_metadata(self, app_id):
            return {
                "app_id": app_id,
                "entity_id": f"http://www.okta.com/{app_id}",
                "sso_url": f"https://test.okta.com/app/{app_id}/sso/saml",
                "certificate": "mock_certificate_data"
            }
        
        async def test_connection(self):
            return {
                "success": True,
                "org_id": "mock_org_123",
                "org_name": "Test Organization"
            }
    
    return MockOktaAPIClient


@pytest.fixture
def sample_html_content():
    """Sample HTML content for parsing tests."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TestVendor SAML Configuration Guide</title>
    </head>
    <body>
        <main>
            <h1>SAML SSO Setup</h1>
            <p>This guide explains how to configure SAML SSO with TestVendor.</p>
            
            <h2>Prerequisites</h2>
            <ul>
                <li>Admin access to TestVendor</li>
                <li>Identity Provider metadata</li>
            </ul>
            
            <h2>Configuration Steps</h2>
            <ol>
                <li>Navigate to Settings > SSO</li>
                <li>Click "Enable SAML"</li>
                <li>Enter the Entity ID: <code>your-entity-id</code></li>
                <li>Paste the SSO URL in the designated field</li>
                <li>Upload the X.509 certificate</li>
            </ol>
            
            <h2>Testing</h2>
            <p>Test the configuration by logging in through the SSO portal.</p>
        </main>
    </body>
    </html>
    """


@pytest.fixture
def sample_documentation():
    """Sample processed documentation for tests."""
    return {
        "url": "https://testvendor.com/saml-setup",
        "title": "TestVendor SAML Configuration Guide",
        "vendor_name": "TestVendor",
        "sections": [
            {
                "title": "SAML SSO Setup",
                "content": "This guide explains how to configure SAML SSO with TestVendor.",
                "level": 1,
                "section_type": "overview",
                "subsections": []
            },
            {
                "title": "Configuration Steps",
                "content": "1. Navigate to Settings > SSO\n2. Click 'Enable SAML'",
                "level": 2,
                "section_type": "configuration",
                "subsections": []
            }
        ],
        "saml_info": {
            "metadata_url": None,
            "entity_id": "your-entity-id",
            "acs_url": None,
            "certificate_required": True,
            "configuration_steps": [
                {"number": 1, "description": "Navigate to Settings > SSO"},
                {"number": 2, "description": "Click 'Enable SAML'"}
            ]
        }
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for cognition tests."""
    return {
        "vendor_name": "TestVendor",
        "documentation_url": "https://testvendor.com/saml-setup",
        "plan_version": "1.0",
        "total_steps": 4,
        "steps": [
            {
                "step_id": 1,
                "description": "Create SAML application in Okta",
                "tool_required": "OktaAPIClient",
                "action_to_perform": "create_saml_app",
                "inputs": {"label": "TestVendor SAML"},
                "outputs": {"app_id": "okta_app_id", "sso_url": "sso_url"},
                "dependencies": [],
                "status": "pending",
                "context": "okta_api"
            },
            {
                "step_id": 2,
                "description": "Navigate to vendor admin console",
                "tool_required": "WebInteractor",
                "action_to_perform": "navigate",
                "inputs": {"url": "https://testvendor.com/admin"},
                "outputs": {},
                "dependencies": [1],
                "status": "pending",
                "context": "browser"
            }
        ]
    }


# Pytest markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_browser: Tests that need browser automation")
    config.addinivalue_line("markers", "requires_okta: Tests that need Okta API access")
    config.addinivalue_line("markers", "requires_llm: Tests that need LLM API access")


@pytest.fixture
def mock_http_response():
    """Mock HTTP response for web requests."""
    class MockResponse:
        def __init__(self, text="", status_code=200, headers=None):
            self.text = text
            self.status_code = status_code
            self.headers = headers or {"content-type": "text/html"}
        
        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")
    
    return MockResponse


@pytest.fixture
def temp_screenshot_dir(tmp_path):
    """Temporary directory for screenshots during tests."""
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    return screenshots_dir


@pytest.fixture
async def clean_database(db_manager):
    """Clean database before and after tests."""
    # Clean before test
    async with db_manager.get_async_session() as session:
        # Delete all test data
        from konfig.database.models import IntegrationJob, ExecutionTrace, ProceduralMemoryPattern, KnowledgeBaseVector
        
        await session.execute("DELETE FROM execution_traces")
        await session.execute("DELETE FROM knowledge_base_vectors")
        await session.execute("DELETE FROM procedural_memory_patterns")
        await session.execute("DELETE FROM integration_jobs")
        await session.commit()
    
    yield
    
    # Clean after test
    async with db_manager.get_async_session() as session:
        await session.execute("DELETE FROM execution_traces")
        await session.execute("DELETE FROM knowledge_base_vectors")
        await session.execute("DELETE FROM procedural_memory_patterns")
        await session.execute("DELETE FROM integration_jobs")
        await session.commit()


# Async test utilities
class AsyncMock:
    """Simple async mock for testing."""
    
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0
        self.call_args_list = []
    
    async def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.call_args_list.append((args, kwargs))
        
        if self.side_effect:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        
        return self.return_value


@pytest.fixture
def async_mock():
    """Factory for creating async mocks."""
    return AsyncMock


# Test data factories
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_job_data(vendor_name="TestVendor"):
        return {
            "job_id": uuid.uuid4(),
            "input_url": f"https://{vendor_name.lower()}.com/saml-setup",
            "vendor_name": vendor_name,
            "okta_domain": "test.okta.com"
        }
    
    @staticmethod
    def create_execution_trace(job_id, trace_type="thought"):
        return {
            "job_id": job_id,
            "trace_type": trace_type,
            "content": {"message": f"Test {trace_type}"},
            "status": "success"
        }
    
    @staticmethod
    def create_knowledge_chunk(job_id, chunk_index=0):
        return {
            "text": f"Test knowledge chunk {chunk_index}",
            "metadata": {"source": "test", "chunk_index": chunk_index},
            "content_type": "documentation",
            "chunk_index": chunk_index
        }


@pytest.fixture
def test_factory():
    """Test data factory."""
    return TestDataFactory