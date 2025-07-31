"""
Unit tests for the Memory Module.

Tests the core functionality of episodic, semantic, and procedural memory.
"""

import uuid
from datetime import datetime

import pytest

from konfig.modules.memory.memory_module import MemoryModule


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryModule:
    """Test cases for Memory Module functionality."""
    
    async def test_create_job(self, memory_module, clean_database):
        """Test creating a new integration job."""
        job_id = uuid.uuid4()
        
        job = await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor",
            okta_domain="test.okta.com",
            initial_state={"test": "data"},
            metadata={"version": "1.0"}
        )
        
        assert job.job_id == job_id
        assert job.input_url == "https://testvendor.com/saml"
        assert job.vendor_name == "TestVendor"
        assert job.okta_domain == "test.okta.com"
        assert job.status == "pending"
        assert job.state_snapshot == {"test": "data"}
        assert job.job_metadata == {"version": "1.0"}
    
    async def test_get_job(self, memory_module, clean_database):
        """Test retrieving a job by ID."""
        job_id = uuid.uuid4()
        
        # Create job
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        # Retrieve job
        retrieved_job = await memory_module.get_job(job_id)
        
        assert retrieved_job is not None
        assert retrieved_job.job_id == job_id
        assert retrieved_job.vendor_name == "TestVendor"
    
    async def test_get_nonexistent_job(self, memory_module, clean_database):
        """Test retrieving a job that doesn't exist."""
        job_id = uuid.uuid4()
        job = await memory_module.get_job(job_id)
        assert job is None
    
    async def test_update_job_status(self, memory_module, clean_database):
        """Test updating job status."""
        job_id = uuid.uuid4()
        
        # Create job
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        # Update status
        success = await memory_module.update_job_status(
            job_id=job_id,
            status="running",
            error_message=None
        )
        
        assert success is True
        
        # Verify update
        job = await memory_module.get_job(job_id)
        assert job.status == "running"
    
    async def test_store_and_load_job_state(self, memory_module, clean_database):
        """Test storing and loading job state."""
        job_id = uuid.uuid4()
        
        # Create job
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        # Store state
        test_state = {
            "current_step": 5,
            "working_memory": {"app_id": "abc123"},
            "context": "browser"
        }
        
        success = await memory_module.store_job_state(job_id, test_state)
        assert success is True
        
        # Load state
        loaded_state = await memory_module.load_job_state(job_id)
        assert loaded_state == test_state
    
    async def test_list_jobs(self, memory_module, clean_database):
        """Test listing jobs with filters."""
        # Create test jobs
        job1_id = uuid.uuid4()
        job2_id = uuid.uuid4()
        
        await memory_module.create_job(
            job_id=job1_id,
            input_url="https://vendor1.com/saml",
            vendor_name="Vendor1"
        )
        
        await memory_module.create_job(
            job_id=job2_id,
            input_url="https://vendor2.com/saml",
            vendor_name="Vendor2"
        )
        
        # Update one job status
        await memory_module.update_job_status(job1_id, "completed_success")
        
        # List all jobs
        all_jobs = await memory_module.list_jobs()
        assert len(all_jobs) == 2
        
        # List by status
        completed_jobs = await memory_module.list_jobs(status="completed_success")
        assert len(completed_jobs) == 1
        assert completed_jobs[0].job_id == job1_id
        
        # List with limit
        limited_jobs = await memory_module.list_jobs(limit=1)
        assert len(limited_jobs) == 1
    
    async def test_store_trace(self, memory_module, clean_database):
        """Test storing execution traces."""
        job_id = uuid.uuid4()
        
        # Create job
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        # Store trace
        trace = await memory_module.store_trace(
            job_id=job_id,
            trace_type="thought",
            content={"reasoning": "Need to create Okta app"},
            status="success",
            duration_ms=150,
            context={"step": 1}
        )
        
        assert trace.job_id == job_id
        assert trace.trace_type == "thought"
        assert trace.content == {"reasoning": "Need to create Okta app"}
        assert trace.status == "success"
        assert trace.duration_ms == 150
        assert trace.context == {"step": 1}
    
    async def test_get_job_traces(self, memory_module, clean_database):
        """Test retrieving job traces."""
        job_id = uuid.uuid4()
        
        # Create job
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        # Store multiple traces
        await memory_module.store_trace(
            job_id=job_id,
            trace_type="thought",
            content={"reasoning": "First thought"},
            status="success"
        )
        
        await memory_module.store_trace(
            job_id=job_id,
            trace_type="tool_call",
            content={"tool": "OktaAPIClient", "action": "create_saml_app"},
            status="success"
        )
        
        await memory_module.store_trace(
            job_id=job_id,
            trace_type="observation",
            content={"result": "App created successfully"},
            status="success"
        )
        
        # Get all traces
        all_traces = await memory_module.get_job_traces(job_id)
        assert len(all_traces) == 3
        
        # Get traces by type
        thought_traces = await memory_module.get_job_traces(job_id, trace_type="thought")
        assert len(thought_traces) == 1
        assert thought_traces[0].content == {"reasoning": "First thought"}
    
    async def test_store_knowledge_chunks(self, memory_module, clean_database):
        """Test storing knowledge chunks."""
        job_id = uuid.uuid4()
        
        # Create job
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        # Prepare chunks
        chunks = [
            {
                "text": "Configure SAML SSO by navigating to Settings > SSO",
                "metadata": {"source": "section1", "section_type": "configuration"},
                "content_type": "configuration",
                "chunk_index": 0
            },
            {
                "text": "Enter your Entity ID in the designated field",
                "metadata": {"source": "section2", "section_type": "configuration"},
                "content_type": "configuration", 
                "chunk_index": 1
            }
        ]
        
        # Store chunks
        stored_chunks = await memory_module.store_knowledge_chunks(job_id, chunks)
        
        assert len(stored_chunks) == 2
        assert stored_chunks[0].chunk_text == chunks[0]["text"]
        assert stored_chunks[0].content_type == "configuration"
        assert stored_chunks[1].chunk_index == 1
    
    async def test_search_knowledge(self, memory_module, clean_database):
        """Test semantic knowledge search."""
        job_id = uuid.uuid4()
        
        # Create job
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        # Store knowledge chunks
        chunks = [
            {
                "text": "To configure SAML SSO, first navigate to the Settings page",
                "metadata": {"section": "setup"},
                "content_type": "configuration",
                "chunk_index": 0
            },
            {
                "text": "Click the SSO tab and enable SAML authentication",
                "metadata": {"section": "setup"},
                "content_type": "configuration",
                "chunk_index": 1
            },
            {
                "text": "For troubleshooting login issues, check the error logs",
                "metadata": {"section": "troubleshooting"},
                "content_type": "troubleshooting",
                "chunk_index": 2
            }
        ]
        
        await memory_module.store_knowledge_chunks(job_id, chunks)
        
        # Search for configuration-related content
        results = await memory_module.search_knowledge(
            job_id=job_id,
            query="how to setup SAML configuration",
            limit=2,
            similarity_threshold=0.1  # Lower threshold for test
        )
        
        # Should find relevant chunks
        assert len(results) >= 1
        chunk, similarity = results[0]
        assert "SAML" in chunk.chunk_text or "SSO" in chunk.chunk_text
        assert 0 <= similarity <= 1
    
    async def test_store_pattern(self, memory_module, clean_database):
        """Test storing procedural memory patterns."""
        pattern = await memory_module.store_pattern(
            vendor_domain="testvendor.com",
            context_hash="login_form",
            successful_action={"tool": "WebInteractor", "action": "click", "selector": "#login-btn"},
            failed_action={"tool": "WebInteractor", "action": "click", "selector": "#signin-btn"},
            pattern_type="error_recovery",
            description="Use #login-btn instead of #signin-btn for login",
            confidence_score=0.9,
            tags=["login", "button_selector"]
        )
        
        assert pattern.vendor_domain == "testvendor.com"
        assert pattern.context_hash == "login_form"
        assert pattern.pattern_type == "error_recovery"
        assert pattern.confidence_score == 0.9
        assert pattern.successful_action["selector"] == "#login-btn"
    
    async def test_find_patterns(self, memory_module, clean_database):
        """Test finding procedural memory patterns."""
        # Store test patterns
        await memory_module.store_pattern(
            vendor_domain="testvendor.com",
            context_hash="login_form",
            successful_action={"selector": "#login-btn"},
            confidence_score=0.9
        )
        
        await memory_module.store_pattern(
            vendor_domain="testvendor.com",
            context_hash="config_form",
            successful_action={"selector": "#save-config"},
            confidence_score=0.8
        )
        
        await memory_module.store_pattern(
            vendor_domain="othervendor.com",
            context_hash="login_form",
            successful_action={"selector": "#signin"},
            confidence_score=0.7
        )
        
        # Find patterns for testvendor.com
        patterns = await memory_module.find_patterns(
            vendor_domain="testvendor.com",
            min_confidence=0.5
        )
        
        assert len(patterns) == 2
        assert all(p.vendor_domain == "testvendor.com" for p in patterns)
        
        # Find specific context patterns
        login_patterns = await memory_module.find_patterns(
            vendor_domain="testvendor.com",
            context_hash="login_form"
        )
        
        assert len(login_patterns) == 1
        assert login_patterns[0].context_hash == "login_form"
    
    async def test_update_pattern_usage(self, memory_module, clean_database):
        """Test updating pattern usage statistics."""
        # Store pattern
        pattern = await memory_module.store_pattern(
            vendor_domain="testvendor.com",
            context_hash="test_context",
            successful_action={"test": "action"},
            confidence_score=0.8
        )
        
        initial_usage_count = pattern.usage_count
        initial_success_count = pattern.success_count
        initial_confidence = pattern.confidence_score
        
        # Update usage (successful)
        success = await memory_module.update_pattern_usage(pattern.pattern_id, success=True)
        assert success is True
        
        # Verify update
        updated_patterns = await memory_module.find_patterns(
            vendor_domain="testvendor.com",
            context_hash="test_context"
        )
        
        updated_pattern = updated_patterns[0]
        assert updated_pattern.usage_count == initial_usage_count + 1
        assert updated_pattern.success_count == initial_success_count + 1
        assert updated_pattern.confidence_score >= initial_confidence  # Should increase or stay same
    
    async def test_get_memory_statistics(self, memory_module, clean_database):
        """Test getting memory usage statistics."""
        # Create some test data
        job_id = uuid.uuid4()
        await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor"
        )
        
        await memory_module.store_trace(
            job_id=job_id,
            trace_type="thought",
            content={"test": "trace"},
            status="success"
        )
        
        await memory_module.store_pattern(
            vendor_domain="testvendor.com",
            context_hash="test",
            successful_action={"test": "action"},
            confidence_score=0.9
        )
        
        # Get statistics
        stats = await memory_module.get_memory_statistics()
        
        assert "total_jobs" in stats
        assert "total_traces" in stats
        assert "total_patterns" in stats
        assert "table_sizes" in stats
        
        assert stats["total_jobs"] >= 1
        assert stats["total_traces"] >= 1
        assert stats["total_patterns"] >= 1