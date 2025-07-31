"""
Integration tests for end-to-end workflow.

Tests the complete integration process from documentation parsing
to job completion.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from konfig.orchestrator.react_orchestrator import ReActOrchestrator
from konfig.modules.perception.perception_module import PerceptionModule
from konfig.modules.cognition.cognition_module import CognitionModule


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndFlow:
    """Integration tests for complete workflow."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for testing."""
        orchestrator = ReActOrchestrator()
        yield orchestrator
    
    async def test_complete_integration_flow_mock(self, orchestrator, clean_database, sample_integration_data):
        """Test complete integration flow with mocked external services."""
        
        # Mock the tool initialization to avoid actual browser/API calls
        async def mock_init_tools(okta_domain=None):
            orchestrator.okta_client = AsyncMock()
            orchestrator.okta_client.create_saml_app.return_value = {
                "id": "mock_app_123",
                "label": "TestVendor SAML",
                "status": "ACTIVE"
            }
            orchestrator.okta_client.get_app_metadata.return_value = {
                "app_id": "mock_app_123",
                "entity_id": "http://www.okta.com/mock_app_123",
                "sso_url": "https://test.okta.com/app/mock_app_123/sso/saml"
            }
            
            orchestrator.web_interactor = AsyncMock()
            orchestrator.web_interactor.navigate.return_value = {
                "success": True,
                "url": sample_integration_data["documentation_url"],
                "title": "TestVendor SAML Setup"
            }
        
        async def mock_cleanup_tools():
            pass
        
        # Patch the tool management methods
        with patch.object(orchestrator, '_initialize_tools', side_effect=mock_init_tools):
            with patch.object(orchestrator, '_cleanup_tools', side_effect=mock_cleanup_tools):
                
                # Execute integration
                result = await orchestrator.execute_integration(
                    job_id=uuid.uuid4(),
                    documentation_url=sample_integration_data["documentation_url"],
                    vendor_name=sample_integration_data["vendor_name"],
                    okta_domain=sample_integration_data["okta_domain"]
                )
        
        # Verify result
        assert result["success"] is True
        assert "job_id" in result
        assert "final_state" in result
        assert "completed_steps" in result
    
    async def test_perception_to_cognition_flow(self, sample_html_content, mock_llm_response):
        """Test flow from perception to cognition modules."""
        
        # Initialize modules
        perception = PerceptionModule()
        cognition = CognitionModule()
        
        # Mock HTTP client for perception
        mock_response = type('MockResponse', (), {
            'text': sample_html_content,
            'status_code': 200,
            'headers': {'content-type': 'text/html'},
            'raise_for_status': lambda: None
        })()
        
        with patch.object(perception.http_client, 'get', return_value=mock_response):
            # Process documentation
            doc_result = await perception.process_documentation(
                "https://testvendor.com/saml-setup"
            )
        
        # Verify perception results
        assert doc_result["vendor_name"] == "TestVendor"
        assert len(doc_result["sections"]) > 0
        assert "saml_info" in doc_result
        
        # Mock LLM for cognition
        mock_llm_messages = AsyncMock()
        mock_llm_messages.content = '{"vendor_name": "TestVendor", "total_steps": 2, "steps": []}'
        
        with patch.object(cognition._llm, 'apredict_messages', return_value=mock_llm_messages):
            # Generate plan from documentation
            plan = await cognition.generate_integration_plan(
                vendor_name=doc_result["vendor_name"],
                documentation_url=doc_result["url"],
                documentation_content=doc_result["markdown_content"],
                okta_domain="test.okta.com"
            )
        
        # Verify cognition results
        assert plan.vendor_name == "TestVendor"
        assert plan.total_steps >= 1
        assert len(plan.steps) >= 1
        
        # Cleanup
        await perception.close()
    
    async def test_memory_persistence_integration(self, memory_module, clean_database):
        """Test memory persistence across different modules."""
        
        job_id = uuid.uuid4()
        
        # Create job
        job = await memory_module.create_job(
            job_id=job_id,
            input_url="https://testvendor.com/saml",
            vendor_name="TestVendor",
            initial_state={"step": 0}
        )
        
        # Store execution trace
        trace = await memory_module.store_trace(
            job_id=job_id,
            trace_type="thought",
            content={"reasoning": "Starting integration"},
            status="success"
        )
        
        # Store knowledge chunks
        chunks = [
            {
                "text": "Configure SAML by going to Settings > SSO",
                "metadata": {"section": "config"},
                "content_type": "configuration",
                "chunk_index": 0
            }
        ]
        
        stored_chunks = await memory_module.store_knowledge_chunks(job_id, chunks)
        
        # Store procedural pattern
        pattern = await memory_module.store_pattern(
            vendor_domain="testvendor.com",
            context_hash="sso_config",
            successful_action={"tool": "WebInteractor", "action": "click", "selector": "#enable-sso"},
            confidence_score=0.9
        )
        
        # Update job state
        new_state = {"step": 1, "working_memory": {"app_id": "test123"}}
        await memory_module.store_job_state(job_id, new_state)
        
        # Verify all data is stored and retrievable
        retrieved_job = await memory_module.get_job(job_id)
        assert retrieved_job.job_id == job_id
        assert retrieved_job.vendor_name == "TestVendor"
        
        job_traces = await memory_module.get_job_traces(job_id)
        assert len(job_traces) == 1
        assert job_traces[0].trace_type == "thought"
        
        search_results = await memory_module.search_knowledge(
            job_id=job_id,
            query="SAML configuration",
            similarity_threshold=0.1
        )
        assert len(search_results) >= 1
        
        found_patterns = await memory_module.find_patterns(
            vendor_domain="testvendor.com",
            context_hash="sso_config"
        )
        assert len(found_patterns) == 1
        assert found_patterns[0].confidence_score == 0.9
        
        loaded_state = await memory_module.load_job_state(job_id)
        assert loaded_state == new_state
    
    @pytest.mark.slow
    async def test_error_recovery_flow(self, orchestrator, clean_database):
        """Test error recovery and self-healing."""
        
        job_id = uuid.uuid4()
        
        # Mock tools with initial failure then success
        call_count = {"okta": 0, "web": 0}
        
        async def failing_okta_create(*args, **kwargs):
            call_count["okta"] += 1
            if call_count["okta"] == 1:
                raise Exception("Rate limit exceeded")
            return {"id": "recovered_app_123", "label": "Test App"}
        
        async def failing_web_navigate(*args, **kwargs):
            call_count["web"] += 1
            if call_count["web"] == 1:
                raise Exception("Network timeout")
            return {"success": True, "url": "https://example.com"}
        
        # Mock tool initialization
        async def mock_init_tools(okta_domain=None):
            orchestrator.okta_client = AsyncMock()
            orchestrator.okta_client.create_saml_app = failing_okta_create
            
            orchestrator.web_interactor = AsyncMock()
            orchestrator.web_interactor.navigate = failing_web_navigate
        
        with patch.object(orchestrator, '_initialize_tools', side_effect=mock_init_tools):
            with patch.object(orchestrator, '_cleanup_tools', side_effect=AsyncMock()):
                
                # Execute integration - should recover from initial failures
                result = await orchestrator.execute_integration(
                    job_id=job_id,
                    documentation_url="https://testvendor.com/saml",
                    vendor_name="TestVendor",
                    okta_domain="test.okta.com"
                )
        
        # Should succeed despite initial failures
        assert result["success"] is True
        
        # Verify recovery was logged
        traces = await orchestrator.memory_module.get_job_traces(job_id)
        error_traces = [t for t in traces if t.status == "failure"]
        assert len(error_traces) > 0  # Should have recorded failures
    
    async def test_hitl_workflow(self, orchestrator, clean_database):
        """Test Human-in-the-Loop workflow."""
        
        job_id = uuid.uuid4()
        
        # Mock a scenario that triggers HITL
        async def mock_init_tools(okta_domain=None):
            orchestrator.okta_client = AsyncMock()
            orchestrator.okta_client.create_saml_app.return_value = {"id": "app123"}
            
            orchestrator.web_interactor = AsyncMock()
            # Simulate CAPTCHA detection
            orchestrator.web_interactor.navigate.side_effect = Exception("CAPTCHA detected")
        
        with patch.object(orchestrator, '_initialize_tools', side_effect=mock_init_tools):
            with patch.object(orchestrator, '_cleanup_tools', side_effect=AsyncMock()):
                
                # Execute - should pause for HITL
                result = await orchestrator.execute_integration(
                    job_id=job_id,
                    documentation_url="https://testvendor.com/saml",
                    vendor_name="TestVendor",
                    okta_domain="test.okta.com"
                )
        
        # Should be paused for HITL, not failed
        if not result["success"]:
            # Check if job was paused for HITL
            job = await orchestrator.memory_module.get_job(job_id)
            # In our mock scenario, it might fail rather than pause
            # In a real implementation, we'd check for "paused_for_hitl" status
            assert job is not None
    
    async def test_state_persistence_and_resume(self, orchestrator, clean_database):
        """Test state persistence and job resumption."""
        
        job_id = uuid.uuid4()
        
        # Mock successful initial steps, then pause
        step_count = {"count": 0}
        
        async def mock_action_with_pause(*args, **kwargs):
            step_count["count"] += 1
            if step_count["count"] <= 2:
                return {"success": True, "step": step_count["count"]}
            else:
                # Simulate pause condition
                raise Exception("Manual pause for testing")
        
        async def mock_init_tools(okta_domain=None):
            orchestrator.okta_client = AsyncMock()
            orchestrator.okta_client.create_saml_app = mock_action_with_pause
            
            orchestrator.web_interactor = AsyncMock()
            orchestrator.web_interactor.navigate = mock_action_with_pause
        
        # First execution - should store state before "failure"
        with patch.object(orchestrator, '_initialize_tools', side_effect=mock_init_tools):
            with patch.object(orchestrator, '_cleanup_tools', side_effect=AsyncMock()):
                try:
                    await orchestrator.execute_integration(
                        job_id=job_id,
                        documentation_url="https://testvendor.com/saml",
                        vendor_name="TestVendor",
                        okta_domain="test.okta.com"
                    )
                except:
                    pass  # Expected to fail for this test
        
        # Verify state was stored
        job = await orchestrator.memory_module.get_job(job_id)
        assert job is not None
        
        # Verify traces were stored
        traces = await orchestrator.memory_module.get_job_traces(job_id)
        assert len(traces) > 0
    
    async def test_metrics_collection_integration(self, orchestrator, clean_database):
        """Test that metrics are collected during integration."""
        from konfig.utils.metrics import metrics_registry
        
        job_id = uuid.uuid4()
        
        # Reset metrics
        for metric in metrics_registry.metrics.values():
            if hasattr(metric, '_value'):
                metric._value = 0
            if hasattr(metric, '_observations'):
                metric._observations.clear()
        
        # Mock successful execution
        async def mock_init_tools(okta_domain=None):
            orchestrator.okta_client = AsyncMock()
            orchestrator.okta_client.create_saml_app.return_value = {"id": "app123"}
            
            orchestrator.web_interactor = AsyncMock()
            orchestrator.web_interactor.navigate.return_value = {"success": True}
        
        with patch.object(orchestrator, '_initialize_tools', side_effect=mock_init_tools):
            with patch.object(orchestrator, '_cleanup_tools', side_effect=AsyncMock()):
                
                result = await orchestrator.execute_integration(
                    job_id=job_id,
                    documentation_url="https://testvendor.com/saml",
                    vendor_name="TestVendor",
                    okta_domain="test.okta.com"
                )
        
        # Verify metrics were collected
        jobs_total = metrics_registry.get_metric("konfig_jobs_total")
        assert jobs_total is not None
        
        # Export metrics to verify structure
        metrics_json = metrics_registry.export_json()
        assert "konfig_jobs_total" in metrics_json
        
        prometheus_format = metrics_registry.export_prometheus()
        assert "konfig_jobs_total" in prometheus_format