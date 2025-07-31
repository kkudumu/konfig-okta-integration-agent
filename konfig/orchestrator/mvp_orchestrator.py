"""
Phase 1 MVP Orchestrator for Konfig.

This module provides a simplified orchestrator with hard-coded plans
for demonstrating the core functionality without requiring full LLM integration.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from konfig.config.settings import get_settings
from konfig.modules.memory.memory_module import MemoryModule
from konfig.modules.perception.perception_module import PerceptionModule
from konfig.tools.okta_api_client import OktaAPIClient
from konfig.tools.web_interactor import WebInteractor
from konfig.utils.logging import LoggingMixin
from konfig.utils.metrics import job_started, job_completed, tool_call_started, tool_call_completed
from konfig.utils.tracing import start_trace, finish_trace, span


class MVPOrchestrator(LoggingMixin):
    """
    Phase 1 MVP Orchestrator with hard-coded integration plans.
    
    This orchestrator demonstrates the core Konfig functionality using
    predefined plans for known applications, without requiring LLM integration.
    """
    
    def __init__(self):
        """Initialize the MVP Orchestrator."""
        super().__init__()
        self.setup_logging("mvp_orchestrator")
        
        self.settings = get_settings()
        self.memory_module = MemoryModule()
        self.perception_module = PerceptionModule()
        
        # Known vendor configurations
        self.known_vendors = {
            "google_workspace": {
                "name": "Google Workspace",
                "admin_url": "https://admin.google.com",
                "saml_path": "/ac/security/sso",
                "plan": self._get_google_workspace_plan()
            },
            "slack": {
                "name": "Slack",
                "admin_url": "https://{workspace}.slack.com/admin",
                "saml_path": "/settings/auth",
                "plan": self._get_slack_plan()
            },
            "atlassian": {
                "name": "Atlassian",
                "admin_url": "https://admin.atlassian.com",
                "saml_path": "/security/saml",
                "plan": self._get_atlassian_plan()
            }
        }
        
        self.logger.info("MVP Orchestrator initialized")
    
    async def integrate_application(
        self,
        documentation_url: str,
        okta_domain: str,
        app_name: Optional[str] = None,
        vendor_hint: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute SSO integration using hard-coded plans.
        
        Args:
            documentation_url: URL to vendor documentation
            okta_domain: Okta domain for the integration
            app_name: Optional custom application name
            vendor_hint: Optional hint about which vendor this is
            dry_run: If True, simulate without making actual changes
            
        Returns:
            Integration result
        """
        job_id = uuid.uuid4()
        trace_id = start_trace(str(job_id), "mvp_integration")
        
        self.logger.info(
            "Starting MVP integration",
            job_id=str(job_id),
            documentation_url=documentation_url,
            vendor_hint=vendor_hint,
            dry_run=dry_run
        )
        
        # Record metrics
        job_started(str(job_id), vendor_hint or "unknown")
        
        start_time = datetime.now()
        
        try:
            # Create job record
            await self.memory_module.create_job(
                job_id=job_id,
                input_url=documentation_url,
                vendor_name=vendor_hint or "Unknown",
                okta_domain=okta_domain,
                initial_state={
                    "dry_run": dry_run,
                    "start_time": start_time.isoformat(),
                    "phase": "mvp"
                }
            )
            
            # Step 1: Process documentation to understand the vendor
            with span("process_documentation"):
                vendor_info = await self._process_documentation(documentation_url, job_id)
                await self.memory_module.store_trace(
                    job_id=job_id,
                    trace_type="thought",
                    content={"step": "documentation_analysis", "vendor_info": vendor_info},
                    status="success"
                )
            
            # Step 2: Determine integration plan
            with span("determine_plan"):
                plan = await self._determine_integration_plan(vendor_info, vendor_hint)
                await self.memory_module.store_trace(
                    job_id=job_id,
                    trace_type="thought",
                    content={"step": "plan_selection", "plan_name": plan["name"]},
                    status="success"
                )
            
            # Step 3: Execute the plan
            with span("execute_plan"):
                execution_result = await self._execute_plan(
                    job_id=job_id,
                    plan=plan,
                    okta_domain=okta_domain,
                    app_name=app_name or vendor_info.get("vendor_name", "Unknown App"),
                    dry_run=dry_run
                )
            
            # Update job status
            await self.memory_module.update_job_status(job_id, "completed_success")
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record success metrics
            job_completed(str(job_id), True, duration, vendor_hint or "unknown")
            
            result = {
                "success": True,
                "job_id": str(job_id),
                "vendor": vendor_info.get("vendor_name"),
                "plan_used": plan["name"],
                "duration_seconds": duration,
                "okta_app_id": execution_result.get("okta_app_id"),
                "integration_status": "completed",
                "steps_completed": execution_result.get("steps_completed", 0),
                "dry_run": dry_run
            }
            
            self.logger.info("MVP integration completed successfully", **result)
            return result
            
        except Exception as e:
            self.log_error("integrate_application", e, job_id=str(job_id))
            
            # Update job with failure
            await self.memory_module.update_job_status(job_id, "completed_failure", str(e))
            
            # Record failure metrics
            duration = (datetime.now() - start_time).total_seconds()
            job_completed(str(job_id), False, duration, vendor_hint or "unknown")
            
            return {
                "success": False,
                "job_id": str(job_id),
                "error": str(e),
                "duration_seconds": duration
            }
        finally:
            await finish_trace(trace_id)
    
    async def _process_documentation(self, url: str, job_id: uuid.UUID) -> Dict[str, Any]:
        """Process vendor documentation to extract key information."""
        try:
            doc_result = await self.perception_module.process_documentation(url, str(job_id))
            
            # Extract key information
            vendor_info = {
                "vendor_name": doc_result.get("vendor_name", "Unknown"),
                "title": doc_result.get("title", ""),
                "saml_info": doc_result.get("saml_info", {}),
                "relevant_sections": doc_result.get("relevant_sections", []),
                "has_configuration_steps": len(doc_result.get("saml_info", {}).get("configuration_steps", [])) > 0
            }
            
            self.logger.info("Documentation processed", vendor=vendor_info["vendor_name"])
            return vendor_info
            
        except Exception as e:
            self.logger.warning(f"Failed to process documentation: {e}")
            # Fallback to basic info
            return {
                "vendor_name": "Unknown Vendor",
                "title": url,
                "saml_info": {},
                "relevant_sections": [],
                "has_configuration_steps": False
            }
    
    async def _determine_integration_plan(
        self,
        vendor_info: Dict[str, Any],
        vendor_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Determine which integration plan to use."""
        
        vendor_name = vendor_info.get("vendor_name", "").lower()
        title = vendor_info.get("title", "").lower()
        
        # Try to match against known vendors
        if vendor_hint and vendor_hint.lower() in self.known_vendors:
            plan = self.known_vendors[vendor_hint.lower()]
            self.logger.info(f"Using plan for {plan['name']} (from hint)")
            return plan
        
        # Match by vendor name patterns
        for vendor_key, vendor_config in self.known_vendors.items():
            vendor_config_name = vendor_config["name"].lower()
            if (vendor_config_name in vendor_name or 
                vendor_config_name in title or
                vendor_key.replace("_", " ") in vendor_name):
                self.logger.info(f"Detected vendor: {vendor_config['name']}")
                return vendor_config
        
        # Default generic plan
        self.logger.info("Using generic integration plan")
        return {
            "name": "Generic SAML",
            "admin_url": None,
            "saml_path": None,
            "plan": self._get_generic_plan()
        }
    
    async def _execute_plan(
        self,
        job_id: uuid.UUID,
        plan: Dict[str, Any],
        okta_domain: str,
        app_name: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute the integration plan."""
        
        steps = plan["plan"]
        completed_steps = 0
        okta_app_id = None
        
        # Initialize tools
        async with OktaAPIClient(okta_domain=okta_domain) as okta_client:
            if not dry_run:
                async with WebInteractor() as web_interactor:
                    return await self._execute_steps_with_tools(
                        job_id, steps, okta_client, web_interactor, app_name, dry_run
                    )
            else:
                return await self._execute_steps_with_tools(
                    job_id, steps, okta_client, None, app_name, dry_run
                )
    
    async def _execute_steps_with_tools(
        self,
        job_id: uuid.UUID,
        steps: List[Dict[str, Any]],
        okta_client: OktaAPIClient,
        web_interactor: Optional[WebInteractor],
        app_name: str,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Execute steps with initialized tools."""
        
        completed_steps = 0
        okta_app_id = None
        working_memory = {}
        
        for i, step in enumerate(steps):
            step_name = step.get("name", f"Step {i+1}")
            tool = step.get("tool")
            action = step.get("action")
            params = step.get("params", {})
            
            self.logger.info(f"Executing step: {step_name}")
            
            # Record step start
            tool_call_started(tool, action)
            step_start_time = datetime.now()
            
            try:
                # Store step trace
                await self.memory_module.store_trace(
                    job_id=job_id,
                    trace_type="tool_call",
                    content={
                        "step": i + 1,
                        "name": step_name,
                        "tool": tool,
                        "action": action,
                        "params": params,
                        "dry_run": dry_run
                    },
                    status="success"
                )
                
                # Execute step
                if dry_run:
                    result = await self._simulate_step(step, working_memory, app_name)
                else:
                    result = await self._execute_step(
                        step, okta_client, web_interactor, working_memory, app_name
                    )
                
                # Store result
                working_memory.update(result.get("extracted_data", {}))
                if "okta_app_id" in result:
                    okta_app_id = result["okta_app_id"]
                
                # Record step completion
                step_duration = (datetime.now() - step_start_time).total_seconds()
                tool_call_completed(tool, action, True, step_duration)
                
                # Store observation
                await self.memory_module.store_trace(
                    job_id=job_id,
                    trace_type="observation",
                    content={
                        "step": i + 1,
                        "result": result,
                        "working_memory": working_memory
                    },
                    status="success",
                    duration_ms=int(step_duration * 1000)
                )
                
                completed_steps += 1
                
                # Brief pause between steps
                await asyncio.sleep(1)
                
            except Exception as e:
                step_duration = (datetime.now() - step_start_time).total_seconds()
                tool_call_completed(tool, action, False, step_duration)
                
                self.logger.error(f"Step '{step_name}' failed: {e}")
                
                # Store error trace
                await self.memory_module.store_trace(
                    job_id=job_id,
                    trace_type="observation",
                    content={
                        "step": i + 1,
                        "error": str(e),
                        "step_name": step_name
                    },
                    status="failure",
                    error_details=str(e)
                )
                
                # For MVP, continue with remaining steps rather than failing completely
                self.logger.warning(f"Continuing with remaining steps after failure in: {step_name}")
        
        return {
            "steps_completed": completed_steps,
            "total_steps": len(steps),
            "okta_app_id": okta_app_id,
            "working_memory": working_memory
        }
    
    async def _execute_step(
        self,
        step: Dict[str, Any],
        okta_client: OktaAPIClient,
        web_interactor: Optional[WebInteractor],
        working_memory: Dict[str, Any],
        app_name: str
    ) -> Dict[str, Any]:
        """Execute a single step."""
        
        tool = step.get("tool")
        action = step.get("action")
        params = step.get("params", {})
        
        # Substitute variables from working memory
        resolved_params = self._resolve_parameters(params, working_memory, app_name)
        
        if tool == "OktaAPIClient":
            return await self._execute_okta_action(okta_client, action, resolved_params)
        elif tool == "WebInteractor" and web_interactor:
            return await self._execute_web_action(web_interactor, action, resolved_params)
        else:
            return {"status": "skipped", "reason": f"Tool {tool} not available"}
    
    async def _simulate_step(
        self,
        step: Dict[str, Any],
        working_memory: Dict[str, Any],
        app_name: str
    ) -> Dict[str, Any]:
        """Simulate step execution for dry run."""
        
        tool = step.get("tool")
        action = step.get("action")
        
        # Return realistic mock data
        if tool == "OktaAPIClient" and action == "create_saml_app":
            return {
                "status": "simulated",
                "okta_app_id": "mock_app_12345",
                "extracted_data": {
                    "okta_app_id": "mock_app_12345",
                    "sso_url": f"https://mock.okta.com/app/mock_app_12345/sso/saml",
                    "entity_id": "http://www.okta.com/mock_app_12345"
                }
            }
        elif tool == "OktaAPIClient" and action == "get_app_metadata":
            return {
                "status": "simulated",
                "extracted_data": {
                    "certificate": "mock_certificate_data",
                    "metadata_xml": "mock_saml_metadata"
                }
            }
        elif tool == "WebInteractor":
            return {
                "status": "simulated",
                "extracted_data": {
                    "vendor_acs_url": "https://mock-vendor.com/saml/acs",
                    "vendor_entity_id": "https://mock-vendor.com/saml/metadata"
                }
            }
        else:
            return {"status": "simulated"}
    
    async def _execute_okta_action(
        self,
        okta_client: OktaAPIClient,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an Okta API action."""
        
        if action == "create_saml_app":
            result = await okta_client.create_saml_app(
                label=params.get("label", "SAML Application"),
                settings=params.get("settings")
            )
            
            return {
                "status": "success",
                "okta_app_id": result.get("id"),
                "extracted_data": {
                    "okta_app_id": result.get("id"),
                    "app_label": result.get("label")
                }
            }
        
        elif action == "get_app_metadata":
            app_id = params.get("app_id")
            metadata = await okta_client.get_app_metadata(app_id)
            
            return {
                "status": "success",
                "extracted_data": {
                    "sso_url": metadata.get("sso_url"),
                    "entity_id": metadata.get("entity_id"),
                    "certificate": metadata.get("certificate")
                }
            }
        
        elif action == "update_saml_settings":
            app_id = params.get("app_id")
            settings = params.get("settings", {})
            result = await okta_client.update_saml_settings(app_id, settings)
            
            return {
                "status": "success",
                "extracted_data": {
                    "updated_app_id": result.get("id")
                }
            }
        
        else:
            raise ValueError(f"Unknown Okta action: {action}")
    
    async def _execute_web_action(
        self,
        web_interactor: WebInteractor,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a web browser action."""
        
        if action == "navigate":
            result = await web_interactor.navigate(params.get("url"))
            return {
                "status": "success" if result.get("success") else "failed",
                "extracted_data": {
                    "current_url": result.get("final_url"),
                    "page_title": result.get("title")
                }
            }
        
        elif action == "click":
            result = await web_interactor.click(
                params.get("selector"),
                text_content=params.get("text_content")
            )
            return {
                "status": "success" if result.get("success") else "failed",
                "extracted_data": {}
            }
        
        elif action == "type":
            result = await web_interactor.type(
                params.get("selector"),
                params.get("text"),
                secret=params.get("secret", False)
            )
            return {
                "status": "success" if result.get("success") else "failed",
                "extracted_data": {}
            }
        
        elif action == "get_element_text":
            text = await web_interactor.get_element_text(params.get("selector"))
            return {
                "status": "success",
                "extracted_data": {
                    params.get("extract_as", "text"): text
                }
            }
        
        else:
            raise ValueError(f"Unknown web action: {action}")
    
    def _resolve_parameters(
        self,
        params: Dict[str, Any],
        working_memory: Dict[str, Any],
        app_name: str
    ) -> Dict[str, Any]:
        """Resolve parameter variables with actual values."""
        
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Replace variables
                resolved_value = value
                resolved_value = resolved_value.replace("{app_name}", app_name)
                
                # Replace working memory variables
                for mem_key, mem_value in working_memory.items():
                    resolved_value = resolved_value.replace(f"{{{mem_key}}}", str(mem_value))
                
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        return resolved
    
    # Hard-coded integration plans for known vendors
    
    def _get_google_workspace_plan(self) -> List[Dict[str, Any]]:
        """Get integration plan for Google Workspace."""
        return [
            {
                "name": "Create Okta SAML App for Google Workspace",
                "tool": "OktaAPIClient",
                "action": "create_saml_app",
                "params": {
                    "label": "{app_name}",
                    "settings": {
                        "signOn": {
                            "ssoAcsUrl": "https://www.google.com/a/{domain}/acs",
                            "audience": "google.com/a/{domain}",
                            "recipient": "https://www.google.com/a/{domain}/acs",
                            "destination": "https://www.google.com/a/{domain}/acs"
                        }
                    }
                }
            },
            {
                "name": "Get Okta SAML metadata",
                "tool": "OktaAPIClient",
                "action": "get_app_metadata",
                "params": {
                    "app_id": "{okta_app_id}"
                }
            }
        ]
    
    def _get_slack_plan(self) -> List[Dict[str, Any]]:
        """Get integration plan for Slack."""
        return [
            {
                "name": "Create Okta SAML App for Slack",
                "tool": "OktaAPIClient",
                "action": "create_saml_app",
                "params": {
                    "label": "{app_name}",
                    "settings": {
                        "signOn": {
                            "subjectNameIdTemplate": "${user.email}",
                            "subjectNameIdFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                        }
                    }
                }
            },
            {
                "name": "Get Okta SAML metadata",
                "tool": "OktaAPIClient", 
                "action": "get_app_metadata",
                "params": {
                    "app_id": "{okta_app_id}"
                }
            }
        ]
    
    def _get_atlassian_plan(self) -> List[Dict[str, Any]]:
        """Get integration plan for Atlassian."""
        return [
            {
                "name": "Create Okta SAML App for Atlassian",
                "tool": "OktaAPIClient",
                "action": "create_saml_app",
                "params": {
                    "label": "{app_name}",
                    "settings": {
                        "signOn": {
                            "subjectNameIdTemplate": "${user.email}",
                            "subjectNameIdFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                        }
                    }
                }
            },
            {
                "name": "Get Okta SAML metadata",
                "tool": "OktaAPIClient",
                "action": "get_app_metadata", 
                "params": {
                    "app_id": "{okta_app_id}"
                }
            }
        ]
    
    def _get_generic_plan(self) -> List[Dict[str, Any]]:
        """Get generic integration plan for unknown vendors."""
        return [
            {
                "name": "Create generic SAML application in Okta",
                "tool": "OktaAPIClient",
                "action": "create_saml_app",
                "params": {
                    "label": "{app_name}"
                }
            },
            {
                "name": "Get Okta SAML metadata",
                "tool": "OktaAPIClient",
                "action": "get_app_metadata",
                "params": {
                    "app_id": "{okta_app_id}"
                }
            }
        ]
    
    async def get_job_status(self, job_id: uuid.UUID) -> Dict[str, Any]:
        """Get the status of an integration job."""
        job = await self.memory_module.get_job(job_id)
        
        if not job:
            return {"error": "Job not found"}
        
        # Get recent traces
        traces = await self.memory_module.get_job_traces(job_id, limit=10)
        
        return {
            "job_id": str(job_id),
            "status": job.status,
            "vendor_name": job.vendor_name,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "okta_app_id": job.okta_app_id,
            "error_message": job.error_message,
            "recent_traces": [
                {
                    "timestamp": trace.timestamp.isoformat(),
                    "type": trace.trace_type,
                    "status": trace.status,
                    "content": trace.content
                }
                for trace in traces
            ]
        }
    
    async def list_known_vendors(self) -> List[Dict[str, Any]]:
        """List all known vendor configurations."""
        return [
            {
                "key": key,
                "name": config["name"],
                "admin_url": config["admin_url"],
                "steps": len(config["plan"])
            }
            for key, config in self.known_vendors.items()
        ]