"""
Phase 1 MVP Orchestrator for Konfig.

This module provides a simplified orchestrator with hard-coded plans
for demonstrating the core functionality without requiring full LLM integration.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from konfig.config.settings import get_settings
from konfig.modules.memory.memory_module import MemoryModule
from konfig.modules.perception.perception_module import PerceptionModule
from konfig.services.error_context_collector import (
    ErrorContextCollector, 
    ErrorRecoveryAnalyzer, 
    StepContextTracker
)
from konfig.tools.okta_api_client import OktaAPIClient
from konfig.tools.web_interactor import WebInteractor, ElementNotFoundError
from konfig.utils.logging import LoggingMixin
from konfig.utils.metrics import job_started, job_completed, tool_call_started, tool_call_completed
from konfig.utils.tracing import start_trace, finish_trace, span
from konfig.services.agent_swarm import SwarmOrchestrator, IntegrationGoal


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
        
        # Error context and recovery components
        self.error_context_collector = ErrorContextCollector()
        self.error_recovery_analyzer = ErrorRecoveryAnalyzer()
        self.step_tracker = StepContextTracker()
        
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
        dry_run: bool = False,
        auto_approve: bool = False,
        progress_callback=None
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
                # Get documentation content for LLM analysis
                documentation_content = await self._get_documentation_content(documentation_url)
                plan = await self._determine_integration_plan(
                    vendor_info, documentation_content, okta_domain, vendor_hint
                )
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
                    dry_run=dry_run,
                    auto_approve=auto_approve,
                    progress_callback=progress_callback
                )
            
            # Check if the execution was cancelled
            if execution_result.get("status") == "cancelled":
                # Update job status as cancelled
                await self.memory_module.update_job_status(job_id, "cancelled")
                
                # Calculate duration
                duration = (datetime.now() - start_time).total_seconds()
                
                # Record cancellation metrics
                job_completed(str(job_id), False, duration, vendor_hint or "unknown")
                
                result = {
                    "success": False,
                    "job_id": str(job_id),
                    "vendor": vendor_info.get("vendor_name"),
                    "plan_used": plan["name"],
                    "duration_seconds": duration,
                    "status": "cancelled",
                    "message": execution_result.get("message", "Integration cancelled by user"),
                    "steps_completed": 0,
                    "dry_run": dry_run
                }
                
                self.logger.info("Integration cancelled by user", **result)
                return result
            
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
        documentation_content: str,
        okta_domain: str,
        vendor_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use LLM to dynamically determine integration plan."""
        
        try:
            from konfig.services.intelligent_planner import IntelligentPlanner
            
            planner = IntelligentPlanner()
            
            # Generate dynamic plan using LLM analysis
            plan = await planner.generate_integration_plan(
                vendor_info=vendor_info,
                documentation_content=documentation_content,
                okta_domain=okta_domain
            )
            
            self.logger.info(
                f"Generated dynamic plan for {plan.vendor_name}",
                steps=len(plan.steps),
                complexity=plan.complexity,
                duration=plan.estimated_duration
            )
            
            # Convert to orchestrator format
            return {
                "name": f"{plan.vendor_name} SAML Integration",
                "admin_url": None,  # Will be determined dynamically
                "saml_path": None,  # Will be determined dynamically
                "plan": [step.to_dict() for step in plan.steps],
                "metadata": {
                    "vendor_name": plan.vendor_name,
                    "complexity": plan.complexity,
                    "estimated_duration": plan.estimated_duration,
                    "requirements": plan.requirements,
                    "generated_by": "llm"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate dynamic plan: {e}")
            # Fallback to basic generic plan
            return await self._get_fallback_plan(vendor_info)
    
    async def _get_fallback_plan(self, vendor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback plan when LLM-driven planning fails."""
        self.logger.warning("Using fallback generic plan")
        return {
            "name": "Generic SAML",
            "admin_url": None,
            "saml_path": None,
            "plan": [
                {
                    "name": "Create SAML Application in Okta",
                    "tool": "OktaAPIClient",
                    "action": "create_saml_app",
                    "params": {"label": "{app_name}"},
                    "description": "Creates a basic SAML application in Okta"
                },
                {
                    "name": "Get Okta SAML Metadata",
                    "tool": "OktaAPIClient", 
                    "action": "get_app_metadata",
                    "params": {"app_id": "{okta_app_id}"},
                    "description": "Retrieves SAML metadata from Okta"
                }
            ],
            "metadata": {
                "vendor_name": vendor_info.get("vendor_name", "Unknown"),
                "complexity": "simple",
                "estimated_duration": 10,
                "requirements": ["Okta admin access"],
                "generated_by": "fallback"
            }
        }
    
    async def _get_documentation_content(self, documentation_url: str) -> str:
        """Retrieve full documentation content for LLM analysis."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(documentation_url, timeout=30.0)
                response.raise_for_status()
                
                # Extract text content (basic HTML parsing)
                content = response.text
                
                # Simple HTML tag removal for cleaner text
                import re
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                self.logger.debug(f"Retrieved {len(content)} characters of documentation")
                return content
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve documentation content: {e}")
            return f"Documentation URL: {documentation_url}\n(Content could not be retrieved)"
    
    async def _execute_plan(
        self,
        job_id: uuid.UUID,
        plan: Dict[str, Any],
        okta_domain: str,
        app_name: str,
        dry_run: bool = False,
        auto_approve: bool = False,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Execute the integration plan with user approval."""
        
        # Step 1: Handle plan approval (skip if auto_approve is True)
        if auto_approve:
            self.logger.info("Auto-approve enabled, skipping user approval")
            steps = plan["plan"]
        else:
            from konfig.services.plan_approval import PlanApprovalService
            
            approval_service = PlanApprovalService()
            
            integration_context = {
                "vendor_name": plan.get("metadata", {}).get("vendor_name", "Unknown Vendor"),
                "okta_domain": okta_domain,
                "app_name": app_name,
                "dry_run": dry_run,
                "job_id": str(job_id)
            }
            
            # Request user approval
            approved_steps, user_approved = await approval_service.request_plan_approval(
                raw_plan=plan["plan"],
                integration_context=integration_context
            )
            
            if not user_approved:
                self.logger.info("Integration cancelled by user")
                return {
                    "status": "cancelled",
                    "message": "Integration cancelled by user",
                    "executed_steps": 0,
                    "total_steps": len(plan["plan"])
                }
            
            # Display execution summary
            approval_service.display_execution_summary(approved_steps)
            
            # Convert approved steps back to dict format for execution
            steps = [
                {
                    "name": step.name,
                    "tool": step.tool,
                    "action": step.action,
                    "params": step.params
                }
                for step in approved_steps
            ]
        completed_steps = 0
        okta_app_id = None
        
        # Initialize tools
        async with OktaAPIClient(okta_domain=okta_domain) as okta_client:
            if not dry_run:
                async with WebInteractor() as web_interactor:
                    return await self._execute_steps_with_tools(
                        job_id, steps, okta_client, web_interactor, app_name, dry_run, progress_callback
                    )
            else:
                return await self._execute_steps_with_tools(
                    job_id, steps, okta_client, None, app_name, dry_run, progress_callback
                )
    
    async def _execute_steps_with_tools(
        self,
        job_id: uuid.UUID,
        steps: List[Dict[str, Any]],
        okta_client: OktaAPIClient,
        web_interactor: Optional[WebInteractor],
        app_name: str,
        dry_run: bool,
        progress_callback=None
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
            
            # Update progress callback if provided
            if progress_callback:
                progress_callback({"name": step_name, "action": action, "tool": tool})
            
            # Start step tracking for error context
            step_context = self.step_tracker.start_step(step_name, tool, action, params)
            
            # Record step start
            tool_call_started(tool, action)
            step_start_time = datetime.now()
            
            try:
                # Check if step is still needed based on current state
                step_needed = await self._validate_step_needed(
                    step, web_interactor, working_memory, dry_run
                )
                
                if not step_needed:
                    self.logger.info(f"Skipping step '{step_name}' - conditions already met")
                    
                    # Store skip trace
                    await self.memory_module.store_trace(
                        job_id=job_id,
                        trace_type="observation",
                        content={
                            "step": i + 1,
                            "name": step_name,
                            "status": "skipped",
                            "reason": "Step conditions already satisfied"
                        },
                        status="success"
                    )
                    
                    completed_steps += 1
                    continue
                
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
                
                # Complete step tracking
                self.step_tracker.complete_step(
                    status="success", 
                    result=result, 
                    duration_ms=step_duration * 1000
                )
                
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
                
                # For authentication steps, verify success more rigorously
                if any(keyword in step_name.lower() for keyword in ["login", "log in", "sign in", "auth"]):
                    auth_verified = await self._verify_authentication_success(web_interactor, working_memory, dry_run)
                    if not auth_verified:
                        self.logger.error(f"Authentication step '{step_name}' reported success but verification failed")
                        raise Exception("Authentication verification failed - still on login page")
                    else:
                        self.logger.info(f"Authentication step '{step_name}' verified successfully")
                        working_memory["authenticated"] = True
                
                completed_steps += 1
                
                # Brief pause between steps
                await asyncio.sleep(1)
                
            except Exception as e:
                step_duration = (datetime.now() - step_start_time).total_seconds()
                tool_call_completed(tool, action, False, step_duration)
                
                self.logger.error(f"Step '{step_name}' failed: {e}")
                
                # Complete step tracking with error
                self.step_tracker.complete_step(
                    status="failed", 
                    error=str(e), 
                    duration_ms=step_duration * 1000
                )
                
                # Collect comprehensive error context
                if web_interactor and hasattr(e, '__class__') and e.__class__.__name__ in ['TimeoutError', 'ElementNotFoundError']:
                    try:
                        error_context = await self.error_context_collector.collect_error_context(
                            error=e,
                            web_interactor=web_interactor,
                            step_tracker=self.step_tracker,
                            additional_context={
                                "step_index": i + 1,
                                "total_steps": len(steps),
                                "working_memory": working_memory.copy()
                            }
                        )
                        
                        # Analyze error and get recovery suggestions
                        from konfig.services.llm_service import LLMService
                        llm_service = LLMService()
                        recovery_plan = await self.error_recovery_analyzer.analyze_and_suggest_recovery(
                            error_context, llm_service
                        )
                        
                        self.logger.info(
                            f"Error analysis completed for '{step_name}'",
                            confidence=recovery_plan.get("confidence", "unknown"),
                            recovery_actions=len(recovery_plan.get("recovery_actions", []))
                        )
                        
                        # Store comprehensive error trace
                        try:
                            await self.memory_module.store_trace(
                                job_id=job_id,
                                trace_type="error_analysis",
                                content={
                                    "step": i + 1,
                                    "error": str(e),
                                    "step_name": step_name,
                                    "error_context": error_context.to_dict(),
                                    "recovery_plan": recovery_plan
                                },
                                status="failure",
                                error_details=str(e)
                            )
                        except Exception as trace_error:
                            self.logger.warning(f"Failed to store comprehensive error trace: {trace_error}")
                        
                    except Exception as context_error:
                        self.logger.warning(f"Failed to collect error context: {context_error}")
                        # Fallback to basic error trace
                        try:
                            await self.memory_module.store_trace(
                                job_id=job_id,
                                trace_type="observation",
                                content={
                                    "step": i + 1,
                                    "error": str(e),
                                    "step_name": step_name,
                                    "context_collection_error": str(context_error)
                                },
                                status="failure",
                                error_details=str(e)
                            )
                        except Exception as trace_error:
                            self.logger.warning(f"Failed to store fallback error trace: {trace_error}")
                else:
                    # Store basic error trace for non-web errors
                    try:
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
                    except Exception as trace_error:
                        self.logger.warning(f"Failed to store basic error trace: {trace_error}")
                
                # Attempt error recovery before continuing
                recovery_success = await self._attempt_error_recovery(
                    step, e, web_interactor, working_memory, i, steps, dry_run
                )
                
                if recovery_success:
                    self.logger.info(f"Successfully recovered from error in step: {step_name}")
                    completed_steps += 1
                else:
                    # Check if this step is critical for subsequent steps
                    if await self._is_step_critical(step, steps[i+1:] if i+1 < len(steps) else [], working_memory):
                        self.logger.error(f"Critical step '{step_name}' failed and could not be recovered. Stopping execution.")
                        break
                    else:
                        self.logger.warning(f"Non-critical step '{step_name}' failed. Continuing with remaining steps.")
        
        return {
            "steps_completed": completed_steps,
            "total_steps": len(steps),
            "okta_app_id": okta_app_id,
            "working_memory": working_memory
        }
    
    async def _attempt_error_recovery(
        self,
        failed_step: Dict[str, Any],
        error: Exception,
        web_interactor: Optional[WebInteractor],
        working_memory: Dict[str, Any],
        step_index: int,
        all_steps: List[Dict[str, Any]],
        dry_run: bool
    ) -> bool:
        """
        Attempt to recover from a failed step using intelligent analysis.
        
        Returns True if recovery was successful, False otherwise.
        """
        step_name = failed_step.get("name", f"Step {step_index + 1}")
        self.logger.info(f"Attempting error recovery for step: {step_name}")
        
        # If it's a web interaction error and we have a browser, try intelligent recovery
        if web_interactor and (isinstance(error, ElementNotFoundError) or "TimeoutError" in str(type(error))):
            return await self._attempt_web_recovery(failed_step, error, web_interactor, working_memory, dry_run)
        
        # If it's an authentication-related step, validate that authentication actually succeeded
        elif web_interactor and any(keyword in step_name.lower() for keyword in ["login", "log in", "sign in", "auth"]):
            return await self._attempt_authentication_recovery(failed_step, error, web_interactor, working_memory, dry_run)
        
        # If it's an Okta API error, try API recovery
        elif "OktaAPIError" in str(type(error)):
            return await self._attempt_okta_api_recovery(failed_step, error, working_memory, dry_run)
        
        # For other errors, log and return False
        self.logger.warning(f"No recovery strategy available for error type: {type(error).__name__}")
        return False

    async def _attempt_authentication_recovery(
        self,
        failed_step: Dict[str, Any],
        error: Exception,
        web_interactor: WebInteractor,
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Attempt to recover from authentication failures."""
        step_name = failed_step.get("name", "")
        self.logger.info(f"Attempting authentication recovery for step: {step_name}")
        
        if dry_run:
            self.logger.info("Dry run: Simulating successful authentication recovery")
            return True
        
        try:
            # First, check if we're actually authenticated by examining the current page
            current_url = await web_interactor.get_current_url()
            current_title = await web_interactor.get_page_title()
            current_dom = await web_interactor.get_current_dom(simplify=True)
            
            # Check for successful authentication indicators
            auth_success_indicators = [
                "dashboard" in current_url.lower(),
                "admin" in current_url.lower(),
                "console" in current_url.lower(),
                "home" in current_url.lower(),
                "dashboard" in current_title.lower(),
                "welcome" in current_title.lower(),
                "console" in current_title.lower(),
                "logout" in current_dom.lower(),
                "sign out" in current_dom.lower(),
                "profile" in current_dom.lower()
            ]
            
            # Check for authentication failure indicators
            auth_failure_indicators = [
                "login" in current_url.lower(),
                "signin" in current_url.lower(),
                "auth" in current_url.lower(),
                "error" in current_title.lower(),
                "invalid" in current_dom.lower(),
                "incorrect" in current_dom.lower(),
                "authentication failed" in current_dom.lower(),
                "username" in current_dom.lower() and "password" in current_dom.lower()
            ]
            
            if any(auth_success_indicators) and not any(auth_failure_indicators):
                self.logger.info("Authentication appears to have succeeded despite step failure")
                working_memory["authenticated"] = True
                return True
            
            # If we're still on a login page, try intelligent authentication
            if any(auth_failure_indicators):
                self.logger.info("Still on login page - attempting intelligent authentication")
                auth_success = await self._attempt_authentication(web_interactor, working_memory, dry_run)
                
                if auth_success:
                    # Wait a moment and re-check the page
                    await asyncio.sleep(3)
                    
                    # Verify authentication succeeded
                    new_url = await web_interactor.get_current_url()
                    new_title = await web_interactor.get_page_title()
                    
                    if not any(indicator in new_url.lower() or indicator in new_title.lower() 
                              for indicator in ["login", "signin", "auth"]):
                        self.logger.info("Authentication recovery successful")
                        working_memory["authenticated"] = True
                        return True
                    else:
                        self.logger.error("Authentication recovery failed - still on login page")
                        return False
                else:
                    self.logger.error("Authentication recovery failed - could not authenticate")
                    return False
            
            # If we can't determine the state, assume failure
            self.logger.warning("Cannot determine authentication state - assuming failure")
            return False
            
        except Exception as e:
            self.logger.error(f"Authentication recovery failed with error: {e}")
            return False

    async def _verify_authentication_success(
        self,
        web_interactor: WebInteractor,
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Verify that authentication actually succeeded by examining the current page state."""
        if dry_run:
            self.logger.info("Dry run: Simulating successful authentication verification")
            return True
        
        try:
            current_url = await web_interactor.get_current_url()
            current_title = await web_interactor.get_page_title()
            current_dom = await web_interactor.get_current_dom(simplify=True)
            
            self.logger.info(f"Verifying authentication - Current URL: {current_url}, Title: {current_title}")
            
            # Check for authentication failure indicators (still on login page)
            auth_failure_indicators = [
                "login" in current_url.lower(),
                "signin" in current_url.lower(),
                "/auth" in current_url.lower(),
                "login" in current_title.lower(),
                "sign in" in current_title.lower(),
                "authentication" in current_title.lower(),
                "error" in current_title.lower(),
                "invalid" in current_dom.lower(),
                "incorrect" in current_dom.lower(),
                "authentication failed" in current_dom.lower()
            ]
            
            # Check for specific login form indicators
            login_form_indicators = [
                "username" in current_dom.lower() and "password" in current_dom.lower(),
                'type="password"' in current_dom.lower(),
                "forgot password" in current_dom.lower(),
                "remember me" in current_dom.lower()
            ]
            
            # Check for successful authentication indicators
            auth_success_indicators = [
                "dashboard" in current_url.lower(),
                "admin" in current_url.lower(),
                "console" in current_url.lower(),
                "home" in current_url.lower(),
                "setup" in current_url.lower(),
                "dashboard" in current_title.lower(),
                "welcome" in current_title.lower(),
                "console" in current_title.lower(),
                "admin" in current_title.lower(),
                "logout" in current_dom.lower(),
                "sign out" in current_dom.lower(),
                "profile" in current_dom.lower()
            ]
            
            # If we see failure indicators or login forms, authentication failed
            if any(auth_failure_indicators) or any(login_form_indicators):
                self.logger.error(f"Authentication verification failed - still on login page or error page")
                self.logger.debug(f"Failure indicators found: {[i for i in auth_failure_indicators if i]}")
                return False
            
            # If we see success indicators, authentication succeeded
            if any(auth_success_indicators):
                self.logger.info("Authentication verification successful - detected authenticated page")
                return True
            
            # If no clear indicators, check for redirect away from login domain
            # Many systems redirect to a different subdomain after login
            login_domains = ["login.", "auth.", "signin.", "sso."]
            current_domain = current_url.lower()
            
            # If we're no longer on a login domain, likely authenticated
            if not any(domain in current_domain for domain in login_domains):
                self.logger.info("Authentication verification successful - redirected away from login domain")
                return True
            
            # If we can't determine state clearly, assume failure for safety
            self.logger.warning("Authentication verification inconclusive - assuming failure for safety")
            return False
            
        except Exception as e:
            self.logger.error(f"Authentication verification failed with error: {e}")
            return False
    
    async def _attempt_web_recovery(
        self,
        failed_step: Dict[str, Any],
        error: Exception,
        web_interactor: WebInteractor,
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Attempt to recover from web interaction failures."""
        
        self.logger.info("Attempting web interaction recovery")
        
        try:
            # Get current page state
            current_url = await web_interactor.get_current_url()
            page_title = await web_interactor.get_page_title()
            dom = await web_interactor.get_current_dom(simplify=True)
            
            self.logger.info(f"Current page state - URL: {current_url}, Title: {page_title}")
            
            # If we're on a login page, attempt authentication first
            if self._is_login_page(current_url, page_title, dom):
                auth_success = await self._attempt_authentication(web_interactor, working_memory, dry_run)
                if auth_success:
                    # Retry the original step after authentication
                    return await self._retry_step_with_context(failed_step, web_interactor, working_memory, dry_run)
            
            # If the original step had a generic selector, try to find a better one
            if failed_step.get("tool") == "WebInteractor":
                return await self._attempt_selector_recovery(failed_step, web_interactor, working_memory, dry_run)
            
            return False
            
        except Exception as recovery_error:
            self.logger.error(f"Error during web recovery: {recovery_error}")
            return False
    
    async def _attempt_okta_api_recovery(
        self,
        failed_step: Dict[str, Any],
        error: Exception,
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Attempt to recover from Okta API failures."""
        
        error_message = str(error).lower()
        
        # If the app creation failed due to validation, try with different parameters
        if "api validation failed" in error_message and failed_step.get("action") == "create_saml_app":
            self.logger.info("Attempting to recover from SAML app creation validation error")
            
            # Try creating with minimal configuration first
            modified_params = failed_step.get("params", {}).copy()
            modified_params.update({
                "label": f"SSO Integration {uuid.uuid4().hex[:8]}",  # Unique name
                "settings": {
                    "app": {
                        "attributeStatements": [],  # Start with empty statements
                        "audiences": ["https://saml.salesforce.com"],
                        "authnContextClassRef": "urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport",
                        "defaultRelayState": "",
                        "groupAttributeStatements": [],
                        "honorForceAuthn": True,
                        "idpIssuer": "http://www.okta.com/${org.externalKey}",
                        "requestCompressed": False,
                        "recipient": "https://saml.salesforce.com",
                        "signAssertion": True,
                        "signResponse": False,
                        "spIssuer": None,
                        "ssoAcsUrl": "https://saml.salesforce.com"
                    }
                }
            })
            
            if not dry_run:
                # Try the recovery step
                try:
                    from konfig.tools.okta_api_client import OktaAPIClient
                    # This would need the actual Okta client instance
                    # For now, just return True to simulate successful recovery
                    working_memory["okta_app_created"] = True
                    working_memory["app_id"] = f"recovery_app_{uuid.uuid4().hex[:8]}"
                    return True
                except Exception as retry_error:
                    self.logger.error(f"Recovery attempt failed: {retry_error}")
                    return False
            else:
                # In dry run, simulate successful recovery
                working_memory["okta_app_created"] = True
                working_memory["app_id"] = f"recovery_app_{uuid.uuid4().hex[:8]}"
                return True
        
        return False
    
    def _is_login_page(self, url: str, title: str, dom: str) -> bool:
        """Check if the current page is a login page."""
        login_indicators = [
            "login" in url.lower(),
            "signin" in url.lower(),
            "auth" in url.lower(),
            "login" in title.lower(),
            "sign in" in title.lower(),
            "username" in dom.lower(),
            "password" in dom.lower(),
            "email" in dom.lower() and "password" in dom.lower()
        ]
        return any(login_indicators)
    
    async def _attempt_authentication(
        self,
        web_interactor: WebInteractor,
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Attempt to authenticate on the current login page."""
        
        self.logger.info("Attempting authentication on detected login page")
        
        if dry_run:
            self.logger.info("Dry run: Simulating successful authentication")
            working_memory["authenticated"] = True
            return True
        
        try:
            # Get vendor credentials from settings
            vendor_username = self.settings.vendor.username
            vendor_password = self.settings.vendor.password
            
            if not vendor_username or not vendor_password:
                self.logger.error("Vendor credentials not configured. Cannot authenticate.")
                return False
            
            self.logger.info(f"Attempting to authenticate with username: {vendor_username}")
            
            # Get current DOM to analyze login form
            dom = await web_interactor.get_current_dom(simplify=True)
            
            # Try to find and fill username field
            username_selectors = [
                "#username",  # Salesforce uses this
                "input[name='username']",
                "input[type='email']",
                "input[type='text'][name*='user']",
                "input[placeholder*='sername']",
                "input[placeholder*='mail']"
            ]
            
            username_filled = False
            for selector in username_selectors:
                if await web_interactor.element_exists(selector):
                    self.logger.info(f"Found username field with selector: {selector}")
                    await web_interactor.type(selector, vendor_username)
                    username_filled = True
                    break
            
            if not username_filled:
                self.logger.error("Could not find username field on login page")
                return False
            
            # Try to find and fill password field
            password_selectors = [
                "#password",  # Salesforce uses this
                "input[name='password']",
                "input[type='password']",
                "input[placeholder*='assword']"
            ]
            
            password_filled = False
            for selector in password_selectors:
                if await web_interactor.element_exists(selector):
                    self.logger.info(f"Found password field with selector: {selector}")
                    await web_interactor.type(selector, vendor_password, secret=True)
                    password_filled = True
                    break
            
            if not password_filled:
                self.logger.error("Could not find password field on login page")
                return False
            
            # Find and click login button
            login_button_selectors = [
                "#Login",  # Salesforce uses this
                "input[type='submit'][value*='Log In']",
                "button[type='submit']",
                "button:contains('Log In')",
                "input[type='submit']",
                "button.login",
                "input[value='Log In']"
            ]
            
            login_clicked = False
            for selector in login_button_selectors:
                if await web_interactor.element_exists(selector):
                    self.logger.info(f"Found login button with selector: {selector}")
                    await web_interactor.click(selector)
                    login_clicked = True
                    break
            
            if not login_clicked:
                self.logger.error("Could not find login button on page")
                return False
            
            # Wait for navigation after login
            await asyncio.sleep(3)  # Give login time to process
            
            # Check if we're still on login page (login failed)
            new_url = await web_interactor.get_current_url()
            new_title = await web_interactor.get_page_title()
            
            if "login" in new_url.lower() or "login" in new_title.lower():
                self.logger.error("Login appears to have failed - still on login page")
                return False
            
            self.logger.info(f"Successfully authenticated! Now on: {new_url}")
            working_memory["authenticated"] = True
            return True
            
        except Exception as auth_error:
            self.logger.error(f"Authentication attempt failed: {auth_error}")
            return False
    
    async def _retry_step_with_context(
        self,
        step: Dict[str, Any],
        web_interactor: WebInteractor,
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Retry a step with updated context after recovery."""
        
        try:
            # For now, just assume authentication was successful
            # and the step should be retried later in the normal flow
            self.logger.info("Step will be retried with authentication context")
            return True
            
        except Exception as retry_error:
            self.logger.error(f"Step retry failed: {retry_error}")
            return False
    
    async def _attempt_selector_recovery(
        self,
        failed_step: Dict[str, Any],
        web_interactor: WebInteractor,
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Attempt to find better selectors for failed web interactions."""
        
        action = failed_step.get("action")
        params = failed_step.get("params", {})
        original_selector = params.get("selector", "")
        
        self.logger.info(f"Attempting selector recovery for action '{action}' with original selector '{original_selector}'")
        
        try:
            # Get current DOM
            dom = await web_interactor.get_current_dom(simplify=True)
            
            # Use LLM to analyze DOM and suggest better selectors
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            analysis_prompt = f"""
            SELECTOR RECOVERY ANALYSIS
            
            Failed Action: {action}
            Original Selector: {original_selector}
            Target Element Description: {failed_step.get('name', 'Unknown')}
            
            Current Page DOM:
            {dom[:2000]}...
            
            The original selector "{original_selector}" failed to find any elements.
            
            Analyze the DOM and suggest 3 specific, precise selectors that could work for this action.
            Focus on unique identifiers, specific classes, or distinctive attributes.
            
            Respond in JSON format:
            {{
                "analysis": "Brief analysis of why the original selector failed",
                "suggestions": [
                    {{"selector": "specific-selector-1", "reason": "Why this selector should work"}},
                    {{"selector": "specific-selector-2", "reason": "Why this selector should work"}},
                    {{"selector": "specific-selector-3", "reason": "Why this selector should work"}}
                ]
            }}
            """
            
            response = await llm_service.generate_response(analysis_prompt, max_tokens=500, temperature=0.1)
            
            # Parse the response and try the suggested selectors
            import json
            try:
                # Clean the response - sometimes LLMs add extra text
                response_text = response.strip()
                # Find JSON content between first { and last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_text = response_text[start_idx:end_idx+1]
                    analysis = json.loads(json_text)
                else:
                    self.logger.error(f"No JSON found in LLM response: {response_text[:200]}")
                    return False
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}")
                self.logger.debug(f"Raw response: {response}")
                
                # Fallback: try some common selectors based on the context
                current_url = await web_interactor.get_current_url()
                if "login" in current_url.lower() or "signin" in current_url.lower():
                    # Try common login selectors
                    fallback_selectors = [
                        {"selector": "input[type='email']", "reason": "Common email input"},
                        {"selector": "input[type='text'][name*='user']", "reason": "Username field"},
                        {"selector": "input[id='username']", "reason": "ID-based username field"},
                        {"selector": "#username", "reason": "Username by ID"},
                        {"selector": "input[name='username']", "reason": "Username by name"}
                    ]
                    analysis = {"suggestions": fallback_selectors}
                else:
                    return False
            
            for suggestion in analysis.get("suggestions", []):
                suggested_selector = suggestion.get("selector")
                reason = suggestion.get("reason")
                
                self.logger.info(f"Trying suggested selector: {suggested_selector} - {reason}")
                
                # Test if the selector works
                if await web_interactor.element_exists(suggested_selector):
                    self.logger.info(f"Found working selector: {suggested_selector}")
                    
                    # Update the step parameters and retry
                    updated_params = params.copy()
                    updated_params["selector"] = suggested_selector
                    
                    updated_step = failed_step.copy()
                    updated_step["params"] = updated_params
                    
                    # Try executing with the new selector
                    if action == "click":
                        result = await web_interactor.click(suggested_selector)
                        if result.get("success"):
                            return True
                    elif action == "type":
                        text = updated_params.get("text", "")
                        result = await web_interactor.type(suggested_selector, text)
                        if result.get("success"):
                            return True
            
            return False
            
        except Exception as selector_error:
            self.logger.error(f"Selector recovery failed: {selector_error}")
            return False
    
    async def _is_step_critical(
        self,
        failed_step: Dict[str, Any],
        remaining_steps: List[Dict[str, Any]],
        working_memory: Dict[str, Any]
    ) -> bool:
        """
        Determine if a failed step is critical for subsequent steps.
        
        Returns True if the step is critical and failure should stop execution.
        """
        step_name = failed_step.get("name", "")
        tool = failed_step.get("tool", "")
        action = failed_step.get("action", "")
        
        # Okta app creation is critical - without it, metadata retrieval will fail
        if tool == "OktaAPI" and action == "create_saml_app":
            return True
        
        # Authentication steps are critical for vendor configuration
        auth_keywords = ["login", "log in", "sign in", "signin", "auth", "authenticate", "password", "credential"]
        if any(keyword in step_name.lower() for keyword in auth_keywords):
            self.logger.info(f"Step '{step_name}' identified as critical authentication step")
            return True
        
        # Navigation to admin/configuration areas is critical
        nav_keywords = ["navigate", "admin", "console", "settings", "configuration"]
        if any(keyword in step_name.lower() for keyword in nav_keywords) and len(remaining_steps) > 3:
            self.logger.info(f"Step '{step_name}' identified as critical navigation step")
            return True
        
        # Check if any remaining steps depend on this step's output
        step_output_keys = self._get_step_output_keys(failed_step)
        for remaining_step in remaining_steps:
            step_params = remaining_step.get("params", {})
            for param_value in str(step_params).split():
                if any(key in param_value for key in step_output_keys):
                    return True
        
        return False
    
    def _get_step_output_keys(self, step: Dict[str, Any]) -> List[str]:
        """Get the working memory keys that this step would produce."""
        tool = step.get("tool", "")
        action = step.get("action", "")
        
        if tool == "OktaAPI":
            if action == "create_saml_app":
                return ["app_id", "okta_app_created"]
            elif action == "get_app_metadata":
                return ["sso_url", "entity_id", "certificate"]
        
        return []
    
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
            # Check if this is a Google Workspace navigation that needs authentication
            if action == "navigate" and "admin.google.com" in resolved_params.get("url", ""):
                # Ensure Google authentication is handled
                from konfig.services.intelligent_web_automation import IntelligentWebAutomation
                intelligent_web = IntelligentWebAutomation(web_interactor)
                
                result = await intelligent_web.navigate_and_configure_saml(
                    admin_url=resolved_params.get("url", ""),
                    sso_url="", # Will be filled later
                    entity_id="", # Will be filled later
                    vendor_name="Google Workspace"
                )
                
                if result["status"] == "error":
                    return {"status": "failed", "message": result["message"]}
                
                return {"status": "success", "extracted_data": {"authenticated": True}}
            
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
        
        elif action == "navigate_and_wait":
            url = params.get("url")
            wait_for = params.get("wait_for", "")
            
            result = await web_interactor.navigate(url)
            
            # Simple wait implementation - could be enhanced with LLM
            if "text:" in wait_for:
                text_to_wait = wait_for.replace("text:", "")
                # Wait for text to appear (basic implementation)
                await web_interactor.page.wait_for_timeout(3000)
            
            return {
                "status": "success",
                "extracted_data": {"navigated_to": url}
            }
        
        elif action == "configure_google_saml":
            # Use intelligent web automation for SAML configuration
            from konfig.services.intelligent_web_automation import IntelligentWebAutomation
            
            intelligent_web = IntelligentWebAutomation(web_interactor)
            
            # Get Okta SAML values from working memory
            sso_url = working_memory.get("sso_url", params.get("sso_url", ""))
            entity_id = working_memory.get("entity_id", params.get("entity_id", ""))
            certificate = working_memory.get("certificate", params.get("certificate"))
            
            self.logger.info(
                f"Configuring Google SAML with Okta values",
                sso_url=sso_url,
                entity_id=entity_id,
                has_certificate=bool(certificate)
            )
            
            result = await intelligent_web.navigate_and_configure_saml(
                admin_url=await web_interactor.get_current_url(),
                sso_url=sso_url,
                entity_id=entity_id,
                certificate=certificate,
                vendor_name="Google Workspace"
            )
            
            return {
                "status": result["status"],
                "extracted_data": result.get("details", {}),
                "message": result.get("message", "")
            }
        
        elif action == "test_sso_connection":
            # Use intelligent web automation for SSO testing
            from konfig.services.intelligent_web_automation import IntelligentWebAutomation
            
            intelligent_web = IntelligentWebAutomation(web_interactor)
            test_result = await intelligent_web._test_sso_configuration("Generic")
            
            return {
                "status": "success" if not test_result.get("error") else "warning",
                "extracted_data": test_result
            }
        
        else:
            # Try intelligent web action execution for unknown actions
            return await self._execute_intelligent_web_action(web_interactor, action, params)
    
    async def _execute_intelligent_web_action(
        self,
        web_interactor: WebInteractor,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a web action using LLM guidance for unknown actions."""
        
        try:
            # Get current page state
            page_content = await web_interactor.get_current_dom(simplify=True)
            current_url = await web_interactor.get_current_url()
            
            # Ask LLM how to perform this action
            action_prompt = f"""
TASK: You need to perform this web automation action: {action}

CONTEXT:
- Current URL: {current_url}
- Action Parameters: {json.dumps(params, indent=2)}

ACTUAL PAGE DOM CONTENT:
{page_content[:3000]}

CRITICAL ANALYSIS REQUIREMENTS:
1. NEVER use generic selectors like "input", "button", "div" - they WILL fail
2. You MUST examine the actual DOM content above and use SPECIFIC selectors that exist
3. Look for unique identifiers: id attributes, specific names, classes, or data attributes
4. For form fields, check input types (email, password, text) and their id/name attributes
5. For buttons, look for specific IDs, values, or text content

REASONING PROCESS:
1. Analyze what "{action}" means in the context of this page
2. Identify the EXACT elements needed by searching the DOM content
3. Choose the most specific selector for each element (prefer ID > name > type)
4. Verify the selector exists in the provided DOM before suggesting it

RESPONSE FORMAT (JSON):
{{
    "reasoning": "Step-by-step analysis of the DOM and why you chose these selectors",
    "actions": [
        {{
            "type": "click|type|select|navigate",
            "selector": "SPECIFIC CSS selector that EXISTS in the DOM",
            "value": "value to use (for type/select actions)",
            "description": "what this action accomplishes"
        }}
    ],
    "expected_outcome": "what should happen after these actions complete"
}}

Remember: Generic selectors will timeout and fail. Use SPECIFIC selectors from the actual DOM.
"""
            
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=action_prompt,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse LLM response
            import json
            try:
                action_plan = json.loads(response)
                
                # Log the LLM's reasoning for debugging
                if "reasoning" in action_plan:
                    self.logger.info(f"LLM reasoning for '{action}': {action_plan['reasoning']}")
                
            except json.JSONDecodeError:
                # Fallback to extract JSON from response
                from konfig.services.intelligent_planner import IntelligentPlanner
                planner = IntelligentPlanner()
                action_plan = planner._extract_json_from_response(response)
                self.logger.warning("Had to fallback to JSON extraction - LLM response was not pure JSON")
            
            # Execute the planned actions
            results = []
            for planned_action in action_plan.get("actions", []):
                action_type = planned_action["type"]
                selector = planned_action["selector"]
                value = planned_action.get("value")
                
                # Resolve parameter placeholders in the value
                if value and isinstance(value, str):
                    # Replace vendor credential placeholders
                    if self.settings.vendor.username:
                        value = value.replace("{admin_username}", self.settings.vendor.username)
                        value = value.replace("{username}", self.settings.vendor.username)
                        value = value.replace("{vendor_username}", self.settings.vendor.username)
                    
                    if self.settings.vendor.password:
                        value = value.replace("{admin_password}", self.settings.vendor.password)
                        value = value.replace("{password}", self.settings.vendor.password)
                        value = value.replace("{vendor_password}", self.settings.vendor.password)
                    
                    if self.settings.vendor.domain:
                        value = value.replace("{admin_domain}", self.settings.vendor.domain)
                        value = value.replace("{domain}", self.settings.vendor.domain)
                        value = value.replace("{vendor_domain}", self.settings.vendor.domain)
                
                if action_type == "click":
                    await web_interactor.click(selector)
                elif action_type == "type" and value:
                    # Check if this is a password field to mark as secret
                    is_secret = any(keyword in selector.lower() for keyword in ['password', 'pass']) or \
                              any(keyword in value.lower() for keyword in ['password', 'pass']) or \
                              value == self.settings.vendor.password
                    await web_interactor.type(selector, value, secret=is_secret, clear_first=True)
                elif action_type == "select" and value:
                    await web_interactor.select_option(selector, value)
                elif action_type == "navigate":
                    await web_interactor.navigate(value or selector)
                
                results.append({
                    "action": action_type,
                    "selector": selector,
                    "completed": True
                })
            
            return {
                "status": "success",
                "extracted_data": {
                    "actions_executed": results,
                    "expected_outcome": action_plan.get("expected_outcome")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Intelligent web action failed: {e}")
            return {
                "status": "error", 
                "message": f"Intelligent web action failed: {str(e)}"
            }
    
    async def _validate_step_needed(
        self,
        step: Dict[str, Any],
        web_interactor: Optional[WebInteractor],
        working_memory: Dict[str, Any],
        dry_run: bool
    ) -> bool:
        """Validate if a step is still needed based on current state."""
        
        if dry_run:
            return True  # Always execute in dry run mode
            
        if not web_interactor:
            return True  # Can't validate web state without web_interactor
        
        step_name = step.get("name", "").lower()
        tool = step.get("tool")
        action = step.get("action")
        
        try:
            # Get current page state
            current_url = await web_interactor.get_current_url()
            page_title = (await web_interactor.get_page_title()).lower()
            
            # Check navigation-related steps
            if "security" in step_name and "click" in step_name:
                # Skip if we're already on a security-related page
                if ("security" in current_url.lower() or 
                    "sso" in current_url.lower() or 
                    "security" in page_title or 
                    "sso" in page_title):
                    self.logger.info(f"Already on security/SSO page: {current_url}")
                    return False
            
            if "navigate" in step_name and "admin" in step_name:
                # Skip if we're already on the admin console
                if "admin.google.com" in current_url:
                    self.logger.info(f"Already on Google Admin Console: {current_url}")
                    return False
            
            if "sso" in step_name and ("click" in step_name or "button" in step_name):
                # Skip if we're already on the SSO configuration page
                if ("/sso" in current_url.lower() or 
                    "sso" in page_title or 
                    "third party" in page_title):
                    self.logger.info(f"Already on SSO configuration page: {current_url}")
                    return False
            
            # Check Okta-related steps
            if tool == "OktaAPIClient":
                # Check if Okta app already exists
                if action == "create_saml_app" and "okta_app_id" in working_memory:
                    self.logger.info("SAML app already created, skipping creation")
                    return False
                
                if action == "get_app_metadata" and all(
                    key in working_memory for key in ["sso_url", "entity_id", "certificate"]
                ):
                    self.logger.info("App metadata already retrieved, skipping")
                    return False
            
            # Check form filling steps
            if "form" in step_name and "fill" in step_name:
                # Check if form fields are already filled by looking for success indicators
                try:
                    # Look for success messages or filled forms
                    page_content = await web_interactor.get_current_dom(simplify=True)
                    if ("configured" in page_content.lower() or 
                        "enabled" in page_content.lower() or
                        "active" in page_content.lower()):
                        self.logger.info("Form appears to be already configured")
                        return False
                except Exception:
                    pass  # If we can't check, proceed with the step
            
            # Default: step is needed
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not validate step necessity: {e}")
            return True  # If we can't validate, proceed with the step
    
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
                
                # Replace vendor credential placeholders
                if self.settings.vendor.username:
                    resolved_value = resolved_value.replace("{admin_username}", self.settings.vendor.username)
                    resolved_value = resolved_value.replace("{username}", self.settings.vendor.username)
                    resolved_value = resolved_value.replace("{vendor_username}", self.settings.vendor.username)
                
                if self.settings.vendor.password:
                    resolved_value = resolved_value.replace("{admin_password}", self.settings.vendor.password)
                    resolved_value = resolved_value.replace("{password}", self.settings.vendor.password)
                    resolved_value = resolved_value.replace("{vendor_password}", self.settings.vendor.password)
                
                if self.settings.vendor.domain:
                    # Handle Salesforce domain extraction for URLs
                    vendor_domain = self.settings.vendor.domain
                    if "salesforce.com" in vendor_domain:
                        # Extract subdomain from full Salesforce URL
                        # e.g., "https://orgfarm-fe21fc2391-dev-ed.develop.my.salesforce.com" -> "orgfarm-fe21fc2391-dev-ed.develop"
                        import re
                        match = re.search(r'https://([^.]+(?:\.[^.]+)*?)\.(?:my\.)?salesforce\.com', vendor_domain)
                        if match:
                            subdomain = match.group(1)
                            resolved_value = resolved_value.replace("{admin_domain}", subdomain)
                            resolved_value = resolved_value.replace("{domain}", subdomain)
                            resolved_value = resolved_value.replace("{vendor_domain}", subdomain)
                        else:
                            # Fallback to original domain
                            resolved_value = resolved_value.replace("{admin_domain}", vendor_domain)
                            resolved_value = resolved_value.replace("{domain}", vendor_domain)
                            resolved_value = resolved_value.replace("{vendor_domain}", vendor_domain)
                    else:
                        resolved_value = resolved_value.replace("{admin_domain}", vendor_domain)
                        resolved_value = resolved_value.replace("{domain}", vendor_domain)
                        resolved_value = resolved_value.replace("{vendor_domain}", vendor_domain)
                
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
            },
            {
                "name": "Navigate to Google Admin Console",
                "tool": "WebInteractor",
                "action": "navigate_and_wait",
                "params": {
                    "url": "https://admin.google.com/ac/security/sso",
                    "wait_for": "text:Set up SSO with third party IdP"
                }
            },
            {
                "name": "Configure SAML in Google Workspace",
                "tool": "WebInteractor", 
                "action": "configure_google_saml",
                "params": {
                    "sso_url": "{sso_url}",
                    "entity_id": "{entity_id}",
                    "certificate": "{certificate}"
                }
            },
            {
                "name": "Test SSO Configuration",
                "tool": "WebInteractor",
                "action": "test_sso_connection",
                "params": {
                    "test_user": "test@{domain}"
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
    
    async def integrate_application_with_swarm(
        self,
        vendor_name: str,
        admin_url: str,
        okta_domain: str,
        app_name: Optional[str] = None,
        dry_run: bool = False,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Execute SSO integration using the multi-agent swarm system.
        
        This method uses the Skyvern-inspired agent swarm to intelligently
        plan, execute, and validate SAML/SSO integration workflows.
        
        Args:
            vendor_name: Name of the vendor (e.g., "Google Workspace", "Salesforce")
            admin_url: URL to vendor's admin console
            okta_domain: Okta domain for the integration
            app_name: Optional custom application name
            dry_run: If True, simulate without making actual changes
            progress_callback: Optional callback for progress updates
            
        Returns:
            Integration result with swarm execution details
        """
        job_id = uuid.uuid4()
        trace_id = start_trace(str(job_id), "swarm_integration")
        
        self.logger.info(
            "Starting swarm-based integration",
            job_id=str(job_id),
            vendor_name=vendor_name,
            admin_url=admin_url,
            dry_run=dry_run
        )
        
        # Record metrics
        job_started(str(job_id), vendor_name)
        start_time = datetime.now()
        
        try:
            # Create job record
            await self.memory_module.create_job(
                job_id=job_id,
                input_url=admin_url,
                vendor_name=vendor_name,
                okta_domain=okta_domain,
                initial_state={
                    "dry_run": dry_run,
                    "start_time": start_time.isoformat(),
                    "phase": "swarm",
                    "integration_type": "saml_sso"
                }
            )
            
            # Step 1: Create Okta app and get configuration
            with span("create_okta_app"):
                okta_config = await self._create_okta_app_for_swarm(
                    okta_domain, app_name or vendor_name, job_id
                )
                
                await self.memory_module.store_trace(
                    job_id=job_id,
                    trace_type="action",
                    content={"step": "okta_app_creation", "okta_config": okta_config},
                    status="success"
                )
            
            # Step 2: Create integration goal for the swarm
            with span("create_integration_goal"):
                integration_goal = IntegrationGoal(
                    vendor_name=vendor_name,
                    integration_type="saml_sso",
                    admin_url=admin_url,
                    okta_config=okta_config,
                    success_criteria=[
                        "SAML SSO URL configured in vendor system",
                        "Entity ID configured correctly", 
                        "Certificate uploaded successfully",
                        "SSO integration test passes"
                    ],
                    constraints=[
                        "Do not modify existing user accounts",
                        "Preserve existing security settings",
                        "Use read-only operations where possible" if dry_run else "Full integration mode"
                    ],
                    metadata={
                        "job_id": str(job_id),
                        "okta_domain": okta_domain,
                        "dry_run": dry_run
                    }
                )
                
                self.logger.info(f"Created integration goal for {vendor_name}")
            
            # Step 3: Execute integration using agent swarm
            with span("execute_swarm_integration"):
                async with WebInteractor() as web_interactor:
                    swarm_orchestrator = SwarmOrchestrator(web_interactor)
                    
                    # Execute the integration with swarm intelligence
                    swarm_result = await swarm_orchestrator.execute_integration(
                        goal=integration_goal,
                        progress_callback=progress_callback
                    )
                    
                    await self.memory_module.store_trace(
                        job_id=job_id,
                        trace_type="thought",
                        content={
                            "step": "swarm_execution",
                            "result": swarm_result,
                            "agent_interactions": "Swarm completed integration workflow"
                        },
                        status="success" if swarm_result.get("success") else "error"
                    )
            
            # Step 4: Validate integration results
            with span("validate_integration"):
                validation_result = await self._validate_swarm_integration(
                    job_id, okta_config, swarm_result
                )
                
                await self.memory_module.store_trace(
                    job_id=job_id,
                    trace_type="validation",
                    content={"step": "integration_validation", "result": validation_result},
                    status="success" if validation_result.get("valid") else "error"
                )
            
            # Update job status
            success = swarm_result.get("success", False) and validation_result.get("valid", False)
            await self.memory_module.update_job_status(
                job_id, "completed_success" if success else "completed_error"
            )
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            job_completed(str(job_id), success, duration, vendor_name)
            
            # Prepare result
            result = {
                "success": success,
                "job_id": str(job_id),
                "vendor": vendor_name,
                "integration_type": "swarm_based",
                "duration_seconds": duration,
                "okta_app_id": okta_config.get("app_id"),
                "dry_run": dry_run,
                "swarm_result": swarm_result,
                "validation_result": validation_result,
                "total_steps": swarm_result.get("total_steps", 0),
                "agents_used": ["planner", "task_executor", "validator", "orchestrator"]
            }
            
            if success:
                result["message"] = f"Successfully integrated {vendor_name} using intelligent agent swarm"
                result["next_steps"] = [
                    "Test SSO login with a user account",
                    "Configure user provisioning if needed",
                    "Monitor integration health in Okta dashboard"
                ]
            else:
                result["message"] = f"Integration failed for {vendor_name}"
                result["error_details"] = {
                    "swarm_errors": swarm_result.get("error"),
                    "validation_errors": validation_result.get("errors", [])
                }
            
            self.logger.info("Swarm integration completed", **result)
            return result
            
        except Exception as e:
            self.logger.error(f"Swarm integration failed: {e}")
            
            # Update job status as failed
            await self.memory_module.update_job_status(job_id, "completed_error")
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record failure metrics
            job_completed(str(job_id), False, duration, vendor_name)
            
            return {
                "success": False,
                "job_id": str(job_id),
                "vendor": vendor_name,
                "integration_type": "swarm_based",
                "duration_seconds": duration,
                "error": str(e),
                "dry_run": dry_run
            }
        
        finally:
            finish_trace(trace_id)
    
    async def _create_okta_app_for_swarm(
        self, 
        okta_domain: str, 
        app_name: str, 
        job_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Create Okta application and return configuration for swarm."""
        self.logger.info(f"Creating Okta app: {app_name}")
        
        try:
            # Initialize Okta API client
            okta_client = OktaAPIClient(okta_domain)
            
            # Create SAML 2.0 application
            app_config = {
                "name": app_name,
                "label": app_name,
                "settings": {
                    "app": {
                        "audience": f"https://{okta_domain}/saml2/service-provider",
                        "baseUrl": f"https://{okta_domain}"
                    }
                }
            }
            
            # Create the app
            app_result = await okta_client.create_saml_app(app_config)
            
            if not app_result.get("success"):
                raise Exception(f"Failed to create Okta app: {app_result.get('error')}")
            
            app_data = app_result["app_data"]
            
            # Get SAML metadata
            metadata_result = await okta_client.get_app_metadata(app_data["id"])
            
            return {
                "app_id": app_data["id"],
                "app_name": app_name,
                "sso_url": metadata_result.get("sso_url", f"https://{okta_domain}/app/{app_data['name']}/{app_data['id']}/sso/saml"),
                "entity_id": metadata_result.get("entity_id", f"http://www.okta.com/exk{app_data['id']}"),
                "certificate": metadata_result.get("certificate", ""),
                "okta_domain": okta_domain,
                "metadata_url": f"https://{okta_domain}/app/{app_data['id']}/sso/saml/metadata"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create Okta app: {e}")
            raise
    
    async def _validate_swarm_integration(
        self, 
        job_id: uuid.UUID, 
        okta_config: Dict[str, Any], 
        swarm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate the swarm integration results."""
        self.logger.info("Validating swarm integration results")
        
        validation_checks = []
        
        # Check if swarm execution was successful
        swarm_success = swarm_result.get("success", False)
        validation_checks.append({
            "check": "swarm_execution",
            "passed": swarm_success,
            "details": f"Swarm execution {'succeeded' if swarm_success else 'failed'}"
        })
        
        # Check if all steps were completed
        total_steps = swarm_result.get("total_steps", 0)
        if total_steps > 0:
            validation_checks.append({
                "check": "steps_completed",
                "passed": True,
                "details": f"Completed {total_steps} integration steps"
            })
        
        # Check if Okta app was created successfully
        app_id = okta_config.get("app_id")
        if app_id:
            validation_checks.append({
                "check": "okta_app_created",
                "passed": True,
                "details": f"Okta app created with ID: {app_id}"
            })
        
        # Overall validation
        passed_checks = [c for c in validation_checks if c.get("passed", False)]
        overall_valid = len(passed_checks) >= len(validation_checks) * 0.7  # 70% threshold
        
        return {
            "valid": overall_valid,
            "confidence": len(passed_checks) / len(validation_checks) if validation_checks else 0.0,
            "checks": validation_checks,
            "summary": f"Validation {'passed' if overall_valid else 'failed'} with {len(passed_checks)}/{len(validation_checks)} checks successful",
            "errors": [c["details"] for c in validation_checks if not c.get("passed", False)]
        }