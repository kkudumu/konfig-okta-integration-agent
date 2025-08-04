"""
Multi-Agent Swarm System for Okta Integration

Inspired by Skyvern's Task-Driven autonomous agent design, this system uses
a swarm of specialized agents to comprehend vendor websites, plan integration
steps, and execute SAML/SSO configuration workflows.

Architecture:
- PlannerAgent: Breaks down high-level integration goals into actionable steps
- TaskAgent: Executes individual steps using vision and DOM analysis
- ValidatorAgent: Validates step completion and provides feedback
- SwarmOrchestrator: Coordinates all agents and manages workflow
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from konfig.utils.logging import LoggingMixin


class AgentRole(Enum):
    """Defines the different agent roles in the swarm."""
    PLANNER = "planner"
    TASK_EXECUTOR = "task_executor"
    VALIDATOR = "validator"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_VALIDATION = "requires_validation"


@dataclass
class IntegrationGoal:
    """Represents a high-level integration goal."""
    vendor_name: str
    integration_type: str  # "saml_sso", "scim_provisioning", etc.
    admin_url: str
    okta_config: Dict[str, Any]
    success_criteria: List[str]
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskStep:
    """Represents a single step in the integration workflow."""
    step_id: str
    name: str
    description: str
    action_type: str  # "navigate", "click", "type", "extract", "validate"
    parameters: Dict[str, Any]
    expected_outcome: str
    validation_criteria: List[str]
    status: TaskStatus = TaskStatus.PENDING
    execution_result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Message passed between agents in the swarm."""
    sender: AgentRole
    receiver: AgentRole
    message_type: str
    content: Dict[str, Any]
    timestamp: str
    correlation_id: str


class BaseAgent(ABC, LoggingMixin):
    """Base class for all agents in the swarm."""
    
    def __init__(self, role: AgentRole, agent_id: str):
        super().__init__()
        self.setup_logging(f"agent_swarm_{role.value}")
        self.role = role
        self.agent_id = agent_id
        self.message_queue: List[AgentMessage] = []
        self.working_memory: Dict[str, Any] = {}
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process an incoming message and optionally return a response."""
        pass
    
    async def send_message(
        self, 
        receiver: AgentRole, 
        message_type: str, 
        content: Dict[str, Any],
        correlation_id: str
    ) -> AgentMessage:
        """Create a message to send to another agent."""
        import uuid
        from datetime import datetime
        
        message = AgentMessage(
            sender=self.role,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id
        )
        
        self.logger.debug(f"Sending message: {message_type} to {receiver.value}")
        return message


class PlannerAgent(BaseAgent):
    """
    Agent responsible for breaking down high-level integration goals into
    actionable steps. Uses LLM reasoning to create comprehensive execution plans.
    """
    
    def __init__(self, agent_id: str = "planner_001"):
        super().__init__(AgentRole.PLANNER, agent_id)
        self.integration_templates = self._load_integration_templates()
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process planning requests and feedback."""
        if message.message_type == "plan_integration":
            return await self._create_integration_plan(message)
        elif message.message_type == "replan_step":
            return await self._replan_failed_step(message)
        elif message.message_type == "update_plan":
            return await self._update_plan_with_feedback(message)
        return None
    
    async def _create_integration_plan(self, message: AgentMessage) -> AgentMessage:
        """Create a comprehensive integration plan from a high-level goal."""
        goal: IntegrationGoal = message.content.get("goal")
        
        self.logger.info(f"Creating integration plan for {goal.vendor_name} - {goal.integration_type}")
        
        try:
            # Use LLM to generate comprehensive plan
            plan_prompt = self._build_planning_prompt(goal)
            
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=plan_prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract and validate the plan
            plan_data = self._extract_plan_from_response(response)
            steps = self._create_task_steps(plan_data, goal)
            
            return await self.send_message(
                receiver=AgentRole.ORCHESTRATOR,
                message_type="plan_ready",
                content={
                    "goal_id": goal.metadata.get("goal_id"),
                    "steps": [self._step_to_dict(step) for step in steps],
                    "estimated_duration": plan_data.get("estimated_duration", "30-60 minutes"),
                    "complexity_score": plan_data.get("complexity_score", 0.7),
                    "risk_factors": plan_data.get("risk_factors", [])
                },
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create integration plan: {e}")
            return await self.send_message(
                receiver=AgentRole.ORCHESTRATOR,
                message_type="plan_failed",
                content={"error": str(e), "goal_id": goal.metadata.get("goal_id")},
                correlation_id=message.correlation_id
            )
    
    def _build_planning_prompt(self, goal: IntegrationGoal) -> str:
        """Build LLM prompt for integration planning."""
        return f"""
You are an expert at planning SAML/SSO integrations between Okta and various vendor systems.

INTEGRATION GOAL:
- Vendor: {goal.vendor_name}
- Type: {goal.integration_type}
- Admin URL: {goal.admin_url}
- Okta Configuration: {json.dumps(goal.okta_config, indent=2)}

SUCCESS CRITERIA:
{chr(10).join(f"- {criteria}" for criteria in goal.success_criteria)}

CONSTRAINTS:
{chr(10).join(f"- {constraint}" for constraint in goal.constraints)}

Create a detailed step-by-step plan to configure SAML SSO. Consider:
1. Authentication requirements
2. Navigation to SSO settings
3. SAML configuration fields
4. Certificate handling
5. Testing and validation
6. Common failure points and recovery

Respond with JSON:
{{
    "steps": [
        {{
            "step_id": "step_001",
            "name": "Navigate to vendor admin console",
            "description": "Navigate to the vendor's admin console and authenticate",
            "action_type": "navigate_and_authenticate",
            "parameters": {{
                "url": "{goal.admin_url}",
                "auth_method": "credentials"
            }},
            "expected_outcome": "Successfully authenticated to admin console",
            "validation_criteria": ["URL contains admin dashboard", "No login prompts visible"],
            "dependencies": []
        }},
        {{
            "step_id": "step_002", 
            "name": "Find SSO settings",
            "description": "Navigate to SAML/SSO configuration section",
            "action_type": "navigate_to_section",
            "parameters": {{
                "section": "sso_settings",
                "search_terms": ["SSO", "SAML", "Single Sign-on", "Identity Provider"]
            }},
            "expected_outcome": "On SAML configuration page",
            "validation_criteria": ["Page contains SAML configuration form", "SSO settings visible"],
            "dependencies": ["step_001"]
        }}
    ],
    "estimated_duration": "30-45 minutes",
    "complexity_score": 0.7,
    "risk_factors": ["Complex navigation structure", "Dynamic form fields", "Certificate validation"]
}}

Focus on {goal.vendor_name}-specific workflows and be very detailed about expected DOM elements and validation criteria.
"""
    
    def _extract_plan_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and validate plan from LLM response."""
        try:
            # Use the robust JSON extraction from intelligent planner
            from konfig.services.intelligent_planner import IntelligentPlanner
            planner = IntelligentPlanner()
            return planner._extract_json_from_response(response)
        except Exception as e:
            self.logger.error(f"Failed to extract plan from response: {e}")
            return {"steps": [], "estimated_duration": "Unknown", "complexity_score": 1.0}
    
    def _create_task_steps(self, plan_data: Dict[str, Any], goal: IntegrationGoal) -> List[TaskStep]:
        """Convert plan data into TaskStep objects."""
        steps = []
        for step_data in plan_data.get("steps", []):
            step = TaskStep(
                step_id=step_data.get("step_id", f"step_{len(steps)+1:03d}"),
                name=step_data.get("name", "Unnamed step"),
                description=step_data.get("description", ""),
                action_type=step_data.get("action_type", "manual"),
                parameters=step_data.get("parameters", {}),
                expected_outcome=step_data.get("expected_outcome", ""),
                validation_criteria=step_data.get("validation_criteria", []),
                dependencies=step_data.get("dependencies", [])
            )
            steps.append(step)
        return steps
    
    def _step_to_dict(self, step: TaskStep) -> Dict[str, Any]:
        """Convert TaskStep to dictionary for serialization."""
        return {
            "step_id": step.step_id,
            "name": step.name,
            "description": step.description,
            "action_type": step.action_type,
            "parameters": step.parameters,
            "expected_outcome": step.expected_outcome,
            "validation_criteria": step.validation_criteria,
            "status": step.status.value,
            "dependencies": step.dependencies,
            "retry_count": step.retry_count,
            "max_retries": step.max_retries
        }
    
    async def _replan_failed_step(self, message: AgentMessage) -> AgentMessage:
        """Replan a failed step with alternative approaches."""
        failed_step_data = message.content.get("failed_step")
        error_context = message.content.get("error_context", {})
        
        self.logger.info(f"Replanning failed step: {failed_step_data.get('step_id')}")
        
        # Use LLM to generate alternative approaches
        replan_prompt = f"""
The following step failed during integration:

FAILED STEP:
{json.dumps(failed_step_data, indent=2)}

ERROR CONTEXT:
{json.dumps(error_context, indent=2)}

Generate 2-3 alternative approaches to accomplish this step. Consider:
1. Different selectors or interaction methods
2. Alternative navigation paths
3. Fallback strategies

Respond with JSON:
{{
    "alternatives": [
        {{
            "approach": "alternative_1",
            "modifications": {{
                "parameters": {{"selector": "new_selector"}},
                "action_type": "modified_action"
            }},
            "reasoning": "Why this approach might work better"
        }}
    ]
}}
"""
        
        try:
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=replan_prompt,
                max_tokens=1000,
                temperature=0.2
            )
            
            alternatives = self._extract_plan_from_response(response)
            
            return await self.send_message(
                receiver=AgentRole.ORCHESTRATOR,
                message_type="replan_ready",
                content={
                    "step_id": failed_step_data.get("step_id"),
                    "alternatives": alternatives.get("alternatives", [])
                },
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Failed to replan step: {e}")
            return await self.send_message(
                receiver=AgentRole.ORCHESTRATOR,
                message_type="replan_failed",
                content={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _update_plan_with_feedback(self, message: AgentMessage) -> AgentMessage:
        """Update the plan based on validator feedback."""
        feedback = message.content.get("feedback", {})
        current_plan = message.content.get("current_plan", [])
        
        self.logger.info("Updating plan based on validator feedback")
        
        # Process feedback and adjust plan
        updated_steps = []
        for step_data in current_plan:
            step_feedback = feedback.get(step_data.get("step_id"), {})
            if step_feedback.get("requires_update", False):
                # Apply suggested modifications
                modifications = step_feedback.get("modifications", {})
                step_data.update(modifications)
            updated_steps.append(step_data)
        
        return await self.send_message(
            receiver=AgentRole.ORCHESTRATOR,
            message_type="plan_updated",
            content={"updated_steps": updated_steps},
            correlation_id=message.correlation_id
        )
    
    def _load_integration_templates(self) -> Dict[str, Any]:
        """Load predefined templates for common integrations."""
        return {
            "google_workspace": {
                "sso_paths": [
                    "/admin/security/sso",
                    "/admin/security",
                    "/admin/apps/unified"
                ],
                "common_selectors": {
                    "sso_url_field": ["input[name='ssoUrl']", "#sso-url", "input[placeholder*='Sign-in']"],
                    "entity_id_field": ["input[name='entityId']", "#entity-id", "input[placeholder*='Entity']"],
                    "certificate_field": ["textarea[name='certificate']", "#certificate", "textarea[placeholder*='Certificate']"]
                }
            },
            "salesforce": {
                "sso_paths": [
                    "/lightning/setup/SingleSignOn/home",
                    "/setup/SingleSignOn/home"
                ],
                "common_selectors": {
                    "new_saml_button": ["button:has-text('New')", "#new-saml", "a[title*='New']"]
                }
            }
        }


class TaskAgent(BaseAgent):
    """
    Agent responsible for executing individual task steps using vision and DOM analysis.
    Uses browser automation to interact with vendor websites intelligently.
    """
    
    def __init__(self, web_interactor, agent_id: str = "task_001"):
        super().__init__(AgentRole.TASK_EXECUTOR, agent_id)
        self.web_interactor = web_interactor
        self.execution_history: List[Dict[str, Any]] = []
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process task execution requests."""
        if message.message_type == "execute_step":
            return await self._execute_step(message)
        elif message.message_type == "retry_step":
            return await self._retry_step_with_alternatives(message)
        return None
    
    async def _execute_step(self, message: AgentMessage) -> AgentMessage:
        """Execute a single task step using intelligent web automation."""
        step_data = message.content.get("step")
        step = self._dict_to_step(step_data)
        
        self.logger.info(f"Executing step: {step.step_id} - {step.name}")
        
        try:
            # Record execution start
            execution_context = {
                "step_id": step.step_id,
                "start_time": self._get_timestamp(),
                "initial_url": await self.web_interactor.get_current_url(),
                "initial_title": await self.web_interactor.get_page_title()
            }
            
            # Execute based on action type
            if step.action_type == "navigate_and_authenticate":
                result = await self._execute_navigation_and_auth(step)
            elif step.action_type == "navigate_to_section":
                result = await self._execute_navigation_to_section(step)
            elif step.action_type == "fill_saml_config":
                result = await self._execute_saml_configuration(step)
            elif step.action_type == "click_element":
                result = await self._execute_click(step)
            elif step.action_type == "extract_data":
                result = await self._execute_data_extraction(step)
            else:
                result = await self._execute_generic_action(step)
            
            # Record execution completion
            execution_context.update({
                "end_time": self._get_timestamp(),
                "final_url": await self.web_interactor.get_current_url(),
                "final_title": await self.web_interactor.get_page_title(),
                "result": result,
                "success": result.get("success", False)
            })
            
            self.execution_history.append(execution_context)
            
            # Update step status
            step.status = TaskStatus.COMPLETED if result.get("success") else TaskStatus.FAILED
            step.execution_result = result
            
            return await self.send_message(
                receiver=AgentRole.VALIDATOR,
                message_type="step_completed",
                content={
                    "step": self._step_to_dict(step),
                    "execution_context": execution_context,
                    "requires_validation": True
                },
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            step.status = TaskStatus.FAILED
            step.execution_result = {"success": False, "error": str(e)}
            
            return await self.send_message(
                receiver=AgentRole.ORCHESTRATOR,
                message_type="step_failed",
                content={
                    "step": self._step_to_dict(step),
                    "error": str(e),
                    "execution_history": self.execution_history[-5:]  # Last 5 executions for context
                },
                correlation_id=message.correlation_id
            )
    
    async def _execute_navigation_and_auth(self, step: TaskStep) -> Dict[str, Any]:
        """Execute navigation and authentication."""
        url = step.parameters.get("url")
        auth_method = step.parameters.get("auth_method", "credentials")
        
        self.logger.info(f"Navigating to {url} with auth method: {auth_method}")
        
        # Navigate to URL
        nav_result = await self.web_interactor.navigate(url)
        if not nav_result.get("success"):
            return {"success": False, "error": "Navigation failed", "details": nav_result}
        
        # Check if authentication is needed
        current_url = await self.web_interactor.get_current_url()
        login_indicators = ['login', 'signin', 'sign-in', 'auth', 'authenticate']
        needs_auth = any(indicator in current_url.lower() for indicator in login_indicators)
        
        if needs_auth and auth_method == "credentials":
            # Use the existing authentication system
            from konfig.orchestrator.mvp_orchestrator import MVPOrchestrator
            orchestrator = MVPOrchestrator()
            
            auth_success = await orchestrator._attempt_authentication(
                self.web_interactor, {}, dry_run=False
            )
            
            if not auth_success:
                return {"success": False, "error": "Authentication failed"}
        
        return {
            "success": True,
            "final_url": await self.web_interactor.get_current_url(),
            "authenticated": needs_auth
        }
    
    async def _execute_navigation_to_section(self, step: TaskStep) -> Dict[str, Any]:
        """Execute navigation to a specific section."""
        section = step.parameters.get("section")
        search_terms = step.parameters.get("search_terms", [])
        
        self.logger.info(f"Navigating to section: {section}")
        
        # Use intelligent web automation to find and navigate to section
        from konfig.services.intelligent_web_automation import IntelligentWebAutomation
        intelligent_web = IntelligentWebAutomation(self.web_interactor)
        
        if section == "sso_settings":
            # Use the existing SSO navigation logic
            sso_url = await intelligent_web._find_sso_settings_page("generic")
            if sso_url:
                return {"success": True, "sso_url": sso_url}
            else:
                return {"success": False, "error": "Could not find SSO settings"}
        
        # Generic section navigation
        for term in search_terms:
            try:
                # Try clicking on navigation elements containing the term
                success = await intelligent_web._smart_click_by_text(term)
                if success:
                    await asyncio.sleep(2)  # Wait for navigation
                    return {
                        "success": True,
                        "navigation_term": term,
                        "final_url": await self.web_interactor.get_current_url()
                    }
            except Exception as e:
                self.logger.warning(f"Failed to navigate using term '{term}': {e}")
                continue
        
        return {"success": False, "error": "Could not navigate to section"}
    
    async def _execute_saml_configuration(self, step: TaskStep) -> Dict[str, Any]:
        """Execute SAML configuration form filling."""
        config_data = step.parameters.get("saml_config", {})
        
        self.logger.info("Executing SAML configuration")
        
        # Use intelligent web automation for SAML configuration
        from konfig.services.intelligent_web_automation import IntelligentWebAutomation
        intelligent_web = IntelligentWebAutomation(self.web_interactor)
        
        result = await intelligent_web._configure_saml_settings(
            sso_url=config_data.get("sso_url", ""),
            entity_id=config_data.get("entity_id", ""),
            certificate=config_data.get("certificate"),
            vendor_name=config_data.get("vendor_name", "Unknown")
        )
        
        return {
            "success": len(result.get("configured_fields", [])) > 0,
            "configured_fields": result.get("configured_fields", []),
            "confidence": result.get("confidence", 0.5)
        }
    
    async def _execute_click(self, step: TaskStep) -> Dict[str, Any]:
        """Execute a click action."""
        selector = step.parameters.get("selector")
        text_content = step.parameters.get("text_content")
        
        if selector:
            result = await self.web_interactor.click(selector, text_content=text_content)
            return {"success": result.get("success", False), "method": "selector"}
        elif text_content:
            # Use intelligent clicking by text
            from konfig.services.intelligent_web_automation import IntelligentWebAutomation
            intelligent_web = IntelligentWebAutomation(self.web_interactor)
            success = await intelligent_web._smart_click_by_text(text_content)
            return {"success": success, "method": "text_search"}
        
        return {"success": False, "error": "No selector or text provided"}
    
    async def _execute_data_extraction(self, step: TaskStep) -> Dict[str, Any]:
        """Execute data extraction from the current page."""
        extraction_type = step.parameters.get("extraction_type", "text")
        selectors = step.parameters.get("selectors", [])
        
        extracted_data = {}
        
        if extraction_type == "text":
            for selector in selectors:
                try:
                    element = await self.web_interactor.find_element(selector)
                    if element:
                        text = await element.text_content()
                        extracted_data[selector] = text
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from {selector}: {e}")
        
        return {
            "success": len(extracted_data) > 0,
            "extracted_data": extracted_data
        }
    
    async def _execute_generic_action(self, step: TaskStep) -> Dict[str, Any]:
        """Execute a generic action using adaptive intelligence."""
        self.logger.info(f"Executing generic action: {step.action_type}")
        
        # Create an action dictionary for adaptive intelligence
        action = {
            "name": step.name,
            "tool": "WebInteractor",
            "action": step.action_type,
            "params": step.parameters
        }
        
        try:
            from konfig.services.adaptive_intelligence import AdaptiveIntelligence
            from konfig.services.llm_service import LLMService
            
            llm_service = LLMService()
            adaptive_intelligence = AdaptiveIntelligence(self.web_interactor, llm_service)
            
            result = await adaptive_intelligence.execute_with_adaptive_intelligence(
                action, {"working_memory": self.working_memory}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generic action execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _dict_to_step(self, step_data: Dict[str, Any]) -> TaskStep:
        """Convert dictionary to TaskStep object."""
        return TaskStep(
            step_id=step_data.get("step_id", ""),
            name=step_data.get("name", ""),
            description=step_data.get("description", ""),
            action_type=step_data.get("action_type", ""),
            parameters=step_data.get("parameters", {}),
            expected_outcome=step_data.get("expected_outcome", ""),
            validation_criteria=step_data.get("validation_criteria", []),
            status=TaskStatus(step_data.get("status", "pending")),
            dependencies=step_data.get("dependencies", []),
            retry_count=step_data.get("retry_count", 0),
            max_retries=step_data.get("max_retries", 3)
        )
    
    def _step_to_dict(self, step: TaskStep) -> Dict[str, Any]:
        """Convert TaskStep to dictionary."""
        return {
            "step_id": step.step_id,
            "name": step.name,
            "description": step.description,
            "action_type": step.action_type,
            "parameters": step.parameters,
            "expected_outcome": step.expected_outcome,
            "validation_criteria": step.validation_criteria,
            "status": step.status.value,
            "execution_result": step.execution_result,
            "dependencies": step.dependencies,
            "retry_count": step.retry_count,
            "max_retries": step.max_retries
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


class ValidatorAgent(BaseAgent):
    """
    Agent responsible for validating step completion and providing feedback
    to improve the integration process.
    """
    
    def __init__(self, web_interactor, agent_id: str = "validator_001"):
        super().__init__(AgentRole.VALIDATOR, agent_id)
        self.web_interactor = web_interactor
        self.validation_history: List[Dict[str, Any]] = []
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process validation requests."""
        if message.message_type == "step_completed":
            return await self._validate_step_completion(message)
        elif message.message_type == "validate_integration":
            return await self._validate_full_integration(message)
        return None
    
    async def _validate_step_completion(self, message: AgentMessage) -> AgentMessage:
        """Validate that a step was completed successfully."""
        step_data = message.content.get("step")
        execution_context = message.content.get("execution_context", {})
        
        self.logger.info(f"Validating step completion: {step_data.get('step_id')}")
        
        try:
            # Perform validation checks
            validation_result = await self._perform_validation_checks(step_data, execution_context)
            
            # Record validation
            validation_record = {
                "step_id": step_data.get("step_id"),
                "timestamp": self._get_timestamp(),
                "validation_result": validation_result,
                "confidence": validation_result.get("confidence", 0.5)
            }
            self.validation_history.append(validation_record)
            
            if validation_result.get("passed", False):
                return await self.send_message(
                    receiver=AgentRole.ORCHESTRATOR,
                    message_type="step_validated",
                    content={
                        "step_id": step_data.get("step_id"),
                        "validation_result": validation_result,
                        "next_step_ready": True
                    },
                    correlation_id=message.correlation_id
                )
            else:
                return await self.send_message(
                    receiver=AgentRole.ORCHESTRATOR,
                    message_type="step_validation_failed",
                    content={
                        "step_id": step_data.get("step_id"),
                        "validation_result": validation_result,
                        "suggested_actions": validation_result.get("suggested_actions", [])
                    },
                    correlation_id=message.correlation_id
                )
                
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return await self.send_message(
                receiver=AgentRole.ORCHESTRATOR,
                message_type="validation_error",
                content={"error": str(e), "step_id": step_data.get("step_id")},
                correlation_id=message.correlation_id
            )
    
    async def _perform_validation_checks(
        self, 
        step_data: Dict[str, Any], 
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive validation checks."""
        validation_criteria = step_data.get("validation_criteria", [])
        expected_outcome = step_data.get("expected_outcome", "")
        
        validation_results = []
        overall_confidence = 0.0
        
        # Check execution success
        execution_result = step_data.get("execution_result", {})
        if execution_result.get("success", False):
            validation_results.append({
                "check": "execution_success",
                "passed": True,
                "confidence": 0.8
            })
        else:
            validation_results.append({
                "check": "execution_success", 
                "passed": False,
                "confidence": 0.9,
                "reason": execution_result.get("error", "Execution failed")
            })
        
        # Check URL changes (if expected)
        initial_url = execution_context.get("initial_url", "")
        final_url = execution_context.get("final_url", "")
        
        if initial_url != final_url:
            validation_results.append({
                "check": "navigation_occurred",
                "passed": True,
                "confidence": 0.7,
                "details": {"from": initial_url, "to": final_url}
            })
        
        # Check page title changes
        initial_title = execution_context.get("initial_title", "")
        final_title = execution_context.get("final_title", "")
        
        if initial_title != final_title:
            validation_results.append({
                "check": "page_changed",
                "passed": True,
                "confidence": 0.6,
                "details": {"from": initial_title, "to": final_title}
            })
        
        # LLM-based validation using current page content
        llm_validation = await self._llm_based_validation(
            step_data, execution_context, validation_criteria, expected_outcome
        )
        validation_results.append(llm_validation)
        
        # Calculate overall confidence
        passed_checks = [r for r in validation_results if r.get("passed", False)]
        if passed_checks:
            overall_confidence = sum(r.get("confidence", 0.5) for r in passed_checks) / len(validation_results)
        
        overall_passed = len(passed_checks) >= len(validation_results) * 0.7  # 70% threshold
        
        return {
            "passed": overall_passed,
            "confidence": overall_confidence,
            "validation_results": validation_results,
            "suggested_actions": self._generate_suggested_actions(validation_results, step_data)
        }
    
    async def _llm_based_validation(
        self,
        step_data: Dict[str, Any],
        execution_context: Dict[str, Any], 
        validation_criteria: List[str],
        expected_outcome: str
    ) -> Dict[str, Any]:
        """Use LLM to validate step completion by analyzing page content."""
        try:
            # Get current page state
            current_url = await self.web_interactor.get_current_url()
            current_title = await self.web_interactor.get_page_title()
            dom_content = await self.web_interactor.get_current_dom(simplify=True)
            
            validation_prompt = f"""
Analyze the current page state to validate if the following step was completed successfully:

STEP DETAILS:
- Name: {step_data.get('name')}
- Action: {step_data.get('action_type')}
- Expected Outcome: {expected_outcome}

VALIDATION CRITERIA:
{chr(10).join(f"- {criteria}" for criteria in validation_criteria)}

CURRENT PAGE STATE:
- URL: {current_url}
- Title: {current_title}
- DOM Content (first 2000 chars): {dom_content[:2000]}...

EXECUTION CONTEXT:
- Initial URL: {execution_context.get('initial_url')}
- Final URL: {execution_context.get('final_url')}
- Success reported: {execution_context.get('result', {}).get('success', False)}

Based on this information, determine if the step was completed successfully.

Respond with JSON:
{{
    "passed": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of validation decision",
    "evidence": ["List of evidence supporting the decision"],
    "concerns": ["List of any concerns or issues found"]
}}
"""
            
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=validation_prompt,
                max_tokens=800,
                temperature=0.1
            )
            
            # Extract validation result
            from konfig.services.intelligent_planner import IntelligentPlanner
            planner = IntelligentPlanner()
            validation_data = planner._extract_json_from_response(response)
            
            return {
                "check": "llm_validation",
                "passed": validation_data.get("passed", False),
                "confidence": validation_data.get("confidence", 0.5),
                "reasoning": validation_data.get("reasoning", ""),
                "evidence": validation_data.get("evidence", []),
                "concerns": validation_data.get("concerns", [])
            }
            
        except Exception as e:
            self.logger.error(f"LLM validation failed: {e}")
            return {
                "check": "llm_validation",
                "passed": False,
                "confidence": 0.1,
                "reasoning": f"Validation failed due to error: {str(e)}"
            }
    
    def _generate_suggested_actions(
        self, 
        validation_results: List[Dict[str, Any]], 
        step_data: Dict[str, Any]
    ) -> List[str]:
        """Generate suggested actions based on validation results."""
        suggestions = []
        
        failed_checks = [r for r in validation_results if not r.get("passed", False)]
        
        for check in failed_checks:
            check_type = check.get("check", "unknown")
            
            if check_type == "execution_success":
                suggestions.append("Retry step with alternative selectors or approach")
            elif check_type == "llm_validation":
                concerns = check.get("concerns", [])
                for concern in concerns:
                    suggestions.append(f"Address concern: {concern}")
            elif check_type == "navigation_occurred":
                suggestions.append("Verify correct navigation occurred")
        
        # Generic suggestions based on step type
        action_type = step_data.get("action_type", "")
        if action_type == "click_element" and not any(s for s in suggestions):
            suggestions.append("Try alternative selectors or click methods")
        elif action_type == "fill_saml_config":
            suggestions.append("Verify all required SAML fields were populated")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    async def _validate_full_integration(self, message: AgentMessage) -> AgentMessage:
        """Validate the entire integration workflow."""
        integration_data = message.content.get("integration_data", {})
        
        self.logger.info("Validating full integration")
        
        # Perform comprehensive integration validation
        # This could include testing SSO login, checking configuration persistence, etc.
        
        validation_result = {
            "integration_valid": True,
            "confidence": 0.8,
            "validation_summary": "Integration validation not yet implemented"
        }
        
        return await self.send_message(
            receiver=AgentRole.ORCHESTRATOR,
            message_type="integration_validated",
            content={"validation_result": validation_result},
            correlation_id=message.correlation_id
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


class SwarmOrchestrator(BaseAgent):
    """
    Main orchestrator that coordinates all agents in the swarm to accomplish
    integration goals using collaborative intelligence.
    """
    
    def __init__(self, web_interactor, agent_id: str = "orchestrator_001"):
        super().__init__(AgentRole.ORCHESTRATOR, agent_id)
        self.web_interactor = web_interactor
        
        # Initialize all agents
        self.planner = PlannerAgent()
        self.task_executor = TaskAgent(web_interactor)
        self.validator = ValidatorAgent(web_interactor)
        
        # Orchestration state
        self.active_integrations: Dict[str, Dict[str, Any]] = {}
        self.message_handlers = {
            "plan_ready": self._handle_plan_ready,
            "step_completed": self._handle_step_completed,
            "step_validated": self._handle_step_validated,
            "step_validation_failed": self._handle_step_validation_failed,
            "step_failed": self._handle_step_failed
        }
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process orchestration messages."""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            await handler(message)
            return None
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")
            return None
    
    async def execute_integration(
        self, 
        goal: IntegrationGoal,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Execute a complete integration using the agent swarm."""
        import uuid
        
        correlation_id = str(uuid.uuid4())
        goal.metadata["goal_id"] = correlation_id
        goal.metadata["correlation_id"] = correlation_id
        
        self.logger.info(f"Starting swarm integration for {goal.vendor_name}")
        
        # Initialize integration state
        self.active_integrations[correlation_id] = {
            "goal": goal,
            "status": "planning",
            "current_step": 0,
            "steps": [],
            "start_time": self._get_timestamp(),
            "progress_callback": progress_callback
        }
        
        try:
            # Step 1: Request planning from PlannerAgent
            plan_message = await self.send_message(
                receiver=AgentRole.PLANNER,
                message_type="plan_integration",
                content={"goal": goal},
                correlation_id=correlation_id
            )
            
            # Process the plan
            plan_response = await self.planner.process_message(plan_message)
            await self._handle_message(plan_response)
            
            # Wait for integration completion
            result = await self._wait_for_integration_completion(correlation_id)
            return result
            
        except Exception as e:
            self.logger.error(f"Integration execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "integration_id": correlation_id
            }
    
    async def _handle_message(self, message: AgentMessage):
        """Handle messages from other agents."""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            await handler(message)
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")
    
    async def _handle_plan_ready(self, message: AgentMessage):
        """Handle plan ready from PlannerAgent."""
        correlation_id = message.correlation_id
        integration = self.active_integrations.get(correlation_id)
        
        if not integration:
            return
        
        steps = message.content.get("steps", [])
        integration["steps"] = steps
        integration["status"] = "executing"
        integration["total_steps"] = len(steps)
        
        self.logger.info(f"Plan ready with {len(steps)} steps")
        
        # Start executing steps
        await self._execute_next_step(correlation_id)
    
    async def _execute_next_step(self, correlation_id: str):
        """Execute the next step in the integration."""
        integration = self.active_integrations.get(correlation_id)
        if not integration:
            return
        
        current_step_index = integration["current_step"]
        steps = integration["steps"]
        
        if current_step_index >= len(steps):
            # All steps completed
            await self._complete_integration(correlation_id)
            return
        
        step = steps[current_step_index]
        
        # Update progress
        if integration.get("progress_callback"):
            progress = (current_step_index + 1) / len(steps)
            integration["progress_callback"](
                f"Executing step {current_step_index + 1}/{len(steps)}: {step.get('name')}",
                progress
            )
        
        self.logger.info(f"Executing step {current_step_index + 1}/{len(steps)}: {step.get('name')}")
        
        # Send step to TaskAgent
        execute_message = await self.send_message(
            receiver=AgentRole.TASK_EXECUTOR,
            message_type="execute_step",
            content={"step": step},
            correlation_id=correlation_id
        )
        
        # Process the step execution
        execute_response = await self.task_executor.process_message(execute_message)
        await self._handle_message(execute_response)
    
    async def _handle_step_completed(self, message: AgentMessage):
        """Handle step completion from TaskAgent."""
        # Forward to ValidatorAgent for validation
        validation_response = await self.validator.process_message(message)
        await self._handle_message(validation_response)
    
    async def _handle_step_validated(self, message: AgentMessage):
        """Handle successful step validation."""
        correlation_id = message.correlation_id
        integration = self.active_integrations.get(correlation_id)
        
        if not integration:
            return
        
        step_id = message.content.get("step_id")
        self.logger.info(f"Step validated successfully: {step_id}")
        
        # Move to next step
        integration["current_step"] += 1
        await self._execute_next_step(correlation_id)
    
    async def _handle_step_validation_failed(self, message: AgentMessage):
        """Handle failed step validation."""
        correlation_id = message.correlation_id
        step_id = message.content.get("step_id")
        suggested_actions = message.content.get("suggested_actions", [])
        
        self.logger.warning(f"Step validation failed: {step_id}")
        
        # Request replanning from PlannerAgent
        replan_message = await self.send_message(
            receiver=AgentRole.PLANNER,
            message_type="replan_step", 
            content={
                "failed_step": message.content,
                "suggested_actions": suggested_actions
            },
            correlation_id=correlation_id
        )
        
        replan_response = await self.planner.process_message(replan_message)
        # Handle replan response...
    
    async def _handle_step_failed(self, message: AgentMessage):
        """Handle step execution failure."""
        correlation_id = message.correlation_id
        step_data = message.content.get("step", {})
        error = message.content.get("error", "Unknown error")
        
        self.logger.error(f"Step failed: {step_data.get('step_id')} - {error}")
        
        # Check retry count
        retry_count = step_data.get("retry_count", 0)
        max_retries = step_data.get("max_retries", 3)
        
        if retry_count < max_retries:
            # Retry the step
            step_data["retry_count"] = retry_count + 1
            
            retry_message = await self.send_message(
                receiver=AgentRole.TASK_EXECUTOR,
                message_type="execute_step",
                content={"step": step_data},
                correlation_id=correlation_id
            )
            
            retry_response = await self.task_executor.process_message(retry_message)
            await self._handle_message(retry_response)
        else:
            # Max retries reached, request replanning
            replan_message = await self.send_message(
                receiver=AgentRole.PLANNER,
                message_type="replan_step",
                content={
                    "failed_step": step_data,
                    "error_context": message.content.get("execution_history", [])
                },
                correlation_id=correlation_id
            )
            
            replan_response = await self.planner.process_message(replan_message)
            # Handle replan response...
    
    async def _complete_integration(self, correlation_id: str):
        """Complete the integration process."""
        integration = self.active_integrations.get(correlation_id)
        if not integration:
            return
        
        integration["status"] = "completed"
        integration["completion_time"] = self._get_timestamp()
        
        self.logger.info(f"Integration completed successfully: {correlation_id}")
        
        if integration.get("progress_callback"):
            integration["progress_callback"]("Integration completed successfully!", 1.0)
    
    async def _wait_for_integration_completion(self, correlation_id: str) -> Dict[str, Any]:
        """Wait for integration to complete and return results."""
        max_wait_time = 1800  # 30 minutes
        check_interval = 5  # 5 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            integration = self.active_integrations.get(correlation_id)
            if not integration:
                return {"success": False, "error": "Integration not found"}
            
            status = integration.get("status")
            if status == "completed":
                return {
                    "success": True,
                    "integration_id": correlation_id,
                    "total_steps": integration.get("total_steps", 0),
                    "duration": elapsed_time,
                    "vendor_name": integration["goal"].vendor_name
                }
            elif status == "failed":
                return {
                    "success": False,
                    "error": integration.get("error", "Integration failed"),
                    "integration_id": correlation_id
                }
            
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        return {"success": False, "error": "Integration timed out"}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()