"""
Cognition Module for Konfig.

This module provides LLM-powered planning and task decomposition capabilities,
enabling the agent to analyze documentation and generate executable plans for
SSO integration.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from konfig.config.settings import get_settings
from konfig.modules.memory.memory_module import MemoryModule
from konfig.utils.logging import LoggingMixin


class TaskStep(BaseModel):
    """Model for a single task step in the integration plan."""
    
    step_id: int = Field(description="Unique sequential identifier for this step")
    description: str = Field(description="Human-readable description of the task")
    tool_required: str = Field(description="Tool needed (WebInteractor, OktaAPIClient)")
    action_to_perform: str = Field(description="Specific action/method to call")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters for the action")
    outputs: Dict[str, str] = Field(default_factory=dict, description="Expected outputs to extract")
    dependencies: List[int] = Field(default_factory=list, description="Step IDs that must complete first")
    status: str = Field(default="pending", description="Step status")
    context: str = Field(description="Execution context (browser, okta_api, analysis)")
    
    class Config:
        schema_extra = {
            "example": {
                "step_id": 1,
                "description": "Create SAML application in Okta",
                "tool_required": "OktaAPIClient",
                "action_to_perform": "create_saml_app",
                "inputs": {"label": "Vendor SAML App"},
                "outputs": {"app_id": "okta_app_id", "sso_url": "sso_url"},
                "dependencies": [],
                "status": "pending",
                "context": "okta_api"
            }
        }


class IntegrationPlan(BaseModel):
    """Model for a complete integration plan."""
    
    vendor_name: str = Field(description="Name of the vendor application")
    documentation_url: str = Field(description="URL to the vendor's documentation")
    plan_version: str = Field(default="1.0", description="Plan version")
    total_steps: int = Field(description="Total number of steps in the plan")
    steps: List[TaskStep] = Field(description="Ordered list of task steps")
    estimated_duration_minutes: int = Field(default=30, description="Estimated completion time")
    
    class Config:
        schema_extra = {
            "description": "Complete integration plan with dependency graph"
        }


class CognitionModule(LoggingMixin):
    """
    LLM-powered cognition module for planning and reasoning.
    
    This module uses Large Language Models to:
    - Analyze vendor documentation
    - Generate step-by-step integration plans
    - Reason about actions and observations
    - Adapt plans based on feedback
    """
    
    def __init__(self):
        """Initialize the Cognition Module."""
        super().__init__()
        self.setup_logging("cognition")
        
        self.settings = get_settings()
        self.memory_module = MemoryModule()
        
        # Initialize LLM based on configuration
        self._llm = self._initialize_llm()
        
        # Initialize prompt templates
        self._init_prompts()
        
        self.logger.info("Cognition Module initialized")
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        if self.settings.llm.openai_api_key:
            self.logger.info("Using OpenAI LLM")
            return ChatOpenAI(
                model=self.settings.llm.openai_model,
                temperature=self.settings.llm.openai_temperature,
                max_tokens=self.settings.llm.openai_max_tokens,
                api_key=self.settings.llm.openai_api_key,
            )
        elif self.settings.llm.gemini_api_key:
            self.logger.info("Using Google Gemini LLM")
            return ChatGoogleGenerativeAI(
                model=self.settings.llm.gemini_model,
                temperature=self.settings.llm.gemini_temperature,
                max_tokens=self.settings.llm.gemini_max_tokens,
                google_api_key=self.settings.llm.gemini_api_key,
            )
        else:
            raise ValueError("No LLM API key configured")
    
    def _init_prompts(self):
        """Initialize prompt templates for different tasks."""
        # System prompt for the integration planner
        self.planner_system_prompt = """You are an expert SSO integration engineer specializing in SAML 2.0 configurations.
Your role is to analyze vendor documentation and generate detailed, executable plans for integrating applications with Okta.

You have access to two primary tools:
1. OktaAPIClient: For programmatic Okta operations (create apps, get metadata, assign users)
2. WebInteractor: For browser automation (navigate, click, type, select)

When creating plans:
- Break down complex tasks into atomic, verifiable steps
- Identify dependencies between steps
- Extract specific selectors, URLs, and field names from documentation
- Consider error cases and validation steps
- Ensure data flows correctly between steps (outputs -> inputs)

Output your plan as a structured JSON object following the provided schema."""

        # Prompt for analyzing documentation and creating plans
        self.plan_generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.planner_system_prompt),
            HumanMessagePromptTemplate.from_template("""Analyze the following vendor documentation and create a detailed integration plan.

Vendor: {vendor_name}
Documentation URL: {documentation_url}
Documentation Content:
{documentation_content}

Current Okta Domain: {okta_domain}

Create a step-by-step plan to:
1. Create a SAML application in Okta
2. Extract necessary metadata from Okta (Entity ID, SSO URL, Certificate)
3. Navigate to the vendor's admin interface
4. Configure SAML settings with Okta metadata
5. Extract vendor metadata (ACS URL, Audience) if needed
6. Update Okta application with vendor metadata
7. Test the integration

Output the plan as a JSON object with this structure:
{{
    "vendor_name": "string",
    "documentation_url": "string",
    "plan_version": "1.0",
    "total_steps": number,
    "steps": [
        {{
            "step_id": number,
            "description": "string",
            "tool_required": "OktaAPIClient|WebInteractor",
            "action_to_perform": "string",
            "inputs": {{}},
            "outputs": {{}},
            "dependencies": [],
            "status": "pending",
            "context": "okta_api|browser|analysis"
        }}
    ],
    "estimated_duration_minutes": number
}}""")
        ])

        # Prompt for reasoning about current state
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an integration engineer executing a SAML SSO configuration.
Analyze the current state and determine the next best action based on:
- What has been completed
- What data is available in working memory
- What errors have occurred
- What the plan specifies as next steps

Be adaptive - if the original plan needs adjustment based on observations, suggest modifications."""),
            HumanMessage(content="""Current State:
{current_state}

Working Memory:
{working_memory}

Original Plan:
{original_plan}

Last Action Result:
{last_result}

What should be the next action? Provide:
1. Reasoning for your decision
2. Specific action to take
3. Required inputs
4. Expected outputs""")
        ])

        # Prompt for error analysis and recovery
        self.error_recovery_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are debugging a failed SSO integration step.
Analyze the error and suggest recovery strategies based on:
- The specific error message
- The context (what was being attempted)
- Available alternative approaches
- Common issues with this type of integration"""),
            HumanMessage(content="""Error Context:
Action Attempted: {action}
Tool Used: {tool}
Inputs: {inputs}
Error Message: {error}

Current DOM/State (if browser action):
{current_state}

Suggest recovery strategies:
1. Immediate retry with modifications
2. Alternative approach
3. Required human intervention (if necessary)""")
        ])
    
    async def generate_integration_plan(
        self,
        vendor_name: str,
        documentation_url: str,
        documentation_content: str,
        okta_domain: str,
        existing_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> IntegrationPlan:
        """
        Generate a complete integration plan from documentation.
        
        Args:
            vendor_name: Name of the vendor application
            documentation_url: URL to the documentation
            documentation_content: Parsed documentation content
            okta_domain: Okta domain for the integration
            existing_patterns: Known patterns for this vendor
            
        Returns:
            IntegrationPlan object with detailed steps
        """
        self.log_method_call(
            "generate_integration_plan",
            vendor_name=vendor_name,
            doc_length=len(documentation_content)
        )
        
        try:
            # Build the prompt
            prompt_inputs = {
                "vendor_name": vendor_name,
                "documentation_url": documentation_url,
                "documentation_content": documentation_content[:8000],  # Limit context size
                "okta_domain": okta_domain
            }
            
            # Get LLM response
            messages = self.plan_generation_prompt.format_messages(**prompt_inputs)
            response = await self._llm.apredict_messages(messages)
            
            # Parse JSON response
            plan_json = self._extract_json_from_response(response.content)
            
            # Validate and create plan object
            plan = IntegrationPlan(**plan_json)
            
            # Apply known patterns if available
            if existing_patterns:
                plan = self._apply_known_patterns(plan, existing_patterns)
            
            self.logger.info(
                "Integration plan generated",
                vendor=vendor_name,
                total_steps=plan.total_steps
            )
            
            return plan
            
        except Exception as e:
            self.log_error("generate_integration_plan", e)
            # Return a basic fallback plan
            return self._create_fallback_plan(vendor_name, documentation_url)
    
    async def reason_next_action(
        self,
        current_state: Dict[str, Any],
        working_memory: Dict[str, Any],
        original_plan: List[Dict[str, Any]],
        last_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reason about the current state and determine the next action.
        
        Args:
            current_state: Current execution state
            working_memory: Available data from previous steps
            original_plan: The original plan steps
            last_result: Result from the last action
            
        Returns:
            Next action specification
        """
        self.log_method_call("reason_next_action", current_step=current_state.get("current_step"))
        
        try:
            # Build reasoning prompt
            prompt_inputs = {
                "current_state": json.dumps(current_state, indent=2),
                "working_memory": json.dumps(working_memory, indent=2),
                "original_plan": json.dumps(original_plan, indent=2),
                "last_result": json.dumps(last_result, indent=2) if last_result else "None"
            }
            
            # Get LLM reasoning
            messages = self.reasoning_prompt.format_messages(**prompt_inputs)
            response = await self._llm.apredict_messages(messages)
            
            # Parse response to extract action
            action = self._parse_reasoning_response(response.content)
            
            self.logger.debug("Next action determined", action_type=action.get("type"))
            return action
            
        except Exception as e:
            self.log_error("reason_next_action", e)
            # Return a safe default action
            return {
                "type": "wait",
                "reasoning": "Error in reasoning, waiting for intervention",
                "inputs": {},
                "outputs": {}
            }
    
    async def analyze_error_and_suggest_recovery(
        self,
        action: Dict[str, Any],
        error: str,
        current_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an error and suggest recovery strategies.
        
        Args:
            action: The action that failed
            error: Error message
            current_state: Current state/DOM if available
            
        Returns:
            Recovery strategy specification
        """
        self.log_method_call("analyze_error_and_suggest_recovery", error=error[:100])
        
        try:
            # Build error analysis prompt
            prompt_inputs = {
                "action": json.dumps(action, indent=2),
                "tool": action.get("tool_required", "Unknown"),
                "inputs": json.dumps(action.get("inputs", {}), indent=2),
                "error": error,
                "current_state": json.dumps(current_state, indent=2) if current_state else "Not available"
            }
            
            # Get LLM analysis
            messages = self.error_recovery_prompt.format_messages(**prompt_inputs)
            response = await self._llm.apredict_messages(messages)
            
            # Parse recovery strategies
            recovery = self._parse_error_recovery_response(response.content)
            
            self.logger.info("Error recovery strategy generated", strategy=recovery.get("strategy"))
            return recovery
            
        except Exception as e:
            self.log_error("analyze_error_and_suggest_recovery", e)
            return {
                "strategy": "escalate",
                "reasoning": "Unable to analyze error",
                "requires_hitl": True
            }
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON from LLM response")
        
        # Fallback: try to parse the entire response
        try:
            return json.loads(response)
        except:
            raise ValueError("Could not extract valid JSON from LLM response")
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse reasoning response to extract action specification."""
        # Look for structured sections in the response
        action = {
            "type": "unknown",
            "reasoning": "",
            "inputs": {},
            "outputs": {},
            "tool_required": None,
            "action_to_perform": None
        }
        
        # Extract reasoning
        reasoning_match = re.search(r'(?:Reasoning|Reason):\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if reasoning_match:
            action["reasoning"] = reasoning_match.group(1).strip()
        
        # Extract action type
        action_match = re.search(r'(?:Action|Next Action):\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()
            action["type"] = action_text.split()[0].lower()
        
        # Try to extract JSON if present
        try:
            json_data = self._extract_json_from_response(response)
            action.update(json_data)
        except:
            pass
        
        return action
    
    def _parse_error_recovery_response(self, response: str) -> Dict[str, Any]:
        """Parse error recovery response."""
        recovery = {
            "strategy": "retry",
            "reasoning": "",
            "modifications": {},
            "alternative_approach": None,
            "requires_hitl": False
        }
        
        # Extract strategy
        if "human intervention" in response.lower() or "hitl" in response.lower():
            recovery["strategy"] = "hitl"
            recovery["requires_hitl"] = True
        elif "alternative" in response.lower():
            recovery["strategy"] = "alternative"
        elif "retry" in response.lower():
            recovery["strategy"] = "retry"
        
        # Try to extract structured data
        try:
            json_data = self._extract_json_from_response(response)
            recovery.update(json_data)
        except:
            recovery["reasoning"] = response[:500]  # Use first part as reasoning
        
        return recovery
    
    def _apply_known_patterns(
        self,
        plan: IntegrationPlan,
        patterns: List[Dict[str, Any]]
    ) -> IntegrationPlan:
        """Apply known patterns to improve the plan."""
        # This would match patterns and update plan steps
        # For now, return the plan as-is
        return plan
    
    def _create_fallback_plan(
        self,
        vendor_name: str,
        documentation_url: str
    ) -> IntegrationPlan:
        """Create a basic fallback plan when LLM fails."""
        return IntegrationPlan(
            vendor_name=vendor_name,
            documentation_url=documentation_url,
            plan_version="1.0-fallback",
            total_steps=4,
            steps=[
                TaskStep(
                    step_id=1,
                    description="Create SAML application in Okta",
                    tool_required="OktaAPIClient",
                    action_to_perform="create_saml_app",
                    inputs={"label": f"{vendor_name} SAML"},
                    outputs={"app_id": "okta_app_id", "sso_url": "sso_url"},
                    dependencies=[],
                    context="okta_api"
                ),
                TaskStep(
                    step_id=2,
                    description="Navigate to vendor admin console",
                    tool_required="WebInteractor", 
                    action_to_perform="navigate",
                    inputs={"url": documentation_url},
                    outputs={},
                    dependencies=[1],
                    context="browser"
                ),
                TaskStep(
                    step_id=3,
                    description="Request human assistance for configuration",
                    tool_required="None",
                    action_to_perform="hitl_request",
                    inputs={"reason": "Fallback plan - need human guidance"},
                    outputs={},
                    dependencies=[2],
                    context="analysis"
                ),
                TaskStep(
                    step_id=4,
                    description="Verify integration",
                    tool_required="WebInteractor",
                    action_to_perform="verify_sso",
                    inputs={},
                    outputs={"verified": "integration_status"},
                    dependencies=[3],
                    context="browser"
                )
            ],
            estimated_duration_minutes=45
        )
    
    async def decompose_complex_step(
        self,
        step: TaskStep,
        additional_context: Dict[str, Any]
    ) -> List[TaskStep]:
        """
        Decompose a complex step into smaller sub-steps.
        
        Args:
            step: The step to decompose
            additional_context: Additional information for decomposition
            
        Returns:
            List of smaller steps
        """
        # This would use LLM to break down complex steps
        # For now, return the original step
        return [step]
    
    async def validate_plan_coherence(self, plan: IntegrationPlan) -> Tuple[bool, List[str]]:
        """
        Validate that a plan is coherent and executable.
        
        Args:
            plan: Plan to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check dependency ordering
        for step in plan.steps:
            for dep_id in step.dependencies:
                if dep_id >= step.step_id:
                    issues.append(f"Step {step.step_id} depends on future step {dep_id}")
        
        # Check required inputs are available
        available_outputs = {}
        for step in plan.steps:
            # Check if required inputs are available
            for input_key, input_value in step.inputs.items():
                if isinstance(input_value, str) and input_value.startswith("{{") and input_value.endswith("}}"):
                    # This is a reference to another step's output
                    ref = input_value[2:-2]
                    if ref not in available_outputs:
                        issues.append(f"Step {step.step_id} requires '{ref}' which is not available")
            
            # Add step outputs to available
            for output_key, output_ref in step.outputs.items():
                available_outputs[output_ref] = step.step_id
        
        # Check tool/action compatibility
        tool_actions = {
            "OktaAPIClient": ["create_saml_app", "get_app_metadata", "update_app", "assign_user_to_app"],
            "WebInteractor": ["navigate", "click", "type", "select_option", "get_element_text"]
        }
        
        for step in plan.steps:
            if step.tool_required in tool_actions:
                if step.action_to_perform not in tool_actions[step.tool_required]:
                    issues.append(f"Step {step.step_id}: Invalid action '{step.action_to_perform}' for tool '{step.tool_required}'")
        
        is_valid = len(issues) == 0
        return is_valid, issues