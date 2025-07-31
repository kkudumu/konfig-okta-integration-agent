"""
ReAct Orchestrator for Konfig.

This module implements the ReAct (Reason + Act) cognitive loop that coordinates
all components of the Konfig system to perform autonomous SSO integrations.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from konfig.config.settings import get_settings
from konfig.modules.memory.memory_module import MemoryModule
from konfig.tools.okta_api_client import OktaAPIClient
from konfig.tools.web_interactor import WebInteractor
from konfig.utils.logging import LoggingMixin


class ExecutionContext(Enum):
    """Execution context for the agent."""
    BROWSER = "browser"
    OKTA_API = "okta_api"
    ANALYSIS = "analysis"
    HITL = "hitl"


class AgentState(Dict[str, Any]):
    """
    Agent state for the ReAct loop.
    
    This extends dict to provide a structured state object that can be
    serialized and restored across interrupts and restarts.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure required state keys exist
        self.setdefault("job_id", None)
        self.setdefault("current_step", 0)
        self.setdefault("max_steps", 50)
        self.setdefault("context", ExecutionContext.ANALYSIS)
        self.setdefault("working_memory", {})
        self.setdefault("current_plan", [])
        self.setdefault("completed_steps", [])
        self.setdefault("error_count", 0)
        self.setdefault("last_error", None)
        self.setdefault("hitl_requested", False)
        self.setdefault("hitl_context", {})
        self.setdefault("integration_data", {})


class ReActOrchestrator(LoggingMixin):
    """
    ReAct-based orchestrator for autonomous SSO integration.
    
    This orchestrator implements the ReAct pattern (Reason + Act) using
    LangGraph to coordinate all system components and manage the agent's
    cognitive loop.
    """
    
    def __init__(self):
        """Initialize the ReAct Orchestrator."""
        super().__init__()
        self.setup_logging("orchestrator")
        
        self.settings = get_settings()
        
        # Initialize core modules and tools
        self.memory_module = MemoryModule()
        self.web_interactor = None  # Initialized when needed
        self.okta_client = None     # Initialized when needed
        
        # Graph and state management
        self.graph: Optional[CompiledStateGraph] = None
        self.checkpointer = MemorySaver()  # In-memory checkpointing for now
        
        # Build the execution graph
        self._build_graph()
        
        self.logger.info("ReAct Orchestrator initialized")
    
    def _build_graph(self) -> None:
        """Build the LangGraph execution graph for the ReAct loop."""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each phase of the ReAct loop
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("check_completion", self._check_completion_node)
        workflow.add_node("handle_error", self._handle_error_node)
        workflow.add_node("hitl_pause", self._hitl_pause_node)
        
        # Define the flow
        workflow.set_entry_point("think")
        
        # Think -> Act
        workflow.add_edge("think", "act")
        
        # Act -> Observe (with conditional error handling)
        workflow.add_conditional_edges(
            "act",
            self._should_handle_error,
            {
                "error": "handle_error",
                "success": "observe",
                "hitl": "hitl_pause"
            }
        )
        
        # Error handling -> Think (retry) or HITL
        workflow.add_conditional_edges(
            "handle_error",
            self._should_retry_or_hitl,
            {
                "retry": "think",
                "hitl": "hitl_pause",
                "end": END
            }
        )
        
        # Observe -> Check completion
        workflow.add_edge("observe", "check_completion")
        
        # Check completion -> Continue or End
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "think",
                "complete": END,
                "hitl": "hitl_pause"
            }
        )
        
        # HITL pause -> End (will be resumed externally)
        workflow.add_edge("hitl_pause", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        
        self.logger.info("Execution graph built successfully")
    
    async def execute_integration(
        self,
        job_id: uuid.UUID,
        documentation_url: str,
        vendor_name: Optional[str] = None,
        okta_domain: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete SSO integration using the ReAct loop.
        
        Args:
            job_id: Unique job identifier
            documentation_url: URL to vendor documentation
            vendor_name: Name of the vendor application
            okta_domain: Okta domain for integration
            initial_state: Optional initial state
            
        Returns:
            Integration result
        """
        self.logger.info(
            "Starting integration execution",
            job_id=str(job_id),
            vendor_name=vendor_name,
            documentation_url=documentation_url
        )
        
        # Initialize tools with proper configuration
        await self._initialize_tools(okta_domain)
        
        # Create initial agent state
        state = AgentState({
            "job_id": str(job_id),
            "documentation_url": documentation_url,
            "vendor_name": vendor_name,
            "okta_domain": okta_domain,
            "start_time": datetime.now().isoformat(),
            "context": ExecutionContext.ANALYSIS,
            **(initial_state or {})
        })
        
        # Create job record in memory
        await self.memory_module.create_job(
            job_id=job_id,
            input_url=documentation_url,
            vendor_name=vendor_name,
            okta_domain=okta_domain,
            initial_state=dict(state)
        )
        
        try:
            # Execute the graph
            thread_config = {"configurable": {"thread_id": str(job_id)}}
            
            final_state = None
            async for event in self.graph.astream(state, config=thread_config):
                self.logger.debug("Graph event", event=event)
                final_state = event
            
            # Extract the final state
            if final_state:
                # Get the last state from the event
                state_key = list(final_state.keys())[0]
                final_state = final_state[state_key]
            
            # Update job with final status
            if final_state and not final_state.get("hitl_requested"):
                await self.memory_module.update_job_status(
                    job_id, "completed_success"
                )
            elif final_state and final_state.get("hitl_requested"):
                await self.memory_module.update_job_status(
                    job_id, "paused_for_hitl"
                )
            
            result = {
                "success": True,
                "job_id": str(job_id),
                "final_state": final_state,
                "completed_steps": final_state.get("completed_steps", []) if final_state else [],
                "integration_data": final_state.get("integration_data", {}) if final_state else {}
            }
            
            self.logger.info("Integration execution completed", job_id=str(job_id))
            return result
            
        except Exception as e:
            self.log_error("execute_integration", e, job_id=str(job_id))
            
            # Update job with failure status
            await self.memory_module.update_job_status(
                job_id, "completed_failure", str(e)
            )
            
            return {
                "success": False,
                "job_id": str(job_id),
                "error": str(e)
            }
        finally:
            # Clean up tools
            await self._cleanup_tools()
    
    async def resume_integration(self, job_id: uuid.UUID) -> Dict[str, Any]:
        """
        Resume a paused integration from its last checkpoint.
        
        Args:
            job_id: Job identifier to resume
            
        Returns:
            Resume result
        """
        self.logger.info("Resuming integration", job_id=str(job_id))
        
        try:
            # Load job state
            job = await self.memory_module.get_job(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status != "paused_for_hitl":
                raise ValueError(f"Job {job_id} is not in a resumable state: {job.status}")
            
            # Initialize tools
            await self._initialize_tools(job.okta_domain)
            
            # Resume from checkpoint
            thread_config = {"configurable": {"thread_id": str(job_id)}}
            
            # Clear HITL flag and resume
            checkpoint = await self.graph.aget_state(thread_config)
            if checkpoint and checkpoint.values:
                resumed_state = checkpoint.values.copy()
                resumed_state["hitl_requested"] = False
                resumed_state["hitl_context"] = {}
                
                # Continue execution
                final_state = None
                async for event in self.graph.astream(None, config=thread_config):
                    final_state = event
                
                # Update job status
                await self.memory_module.update_job_status(job_id, "completed_success")
                
                result = {
                    "success": True,
                    "job_id": str(job_id),
                    "resumed": True,
                    "final_state": final_state
                }
                
                self.logger.info("Integration resumed successfully", job_id=str(job_id))
                return result
            else:
                raise ValueError("No checkpoint found for job")
                
        except Exception as e:
            self.log_error("resume_integration", e, job_id=str(job_id))
            return {
                "success": False,
                "job_id": str(job_id),
                "error": str(e)
            }
        finally:
            await self._cleanup_tools()
    
    # ========== GRAPH NODE IMPLEMENTATIONS ==========
    
    async def _think_node(self, state: AgentState) -> AgentState:
        """
        Think phase: Analyze current state and plan next action.
        
        This node implements the reasoning phase of the ReAct loop,
        analyzing the current situation and determining the next action.
        """
        job_id = state["job_id"]
        current_step = state["current_step"]
        
        await self.memory_module.store_trace(
            uuid.UUID(job_id),
            "thought",
            {"step": current_step, "context": state["context"].value},
            "success"
        )
        
        self.logger.debug(
            "Think phase",
            job_id=job_id,
            step=current_step,
            context=state["context"].value
        )
        
        # For now, implement a simple hard-coded plan
        # This will be replaced with LLM-based planning in the Cognition Module
        if current_step == 0:
            # First step: Create Okta application
            state["next_action"] = {
                "type": "create_okta_app",
                "context": ExecutionContext.OKTA_API,
                "params": {
                    "label": state.get("vendor_name", "Unknown Vendor") + " SAML App"
                }
            }
        elif current_step == 1:
            # Second step: Navigate to vendor documentation
            state["next_action"] = {
                "type": "navigate",
                "context": ExecutionContext.BROWSER,
                "params": {
                    "url": state["documentation_url"]
                }
            }
        elif current_step >= 2:
            # Subsequent steps would involve parsing documentation and configuring
            state["next_action"] = {
                "type": "complete",
                "context": ExecutionContext.ANALYSIS,
                "params": {}
            }
        
        return state
    
    async def _act_node(self, state: AgentState) -> AgentState:
        """
        Act phase: Execute the planned action.
        
        This node executes the action determined in the think phase,
        using the appropriate tool based on the execution context.
        """
        job_id = state["job_id"]
        action = state.get("next_action", {})
        action_type = action.get("type")
        context = action.get("context", ExecutionContext.ANALYSIS)
        params = action.get("params", {})
        
        await self.memory_module.store_trace(
            uuid.UUID(job_id),
            "tool_call",
            {"action_type": action_type, "context": context.value, "params": params},
            "success"
        )
        
        self.logger.debug(
            "Act phase",
            job_id=job_id,
            action_type=action_type,
            context=context.value
        )
        
        try:
            # Switch execution context
            state["context"] = context
            
            # Execute action based on type and context
            if context == ExecutionContext.OKTA_API:
                result = await self._execute_okta_action(action_type, params)
            elif context == ExecutionContext.BROWSER:
                result = await self._execute_browser_action(action_type, params)
            else:
                result = await self._execute_analysis_action(action_type, params)
            
            state["last_action_result"] = result
            state["last_error"] = None
            
        except Exception as e:
            self.logger.error(f"Action failed: {e}", exc_info=True)
            state["last_action_result"] = None
            state["last_error"] = str(e)
            state["error_count"] = state.get("error_count", 0) + 1
        
        return state
    
    async def _observe_node(self, state: AgentState) -> AgentState:
        """
        Observe phase: Process the action result and update working memory.
        
        This node analyzes the result of the action and extracts
        relevant information to store in working memory.
        """
        job_id = state["job_id"]
        result = state.get("last_action_result")
        
        await self.memory_module.store_trace(
            uuid.UUID(job_id),
            "observation",
            {"has_result": result is not None, "result_type": type(result).__name__},
            "success"
        )
        
        self.logger.debug("Observe phase", job_id=job_id, has_result=result is not None)
        
        # Extract and store relevant data from the action result
        if result:
            if isinstance(result, dict):
                # Store useful data in working memory
                if "app_id" in result:
                    state["working_memory"]["okta_app_id"] = result["app_id"]
                if "sso_url" in result:
                    state["working_memory"]["sso_url"] = result["sso_url"]
                if "entity_id" in result:
                    state["working_memory"]["entity_id"] = result["entity_id"]
                
                # Store the full result in integration data
                state["integration_data"][f"step_{state['current_step']}_result"] = result
        
        return state
    
    async def _check_completion_node(self, state: AgentState) -> AgentState:
        """
        Check if the integration is complete or should continue.
        
        This node determines whether the integration process should
        continue, complete, or require human intervention.
        """
        job_id = state["job_id"]
        current_step = state["current_step"]
        max_steps = state["max_steps"]
        
        self.logger.debug(
            "Check completion",
            job_id=job_id,
            current_step=current_step,
            max_steps=max_steps
        )
        
        # Mark current step as completed
        state["completed_steps"].append({
            "step": current_step,
            "action": state.get("next_action", {}),
            "result": state.get("last_action_result"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Increment step counter
        state["current_step"] = current_step + 1
        
        # Check completion conditions
        if current_step >= max_steps:
            state["completion_reason"] = "max_steps_reached"
            state["should_continue"] = False
        elif state.get("next_action", {}).get("type") == "complete":
            state["completion_reason"] = "integration_complete"
            state["should_continue"] = False
        else:
            state["should_continue"] = True
        
        return state
    
    async def _handle_error_node(self, state: AgentState) -> AgentState:
        """
        Handle errors and determine recovery strategy.
        
        This node implements error handling and recovery strategies,
        including self-healing attempts and HITL escalation.
        """
        job_id = state["job_id"]
        error = state.get("last_error")
        error_count = state.get("error_count", 0)
        
        await self.memory_module.store_trace(
            uuid.UUID(job_id),
            "observation",
            {"error": error, "error_count": error_count},
            "failure",
            error_details=error
        )
        
        self.logger.warning(
            "Handling error",
            job_id=job_id,
            error=error,
            error_count=error_count
        )
        
        # Implement error recovery logic
        if error_count < 3:
            # Attempt self-healing
            if self.settings.features.self_healing:
                # TODO: Implement self-healing with Learning Module
                state["recovery_strategy"] = "retry"
            else:
                state["recovery_strategy"] = "retry"
        else:
            # Escalate to HITL after multiple failures
            if self.settings.agent.hitl_enabled:
                state["recovery_strategy"] = "hitl"
                state["hitl_requested"] = True
                state["hitl_context"] = {
                    "reason": "multiple_errors",
                    "error": error,
                    "error_count": error_count,
                    "current_step": state["current_step"]
                }
            else:
                state["recovery_strategy"] = "end"
        
        return state
    
    async def _hitl_pause_node(self, state: AgentState) -> AgentState:
        """
        Pause execution for human-in-the-loop intervention.
        
        This node handles the HITL workflow by saving state and
        preparing for human intervention.
        """
        job_id = state["job_id"]
        hitl_context = state.get("hitl_context", {})
        
        await self.memory_module.store_trace(
            uuid.UUID(job_id),
            "hitl_request",
            hitl_context,
            "success"
        )
        
        self.logger.info(
            "HITL pause requested",
            job_id=job_id,
            reason=hitl_context.get("reason")
        )
        
        # Store current state for resumption
        await self.memory_module.store_job_state(uuid.UUID(job_id), dict(state))
        
        # TODO: Send notification to human operator
        # This would integrate with Slack, email, or web interface
        
        return state
    
    # ========== CONDITIONAL EDGE FUNCTIONS ==========
    
    def _should_handle_error(self, state: AgentState) -> str:
        """Determine if error handling is needed."""
        if state.get("last_error"):
            return "error"
        elif state.get("hitl_requested"):
            return "hitl"
        else:
            return "success"
    
    def _should_retry_or_hitl(self, state: AgentState) -> str:
        """Determine recovery strategy after error."""
        strategy = state.get("recovery_strategy", "end")
        if strategy == "retry":
            return "retry"
        elif strategy == "hitl":
            return "hitl"
        else:
            return "end"
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if execution should continue."""
        if state.get("hitl_requested"):
            return "hitl"
        elif state.get("should_continue", True):
            return "continue"
        else:
            return "complete"
    
    # ========== ACTION EXECUTION METHODS ==========
    
    async def _execute_okta_action(self, action_type: str, params: Dict[str, Any]) -> Any:
        """Execute an Okta API action."""
        if action_type == "create_okta_app":
            label = params.get("label", "SAML Application")
            return await self.okta_client.create_saml_app(label)
        elif action_type == "get_app_metadata":
            app_id = params.get("app_id")
            return await self.okta_client.get_app_metadata(app_id)
        else:
            raise ValueError(f"Unknown Okta action: {action_type}")
    
    async def _execute_browser_action(self, action_type: str, params: Dict[str, Any]) -> Any:
        """Execute a browser action."""
        if action_type == "navigate":
            url = params.get("url")
            return await self.web_interactor.navigate(url)
        elif action_type == "click":
            selector = params.get("selector")
            return await self.web_interactor.click(selector)
        elif action_type == "type":
            selector = params.get("selector")
            text = params.get("text")
            secret = params.get("secret", False)
            return await self.web_interactor.type(selector, text, secret)
        else:
            raise ValueError(f"Unknown browser action: {action_type}")
    
    async def _execute_analysis_action(self, action_type: str, params: Dict[str, Any]) -> Any:
        """Execute an analysis action."""
        if action_type == "complete":
            return {"status": "completed", "message": "Integration completed successfully"}
        else:
            raise ValueError(f"Unknown analysis action: {action_type}")
    
    # ========== TOOL MANAGEMENT ==========
    
    async def _initialize_tools(self, okta_domain: Optional[str] = None) -> None:
        """Initialize the tools needed for execution."""
        # Initialize Okta client
        self.okta_client = OktaAPIClient(okta_domain=okta_domain)
        await self.okta_client.__aenter__()
        
        # Initialize web interactor
        self.web_interactor = WebInteractor()
        await self.web_interactor.__aenter__()
        
        self.logger.debug("Tools initialized")
    
    async def _cleanup_tools(self) -> None:
        """Clean up tools and release resources."""
        if self.okta_client:
            await self.okta_client.__aexit__(None, None, None)
            self.okta_client = None
        
        if self.web_interactor:
            await self.web_interactor.__aexit__(None, None, None)
            self.web_interactor = None
        
        self.logger.debug("Tools cleaned up")