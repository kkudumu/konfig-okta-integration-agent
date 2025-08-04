"""
Error Context Collector

Automatically gathers comprehensive context when errors or timeouts occur,
enabling the LLM to analyze and suggest recovery strategies with full situational awareness.
"""

import asyncio
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from konfig.utils.logging import LoggingMixin


@dataclass
class StepContext:
    """Represents the context of a single execution step."""
    step_name: str
    tool: str
    action: str
    params: Dict[str, Any]
    timestamp: datetime
    duration_ms: Optional[float] = None
    status: str = "pending"  # pending, success, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    llm_reasoning: Optional[str] = None
    attempted_selectors: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ErrorContext:
    """Comprehensive error context for LLM analysis."""
    error_type: str
    error_message: str
    failed_step: StepContext
    recent_steps: List[StepContext]
    visual_context: Dict[str, Any]
    execution_state: Dict[str, Any]
    browser_state: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['failed_step'] = self.failed_step.to_dict()
        data['recent_steps'] = [step.to_dict() for step in self.recent_steps]
        return data


class StepContextTracker:
    """Tracks execution context across steps."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.step_history: List[StepContext] = []
        self.working_memory: Dict[str, Any] = {}
        self.current_step: Optional[StepContext] = None
    
    def start_step(
        self, 
        step_name: str, 
        tool: str, 
        action: str, 
        params: Dict[str, Any]
    ) -> StepContext:
        """Start tracking a new step."""
        step = StepContext(
            step_name=step_name,
            tool=tool,
            action=action,
            params=params.copy(),
            timestamp=datetime.now(),
            attempted_selectors=[]
        )
        self.current_step = step
        return step
    
    def add_selector_attempt(self, selector: str):
        """Track selector attempts for current step."""
        if self.current_step and self.current_step.attempted_selectors is not None:
            self.current_step.attempted_selectors.append(selector)
    
    def add_llm_reasoning(self, reasoning: str):
        """Add LLM reasoning to current step."""
        if self.current_step:
            self.current_step.llm_reasoning = reasoning
    
    def complete_step(
        self, 
        status: str, 
        result: Optional[Dict[str, Any]] = None, 
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ):
        """Complete the current step and add to history."""
        if self.current_step:
            self.current_step.status = status
            self.current_step.result = result
            self.current_step.error = error
            self.current_step.duration_ms = duration_ms
            
            # Add to history
            self.step_history.append(self.current_step)
            
            # Trim history
            if len(self.step_history) > self.max_history:
                self.step_history = self.step_history[-self.max_history:]
            
            # Update working memory with successful results
            if status == "success" and result:
                self.working_memory.update(result.get("extracted_data", {}))
            
            self.current_step = None
    
    def get_recent_steps(self, count: int = 5) -> List[StepContext]:
        """Get recent step history."""
        return self.step_history[-count:] if self.step_history else []
    
    def get_current_step(self) -> Optional[StepContext]:
        """Get currently executing step."""
        return self.current_step


class ErrorContextCollector(LoggingMixin):
    """Collects comprehensive context when errors occur."""
    
    def __init__(self):
        super().__init__()
        self.setup_logging("error_context_collector")
    
    async def collect_error_context(
        self,
        error: Exception,
        web_interactor,
        step_tracker: StepContextTracker,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Collect comprehensive error context for LLM analysis.
        
        Args:
            error: The exception that occurred
            web_interactor: WebInteractor instance for browser state
            step_tracker: Step context tracker
            additional_context: Any additional context to include
            
        Returns:
            ErrorContext with comprehensive information
        """
        self.logger.info("Collecting comprehensive error context")
        
        try:
            # Get basic error information
            error_type = type(error).__name__
            error_message = str(error)
            
            # Get failed step context
            failed_step = step_tracker.get_current_step()
            if not failed_step:
                # Create a synthetic step if none exists
                failed_step = StepContext(
                    step_name="Unknown Step",
                    tool="Unknown",
                    action="Unknown",
                    params={},
                    timestamp=datetime.now(),
                    status="failed",
                    error=error_message
                )
            
            # Collect visual context
            visual_context = await self._collect_visual_context(web_interactor)
            
            # Collect browser state
            browser_state = await self._collect_browser_state(web_interactor)
            
            # Get execution state
            execution_state = {
                "working_memory": step_tracker.working_memory.copy(),
                "total_steps_executed": len(step_tracker.step_history),
                "error_traceback": traceback.format_exc(),
                "additional_context": additional_context or {}
            }
            
            # Create error context
            context = ErrorContext(
                error_type=error_type,
                error_message=error_message,
                failed_step=failed_step,
                recent_steps=step_tracker.get_recent_steps(5),
                visual_context=visual_context,
                execution_state=execution_state,
                browser_state=browser_state,
                timestamp=datetime.now()
            )
            
            self.logger.info(
                f"Error context collected",
                error_type=error_type,
                recent_steps_count=len(context.recent_steps),
                has_screenshot=bool(visual_context.get("screenshot")),
                dom_length=len(visual_context.get("full_dom", ""))
            )
            
            return context
            
        except Exception as collection_error:
            self.logger.error(f"Failed to collect error context: {collection_error}")
            
            # Return minimal context if collection fails
            return ErrorContext(
                error_type=type(error).__name__,
                error_message=str(error),
                failed_step=step_tracker.get_current_step() or StepContext(
                    step_name="Context Collection Failed",
                    tool="Unknown",
                    action="Unknown",
                    params={},
                    timestamp=datetime.now()
                ),
                recent_steps=[],
                visual_context={"error": "Failed to collect visual context"},
                execution_state={"error": "Failed to collect execution state"},
                browser_state={"error": "Failed to collect browser state"},
                timestamp=datetime.now()
            )
    
    async def _collect_visual_context(self, web_interactor) -> Dict[str, Any]:
        """Collect visual context from the browser."""
        context = {}
        
        try:
            # Take screenshot
            screenshot_path = await web_interactor.screenshot()
            context["screenshot_path"] = screenshot_path
            
            # Get full DOM (not simplified)
            full_dom = await web_interactor.get_current_dom(simplify=False)
            context["full_dom"] = full_dom
            
            # Get simplified DOM for comparison
            simple_dom = await web_interactor.get_current_dom(simplify=True)
            context["simple_dom"] = simple_dom
            
            # Get current URL and title
            context["current_url"] = await web_interactor.get_current_url()
            context["page_title"] = await web_interactor.get_page_title()
            
        except Exception as e:
            context["collection_error"] = str(e)
            self.logger.warning(f"Failed to collect some visual context: {e}")
        
        return context
    
    async def _collect_browser_state(self, web_interactor) -> Dict[str, Any]:
        """Collect browser state information."""
        state = {}
        
        try:
            # Get console logs
            console_logs = await web_interactor.get_console_logs()
            state["console_logs"] = console_logs[-20:]  # Last 20 logs
            
            # Get network requests if available
            # Note: This would need to be implemented in WebInteractor
            
            # Get viewport information
            state["viewport"] = await web_interactor.get_viewport_size()
            
            # Check if browser is still responsive
            state["browser_responsive"] = await self._check_browser_responsive(web_interactor)
            
        except Exception as e:
            state["collection_error"] = str(e)
            self.logger.warning(f"Failed to collect some browser state: {e}")
        
        return state
    
    async def _check_browser_responsive(self, web_interactor) -> bool:
        """Check if browser is still responsive."""
        try:
            # Try a simple operation with short timeout
            await asyncio.wait_for(web_interactor.get_current_url(), timeout=2.0)
            return True
        except:
            return False


class ErrorRecoveryAnalyzer(LoggingMixin):
    """Analyzes error context and suggests recovery strategies."""
    
    def __init__(self):
        super().__init__()
        self.setup_logging("error_recovery_analyzer")
    
    async def analyze_and_suggest_recovery(
        self, 
        error_context: ErrorContext,
        llm_service
    ) -> Dict[str, Any]:
        """
        Analyze error context and suggest recovery strategies.
        
        Args:
            error_context: Comprehensive error context
            llm_service: LLM service for analysis
            
        Returns:
            Recovery plan with analysis and suggested actions
        """
        self.logger.info(f"Analyzing error: {error_context.error_type}")
        
        # Create comprehensive analysis prompt
        analysis_prompt = self._create_analysis_prompt(error_context)
        
        try:
            # Get LLM analysis
            response = await llm_service.generate_response(
                prompt=analysis_prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            recovery_plan = json.loads(response)
            
            self.logger.info(
                f"Recovery analysis completed",
                confidence=recovery_plan.get("confidence", "unknown"),
                recovery_actions=len(recovery_plan.get("recovery_actions", []))
            )
            
            return recovery_plan
            
        except Exception as e:
            self.logger.error(f"Failed to analyze error context: {e}")
            return {
                "analysis": "Failed to analyze error context",
                "confidence": "low",
                "recovery_actions": [],
                "error": str(e)
            }
    
    def _create_analysis_prompt(self, error_context: ErrorContext) -> str:
        """Create comprehensive analysis prompt for LLM."""
        
        # Format recent steps for context
        steps_summary = []
        for step in error_context.recent_steps:
            step_info = f"- {step.step_name} ({step.tool}.{step.action}) -> {step.status}"
            if step.error:
                step_info += f" | Error: {step.error}"
            if step.llm_reasoning:
                step_info += f" | Reasoning: {step.llm_reasoning[:100]}..."
            steps_summary.append(step_info)
        
        failed_step = error_context.failed_step
        
        prompt = f"""
ERROR ANALYSIS AND RECOVERY

ERROR DETAILS:
- Type: {error_context.error_type}
- Message: {error_context.error_message}
- Timestamp: {error_context.timestamp}

FAILED STEP:
- Name: {failed_step.step_name}
- Tool: {failed_step.tool}
- Action: {failed_step.action}
- Parameters: {json.dumps(failed_step.params, indent=2)}
- Attempted Selectors: {failed_step.attempted_selectors or 'None'}
- Previous LLM Reasoning: {failed_step.llm_reasoning or 'None'}

RECENT EXECUTION HISTORY:
{chr(10).join(steps_summary) if steps_summary else 'No recent steps'}

BROWSER STATE:
- Current URL: {error_context.visual_context.get('current_url', 'Unknown')}
- Page Title: {error_context.visual_context.get('page_title', 'Unknown')}
- Browser Responsive: {error_context.browser_state.get('browser_responsive', 'Unknown')}
- Console Errors: {len([log for log in error_context.browser_state.get('console_logs', []) if log.get('level') == 'error'])} errors

CURRENT PAGE DOM:
{error_context.visual_context.get('simple_dom', 'DOM not available')[:2000]}...

WORKING MEMORY:
{json.dumps(error_context.execution_state.get('working_memory', {}), indent=2)}

ANALYSIS TASK:
You are a debugging expert. Analyze this error comprehensively and provide a recovery plan.

1. ERROR ROOT CAUSE: What specifically went wrong and why?
2. PAGE STATE ANALYSIS: What does the current page state tell us?
3. SELECTOR ANALYSIS: Were the selectors appropriate? What should be used instead?
4. RECOVERY STRATEGY: How should we recover and continue?

RESPONSE FORMAT (JSON):
{{
    "root_cause_analysis": "Detailed explanation of what went wrong and why",
    "page_state_assessment": "Analysis of current page state and what it reveals",
    "selector_issues": "Problems with selectors and suggested improvements",
    "confidence": "high|medium|low - your confidence in this analysis",
    "recovery_actions": [
        {{
            "action_type": "retry|navigate|wait|alternative_approach",
            "description": "What to do",
            "specific_changes": "Exact changes to make (selectors, timing, etc.)",
            "reasoning": "Why this should work"
        }}
    ],
    "prevention_suggestions": "How to prevent similar errors in the future"
}}

Focus on actionable, specific recovery steps with exact selectors and clear reasoning.
"""
        
        return prompt