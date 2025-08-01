"""
Adaptive Intelligence Service

Provides tiered intelligence capabilities that escalate only when needed,
maintaining speed while adding smart recovery and reasoning capabilities.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from konfig.utils.logging import LoggingMixin


@dataclass
class IntelligenceLevel:
    """Represents a level of intelligence with associated costs and capabilities."""
    level: int
    name: str
    max_time: float  # seconds
    description: str


@dataclass
class ActionAttempt:
    """Records an attempt to execute an action."""
    timestamp: float
    level: int
    action: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class UIPattern:
    """Represents a learned UI pattern."""
    url_pattern: str
    page_indicators: List[str]
    required_actions: List[Dict[str, Any]]
    success_indicators: List[str]
    confidence: float
    learned_from: str
    last_used: float


class PatternCache:
    """Caches learned UI patterns for fast execution."""
    
    def __init__(self):
        self.cache_file = Path("konfig_ui_patterns.json")
        self.patterns: Dict[str, UIPattern] = {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load patterns from cache file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for key, pattern_data in data.items():
                        self.patterns[key] = UIPattern(**pattern_data)
            except Exception:
                pass  # Start with empty cache if load fails
        else:
            # Initialize with known patterns
            self._initialize_known_patterns()
    
    def _initialize_known_patterns(self):
        """Initialize with known patterns from previous learning."""
        # Google Admin Console SSO pattern
        self.patterns["google_admin_sso"] = UIPattern(
            url_pattern="admin.google.com/ac/security/sso",
            page_indicators=["Third-party SSO profiles", "Legacy SSO Profile"],
            required_actions=[
                {
                    "action": "click",
                    "target": "Legacy SSO Profile",
                    "selector": "tr:has-text('Legacy SSO Profile')",
                    "description": "Click on Legacy SSO Profile row to enter configuration"
                }
            ],
            success_indicators=["Sign-in page URL", "Sign-out page URL", "Verification certificate"],
            confidence=0.9,
            learned_from="user_guidance",
            last_used=time.time()
        )
        
        self._save_patterns()
    
    def _save_patterns(self):
        """Save patterns to cache file."""
        try:
            data = {}
            for key, pattern in self.patterns.items():
                data[key] = {
                    "url_pattern": pattern.url_pattern,
                    "page_indicators": pattern.page_indicators,
                    "required_actions": pattern.required_actions,
                    "success_indicators": pattern.success_indicators,
                    "confidence": pattern.confidence,
                    "learned_from": pattern.learned_from,
                    "last_used": pattern.last_used
                }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Continue if save fails
    
    def find_pattern(self, url: str, page_content: str) -> Optional[UIPattern]:
        """Find a matching pattern for the current page."""
        for pattern_key, pattern in self.patterns.items():
            if pattern.url_pattern in url:
                # Check if page indicators match
                matches = sum(1 for indicator in pattern.page_indicators 
                             if indicator.lower() in page_content.lower())
                
                if matches >= len(pattern.page_indicators) * 0.7:  # 70% match threshold
                    pattern.last_used = time.time()
                    self._save_patterns()
                    return pattern
        
        return None
    
    def add_pattern(self, pattern_key: str, pattern: UIPattern):
        """Add a new learned pattern."""
        self.patterns[pattern_key] = pattern
        self._save_patterns()


class AdaptiveIntelligence(LoggingMixin):
    """
    Adaptive intelligence system that provides tiered capabilities:
    
    Level 1: Fast path - Current simple approach (0.5-2s)
    Level 2: Pattern-based recovery - Use cached patterns (1-3s)  
    Level 3: Smart reasoning - LLM-based error recovery (3-8s)
    Level 4: Full analysis - Screenshot + comprehensive reasoning (8-20s)
    Level 5: User escalation - Ask for guidance when stuck (immediate)
    """
    
    INTELLIGENCE_LEVELS = [
        IntelligenceLevel(1, "fast_path", 2.0, "Simple direct execution"),
        IntelligenceLevel(2, "pattern_based", 5.0, "Use cached UI patterns"),
        IntelligenceLevel(3, "smart_reasoning", 10.0, "LLM-based error recovery"),
        IntelligenceLevel(4, "full_analysis", 25.0, "Screenshot + comprehensive analysis"),
        IntelligenceLevel(5, "user_escalation", 0.1, "Request user guidance")
    ]
    
    def __init__(self, web_interactor, llm_service=None):
        super().__init__()
        self.setup_logging("adaptive_intelligence")
        
        self.web_interactor = web_interactor
        self.llm_service = llm_service
        self.pattern_cache = PatternCache()
        
        # Configuration
        self.max_attempts = 4
        self.total_time_budget = 60.0  # seconds
        self.failure_threshold = 2
        
        # State tracking
        self.current_attempts: List[ActionAttempt] = []
        self.start_time = None
    
    async def execute_with_adaptive_intelligence(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an action using adaptive intelligence levels.
        
        Args:
            action: The action to execute
            context: Additional context (working_memory, etc.)
            
        Returns:
            Result of the action execution
        """
        self.start_time = time.time()
        self.current_attempts = []
        
        self.logger.info(f"Starting adaptive execution of: {action.get('name', 'Unknown action')}")
        
        # Try each intelligence level in sequence
        for level_info in self.INTELLIGENCE_LEVELS:
            if self._should_try_level(level_info):
                try:
                    result = await self._execute_at_level(action, level_info, context)
                    
                    if result.get("success"):
                        self._record_success(level_info, action)
                        return result
                    else:
                        self._record_attempt(action, level_info, False, result.get("error"))
                        
                except Exception as e:
                    self._record_attempt(action, level_info, False, str(e))
                    self.logger.warning(f"Level {level_info.level} failed: {e}")
        
        # All levels failed
        return {
            "success": False,
            "error": "All intelligence levels exhausted",
            "attempts": len(self.current_attempts),
            "total_duration": time.time() - self.start_time
        }
    
    def _should_try_level(self, level_info: IntelligenceLevel) -> bool:
        """Determine if we should try this intelligence level."""
        # Check time budget
        elapsed = time.time() - self.start_time
        if elapsed + level_info.max_time > self.total_time_budget:
            return False
        
        # Check attempt count
        if len(self.current_attempts) >= self.max_attempts:
            return False
        
        # Always try user escalation if we've tried other levels
        if level_info.level == 5 and len(self.current_attempts) >= 2:
            return True
        
        return True
    
    async def _execute_at_level(
        self,
        action: Dict[str, Any],
        level_info: IntelligenceLevel,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute action at specific intelligence level."""
        
        self.logger.info(f"Trying level {level_info.level}: {level_info.name}")
        level_start = time.time()
        
        try:
            if level_info.level == 1:
                result = await self._level_1_fast_path(action, context)
            elif level_info.level == 2:
                result = await self._level_2_pattern_based(action, context)
            elif level_info.level == 3:
                result = await self._level_3_smart_reasoning(action, context)
            elif level_info.level == 4:
                result = await self._level_4_full_analysis(action, context)
            elif level_info.level == 5:
                result = await self._level_5_user_escalation(action, context)
            else:
                raise ValueError(f"Unknown intelligence level: {level_info.level}")
            
            duration = time.time() - level_start
            self.logger.info(f"Level {level_info.level} completed in {duration:.2f}s")
            
            return result
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Level {level_info.level} timed out after {level_info.max_time}s"
            }
    
    async def _level_1_fast_path(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 1: Fast path using current simple approach."""
        
        tool = action.get("tool")
        action_type = action.get("action")
        params = action.get("params", {})
        
        if tool == "WebInteractor":
            if action_type == "click":
                selector = params.get("selector")
                text_content = params.get("text_content")
                
                result = await self.web_interactor.click(selector, text_content)
                return {"success": result.get("success", False), "result": result}
                
            elif action_type == "type":
                result = await self.web_interactor.type(
                    params.get("selector"),
                    params.get("text"),
                    secret=params.get("secret", False)
                )
                return {"success": result.get("success", False), "result": result}
                
            elif action_type == "navigate":
                result = await self.web_interactor.navigate(params.get("url"))
                return {"success": result.get("success", False), "result": result}
        
        return {"success": False, "error": "Unsupported action for fast path"}
    
    async def _level_2_pattern_based(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 2: Use cached UI patterns for known scenarios."""
        
        # Get current page info
        current_url = await self.web_interactor.get_current_url()
        page_content = await self.web_interactor.get_current_dom(simplify=True)
        
        # Find matching pattern
        pattern = self.pattern_cache.find_pattern(current_url, page_content)
        
        if pattern:
            self.logger.info(f"Found matching pattern: {pattern.url_pattern}")
            
            # Execute pattern actions
            for pattern_action in pattern.required_actions:
                try:
                    if pattern_action["action"] == "click":
                        # Try multiple selector strategies
                        success = False
                        
                        # Try specific selector first
                        if "selector" in pattern_action:
                            try:
                                await self.web_interactor._page.click(
                                    pattern_action["selector"], 
                                    timeout=3000
                                )
                                success = True
                            except Exception:
                                pass
                        
                        # Try text-based clicking
                        if not success and "target" in pattern_action:
                            success = await self._smart_click_by_text(pattern_action["target"])
                        
                        if success:
                            # Wait for page change
                            await self.web_interactor._page.wait_for_timeout(2000)
                            
                            # Check if we reached expected state
                            new_content = await self.web_interactor.get_current_dom(simplify=True)
                            success_found = any(
                                indicator.lower() in new_content.lower()
                                for indicator in pattern.success_indicators
                            )
                            
                            if success_found:
                                return {
                                    "success": True,
                                    "message": f"Successfully executed pattern: {pattern.url_pattern}",
                                    "pattern_used": pattern.url_pattern
                                }
                
                except Exception as e:
                    self.logger.warning(f"Pattern action failed: {e}")
                    continue
        
        return {"success": False, "error": "No matching pattern found or pattern execution failed"}
    
    async def _level_3_smart_reasoning(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 3: LLM-based error recovery and reasoning."""
        
        if not self.llm_service:
            return {"success": False, "error": "No LLM service available for smart reasoning"}
        
        # Gather context
        current_url = await self.web_interactor.get_current_url()
        page_content = await self.web_interactor.get_current_dom(simplify=True)
        recent_errors = [attempt.error for attempt in self.current_attempts if attempt.error]
        
        reasoning_prompt = f"""
I'm trying to execute this action: {json.dumps(action, indent=2)}

CURRENT PAGE:
URL: {current_url}
Content: {page_content[:2000]}...

PREVIOUS ATTEMPTS FAILED WITH:
{recent_errors}

Based on the current page and failed attempts, what should I try instead?
Look for:
1. Alternative selectors that might work
2. Navigation steps I might be missing  
3. UI patterns I should recognize
4. Elements I should click first before the main action

Respond with JSON:
{{
    "analysis": "What I can see and understand about the current situation",
    "hypothesis": "Why the previous attempts failed",
    "alternative_actions": [
        {{
            "action": "click|type|navigate",
            "selector": "specific selector to try",
            "text_content": "text to find (for clicks)",
            "description": "why this should work",
            "confidence": 0.0-1.0
        }}
    ],
    "confidence": 0.0-1.0
}}
"""
        
        try:
            response = await self.llm_service.generate_response(
                reasoning_prompt,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse response
            from konfig.services.intelligent_planner import IntelligentPlanner
            planner = IntelligentPlanner()
            reasoning = planner._extract_json_from_response(response)
            
            # Try alternative actions
            for alt_action in reasoning.get("alternative_actions", []):
                try:
                    if alt_action["action"] == "click":
                        if alt_action.get("text_content"):
                            success = await self._smart_click_by_text(alt_action["text_content"])
                        else:
                            result = await self.web_interactor.click(alt_action.get("selector"))
                            success = result.get("success", False)
                        
                        if success:
                            return {
                                "success": True,
                                "message": f"Smart reasoning succeeded: {alt_action['description']}",
                                "reasoning": reasoning.get("analysis")
                            }
                
                except Exception as e:
                    continue
            
            return {
                "success": False,
                "error": "Smart reasoning generated alternatives but none worked",
                "reasoning": reasoning.get("analysis")
            }
            
        except Exception as e:
            return {"success": False, "error": f"Smart reasoning failed: {e}"}
    
    async def _level_4_full_analysis(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 4: Full analysis with screenshot and comprehensive reasoning."""
        
        if not self.llm_service:
            return {"success": False, "error": "No LLM service available for full analysis"}
        
        try:
            # Take screenshot and gather full context in parallel
            screenshot_task = asyncio.create_task(
                asyncio.wait_for(self.web_interactor.take_screenshot(), timeout=10.0)
            )
            url_task = asyncio.create_task(self.web_interactor.get_current_url())
            dom_task = asyncio.create_task(self.web_interactor.get_current_dom(simplify=True))
            title_task = asyncio.create_task(self.web_interactor.get_page_title())
            
            # Wait for all context with error handling
            try:
                screenshot_path, current_url, page_content, page_title = await asyncio.gather(
                    screenshot_task, url_task, dom_task, title_task
                )
            except asyncio.TimeoutError:
                # If screenshot times out, continue without it
                screenshot_path = None
                current_url = await url_task
                page_content = await dom_task
                page_title = await title_task
            
            # Note: In a real implementation, you'd send the screenshot to a vision-capable LLM
            # For now, we'll use the text-based comprehensive analysis
            
            full_analysis_prompt = f"""
COMPREHENSIVE PAGE ANALYSIS

ACTION ATTEMPTING: {json.dumps(action, indent=2)}

CURRENT PAGE STATE:
URL: {current_url}
Title: {page_title}
Screenshot saved at: {screenshot_path}

PAGE CONTENT:
{page_content[:3000]}...

PREVIOUS FAILURES:
{[attempt.error for attempt in self.current_attempts]}

Perform a comprehensive analysis:
1. What type of page is this? (list page, form page, dashboard, etc.)
2. What UI pattern am I dealing with?
3. What elements are actually clickable/interactable?
4. Why might my previous attempts have failed?
5. What's the correct sequence of actions needed?

Provide a detailed action plan with specific, testable steps.

Respond with JSON:
{{
    "page_analysis": {{
        "page_type": "list|form|dashboard|other",
        "ui_pattern": "description of UI pattern",
        "key_elements": ["list of important UI elements visible"]
    }},
    "failure_analysis": "why previous attempts failed",
    "action_sequence": [
        {{
            "step": 1,
            "action": "click|type|navigate|wait",
            "target": "what to interact with",
            "selector": "specific selector",
            "expected_result": "what should happen",
            "confidence": 0.0-1.0
        }}
    ],
    "overall_confidence": 0.0-1.0
}}
"""
            
            response = await self.llm_service.generate_response(
                full_analysis_prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse and execute the action sequence
            from konfig.services.intelligent_planner import IntelligentPlanner
            planner = IntelligentPlanner()
            analysis = planner._extract_json_from_response(response)
            
            # Execute the action sequence
            for step in analysis.get("action_sequence", []):
                try:
                    success = await self._execute_analysis_step(step)
                    if success:
                        return {
                            "success": True,
                            "message": f"Full analysis succeeded at step {step['step']}",
                            "analysis": analysis.get("page_analysis"),
                            "screenshot": screenshot_path
                        }
                except Exception as e:
                    self.logger.warning(f"Analysis step {step['step']} failed: {e}")
                    continue
            
            return {
                "success": False,
                "error": "Full analysis generated plan but execution failed",
                "analysis": analysis.get("page_analysis"),
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            return {"success": False, "error": f"Full analysis failed: {e}"}
    
    async def _level_5_user_escalation(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 5: Escalate to user for guidance."""
        
        # Take screenshot for user reference with timeout
        try:
            screenshot_path = await asyncio.wait_for(
                self.web_interactor.take_screenshot(),
                timeout=10.0  # 10 second timeout instead of 30
            )
        except Exception:
            screenshot_path = None
        
        current_url = await self.web_interactor.get_current_url()
        
        # Log the escalation request
        self.logger.warning(
            "Escalating to user - all automated approaches failed",
            action=action.get("name"),
            url=current_url,
            attempts=len(self.current_attempts),
            screenshot=screenshot_path
        )
        
        # In a real implementation, this could:
        # 1. Send notification to user
        # 2. Open interactive session
        # 3. Wait for user input
        # 4. Log the guidance for future learning
        
        return {
            "success": False,
            "error": "USER_GUIDANCE_NEEDED",
            "escalation_info": {
                "action": action,
                "url": current_url,
                "screenshot": screenshot_path,
                "attempts": len(self.current_attempts),
                "message": f"Unable to execute '{action.get('name')}' automatically. Please provide guidance."
            }
        }
    
    async def _smart_click_by_text(self, text: str) -> bool:
        """Smart text-based clicking with multiple strategies."""
        try:
            # Strategy 1: Playwright text selector
            await self.web_interactor._page.click(f"text={text}", timeout=3000)
            return True
        except Exception:
            pass
        
        try:
            # Strategy 2: XPath with contains
            xpath = f"//*[contains(text(), '{text}')]"
            await self.web_interactor._page.click(f"xpath={xpath}", timeout=3000)
            return True
        except Exception:
            pass
        
        try:
            # Strategy 3: Table row containing text
            await self.web_interactor._page.click(f"tr:has-text('{text}')", timeout=3000)
            return True
        except Exception:
            pass
        
        return False
    
    async def _execute_analysis_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single step from the analysis."""
        if step["action"] == "click":
            if step.get("target"):
                return await self._smart_click_by_text(step["target"])
            elif step.get("selector"):
                result = await self.web_interactor.click(step["selector"])
                return result.get("success", False)
        
        elif step["action"] == "wait":
            await self.web_interactor._page.wait_for_timeout(2000)
            return True
        
        return False
    
    def _record_attempt(
        self,
        action: Dict[str, Any],
        level_info: IntelligenceLevel,
        success: bool,
        error: Optional[str] = None
    ):
        """Record an attempt for learning and debugging."""
        attempt = ActionAttempt(
            timestamp=time.time(),
            level=level_info.level,
            action=action,
            success=success,
            error=error,
            duration=time.time() - self.start_time if self.start_time else 0
        )
        self.current_attempts.append(attempt)
    
    def _record_success(self, level_info: IntelligenceLevel, action: Dict[str, Any]):
        """Record successful execution for learning.""" 
        self.logger.info(
            f"Success at level {level_info.level}",
            action=action.get("name"),
            duration=time.time() - self.start_time if self.start_time else 0
        )
        
        # Could add pattern learning here
        # self._learn_new_pattern(action, success_context)