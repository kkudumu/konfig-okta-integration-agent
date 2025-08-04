"""
Action Caching System for Vendor Integration Patterns

This system caches successful action sequences for similar vendor sites,
dramatically speeding up subsequent integrations by reusing proven patterns.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel

from konfig.utils.logging import LoggingMixin


class CachedAction(BaseModel):
    """Represents a cached action with metadata."""
    action_type: str
    selector: str
    text_content: Optional[str] = None
    value: Optional[str] = None
    description: str
    confidence_score: float
    element_hash: Optional[str] = None
    timestamp: datetime
    success_rate: float = 1.0
    usage_count: int = 1


class ActionSequence(BaseModel):
    """Represents a sequence of actions for a specific goal."""
    goal_description: str
    url_pattern: str
    domain: str
    actions: List[CachedAction]
    success_rate: float = 1.0
    usage_count: int = 1
    created_at: datetime
    last_used: datetime
    tags: List[str] = []


class ActionCacheManager(LoggingMixin):
    """
    Manages caching of successful action sequences for vendor integrations.
    
    This system learns from successful integrations and can replay similar
    patterns on new vendor sites, dramatically improving automation speed.
    """
    
    def __init__(self):
        super().__init__()
        self.setup_logging("action_cache")
        
        # In-memory cache (in production, this would use Redis or database)
        self._sequence_cache: Dict[str, ActionSequence] = {}
        self._element_cache: Dict[str, CachedAction] = {}
        
        # Cache configuration
        self.max_cache_age_days = 30
        self.min_confidence_threshold = 0.7
        self.max_sequences_per_domain = 10
    
    def generate_url_pattern(self, url: str) -> str:
        """Generate a pattern from URL for matching similar pages."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Create pattern by abstracting dynamic parts
            # Replace common dynamic patterns with wildcards
            import re
            
            # Replace UUIDs, IDs, and similar patterns
            path = re.sub(r'/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/', '/*/', path)
            path = re.sub(r'/\d+/', '/*/', path)  # Replace numeric IDs
            path = re.sub(r'/[a-zA-Z0-9]{20,}/', '/*/', path)  # Replace long tokens
            
            return f"{domain}{path}"
            
        except Exception as e:
            self.logger.warning(f"Failed to generate URL pattern: {e}")
            return url
    
    def generate_element_hash(self, element_info: Dict[str, Any]) -> str:
        """Generate a stable hash for an element."""
        try:
            # Use relevant element properties for hashing
            hash_data = {
                'tag': element_info.get('tagName', ''),
                'type': element_info.get('type', ''),
                'class': element_info.get('className', ''),
                'role': element_info.get('role', ''),
                'text': (element_info.get('text', '') or '')[:50],  # Limit text length
            }
            
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.md5(hash_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate element hash: {e}")
            return ""
    
    async def cache_action_sequence(
        self,
        goal: str,
        url: str,
        actions: List[Dict[str, Any]],
        success: bool = True,
        tags: Optional[List[str]] = None
    ) -> str:
        """Cache a successful action sequence."""
        try:
            url_pattern = self.generate_url_pattern(url)
            domain = urlparse(url).netloc.lower()
            sequence_id = hashlib.md5(f"{goal}:{url_pattern}".encode()).hexdigest()
            
            cached_actions = []
            for action in actions:
                cached_action = CachedAction(
                    action_type=action.get('action_type', 'unknown'),
                    selector=action.get('selector', ''),
                    text_content=action.get('text_content'),
                    value=action.get('value'),
                    description=action.get('description', ''),
                    confidence_score=action.get('confidence', 0.8),
                    element_hash=action.get('element_hash'),
                    timestamp=datetime.now()
                )
                cached_actions.append(cached_action)
            
            # Check if sequence already exists
            if sequence_id in self._sequence_cache:
                existing = self._sequence_cache[sequence_id]
                existing.usage_count += 1
                existing.last_used = datetime.now()
                if success:
                    # Update success rate using exponential moving average
                    existing.success_rate = 0.8 * existing.success_rate + 0.2 * 1.0
                else:
                    existing.success_rate = 0.8 * existing.success_rate + 0.2 * 0.0
            else:
                # Create new sequence
                sequence = ActionSequence(
                    goal_description=goal,
                    url_pattern=url_pattern,
                    domain=domain,
                    actions=cached_actions,
                    success_rate=1.0 if success else 0.0,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    tags=tags or []
                )
                self._sequence_cache[sequence_id] = sequence
            
            self.logger.info(f"Cached action sequence: {sequence_id} for goal: {goal}")
            return sequence_id
            
        except Exception as e:
            self.logger.error(f"Failed to cache action sequence: {e}")
            return ""
    
    async def find_cached_sequence(
        self,
        goal: str,
        url: str,
        similarity_threshold: float = 0.8
    ) -> Optional[ActionSequence]:
        """Find a cached action sequence that matches the goal and URL."""
        try:
            url_pattern = self.generate_url_pattern(url)
            domain = urlparse(url).netloc.lower()
            
            best_match = None
            best_score = 0.0
            
            for sequence in self._sequence_cache.values():
                # Skip old or low-confidence sequences
                if (datetime.now() - sequence.created_at).days > self.max_cache_age_days:
                    continue
                if sequence.success_rate < self.min_confidence_threshold:
                    continue
                
                score = self._calculate_sequence_similarity(
                    goal, url_pattern, domain, sequence
                )
                
                if score > best_score and score >= similarity_threshold:
                    best_score = score
                    best_match = sequence
            
            if best_match:
                self.logger.info(f"Found cached sequence with similarity {best_score:.2f}")
                return best_match
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find cached sequence: {e}")
            return None
    
    def _calculate_sequence_similarity(
        self,
        goal: str,
        url_pattern: str,
        domain: str,
        sequence: ActionSequence
    ) -> float:
        """Calculate similarity between current request and cached sequence."""
        score = 0.0
        
        # Domain similarity (highest weight)
        if sequence.domain == domain:
            score += 0.4
        elif domain in sequence.domain or sequence.domain in domain:
            score += 0.2
        
        # URL pattern similarity
        if sequence.url_pattern == url_pattern:
            score += 0.3
        else:
            # Calculate partial URL similarity
            url_parts = url_pattern.split('/')
            cached_parts = sequence.url_pattern.split('/')
            common_parts = sum(1 for a, b in zip(url_parts, cached_parts) if a == b)
            max_parts = max(len(url_parts), len(cached_parts))
            if max_parts > 0:
                score += 0.15 * (common_parts / max_parts)
        
        # Goal similarity (using simple keyword matching)
        goal_words = set(goal.lower().split())
        cached_words = set(sequence.goal_description.lower().split())
        if goal_words and cached_words:
            common_words = len(goal_words.intersection(cached_words))
            total_words = len(goal_words.union(cached_words))
            score += 0.2 * (common_words / total_words)
        
        # Boost score for frequently used and successful sequences
        usage_boost = min(0.1, sequence.usage_count * 0.01)
        success_boost = sequence.success_rate * 0.1
        score += usage_boost + success_boost
        
        return min(score, 1.0)
    
    async def execute_cached_sequence(
        self,
        sequence: ActionSequence,
        web_interactor,
        adaptation_mode: bool = True
    ) -> Dict[str, Any]:
        """Execute a cached action sequence with optional adaptation."""
        try:
            results = []
            successful_actions = 0
            
            for i, cached_action in enumerate(sequence.actions):
                try:
                    if adaptation_mode:
                        # Try to adapt selector if element not found
                        result = await self._execute_adaptive_action(
                            cached_action, web_interactor
                        )
                    else:
                        # Execute action as-is
                        result = await self._execute_cached_action(
                            cached_action, web_interactor
                        )
                    
                    results.append(result)
                    
                    if result.get('success'):
                        successful_actions += 1
                    else:
                        self.logger.warning(f"Cached action {i} failed: {result.get('error')}")
                        
                        # If critical early actions fail, abort sequence
                        if i < 2 and not adaptation_mode:
                            break
                    
                    # Small delay between actions
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error executing cached action {i}: {e}")
                    results.append({"success": False, "error": str(e)})
            
            success_rate = successful_actions / len(sequence.actions) if sequence.actions else 0
            
            # Update sequence stats
            sequence.usage_count += 1
            sequence.last_used = datetime.now()
            sequence.success_rate = 0.9 * sequence.success_rate + 0.1 * success_rate
            
            return {
                "success": success_rate > 0.7,
                "success_rate": success_rate,
                "actions_executed": len(results),
                "successful_actions": successful_actions,
                "results": results,
                "sequence_id": sequence.goal_description
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute cached sequence: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_cached_action(
        self,
        cached_action: CachedAction,
        web_interactor
    ) -> Dict[str, Any]:
        """Execute a single cached action."""
        try:
            if cached_action.action_type == "click":
                return await web_interactor.click(
                    cached_action.selector,
                    cached_action.text_content
                )
            elif cached_action.action_type == "type":
                return await web_interactor.type(
                    cached_action.selector,
                    cached_action.value or "",
                    clear_first=True
                )
            elif cached_action.action_type == "smart_fill":
                return await web_interactor.smart_fill(
                    cached_action.description,
                    cached_action.value or ""
                )
            elif cached_action.action_type == "smart_select":
                return await web_interactor.smart_select(
                    cached_action.description,
                    cached_action.value or ""
                )
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {cached_action.action_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_adaptive_action(
        self,
        cached_action: CachedAction,
        web_interactor
    ) -> Dict[str, Any]:
        """Execute action with adaptation if original selector fails."""
        # First try the original cached action
        result = await self._execute_cached_action(cached_action, web_interactor)
        
        if result.get('success'):
            return result
        
        # If failed, try adaptive approaches
        try:
            if cached_action.action_type in ["click", "type"]:
                # Try finding similar elements
                elements = await web_interactor.find_interactable_elements()
                
                # Look for elements with similar properties
                similar_element = self._find_similar_element(cached_action, elements)
                
                if similar_element:
                    adapted_selector = similar_element.get('selector')
                    if adapted_selector:
                        self.logger.info(f"Adapting selector from {cached_action.selector} to {adapted_selector}")
                        
                        # Create adapted action
                        adapted_action = CachedAction(
                            action_type=cached_action.action_type,
                            selector=adapted_selector,
                            text_content=cached_action.text_content,
                            value=cached_action.value,
                            description=cached_action.description,
                            confidence_score=cached_action.confidence_score * 0.8,  # Reduced confidence
                            timestamp=datetime.now()
                        )
                        
                        return await self._execute_cached_action(adapted_action, web_interactor)
            
            # If adaptation failed, try using smart methods
            if cached_action.action_type == "click":
                return await web_interactor.smart_click(cached_action.description)
            elif cached_action.action_type == "type":
                return await web_interactor.smart_fill(
                    cached_action.description,
                    cached_action.value or ""
                )
            
        except Exception as e:
            self.logger.warning(f"Adaptation failed: {e}")
        
        return result  # Return original result if adaptation fails
    
    def _find_similar_element(
        self,
        cached_action: CachedAction,
        current_elements: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find an element similar to the cached action's target."""
        if not cached_action.element_hash:
            return None
        
        best_match = None
        best_score = 0.0
        
        for element in current_elements:
            current_hash = self.generate_element_hash(element)
            
            # Simple similarity based on text content and element properties
            score = 0.0
            
            # Check text similarity
            cached_text = (cached_action.text_content or "").lower()
            element_text = (element.get('text') or "").lower()
            
            if cached_text and element_text:
                if cached_text == element_text:
                    score += 0.4
                elif cached_text in element_text or element_text in cached_text:
                    score += 0.2
            
            # Check element type/tag similarity
            if element.get('tagName', '').lower() in cached_action.selector.lower():
                score += 0.3
            
            # Check context similarity
            cached_desc = cached_action.description.lower()
            element_context = (element.get('context') or "").lower()
            
            if any(word in element_context for word in cached_desc.split()):
                score += 0.2
            
            # Boost confidence elements
            if element.get('confidence', 0) > 0.8:
                score += 0.1
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = element
        
        return best_match
    
    async def cleanup_cache(self):
        """Clean up old and low-performing cache entries."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_cache_age_days)
            
            # Remove old sequences
            sequences_to_remove = []
            for seq_id, sequence in self._sequence_cache.items():
                if (sequence.created_at < cutoff_date or 
                    sequence.success_rate < self.min_confidence_threshold):
                    sequences_to_remove.append(seq_id)
            
            for seq_id in sequences_to_remove:
                del self._sequence_cache[seq_id]
            
            # Limit sequences per domain
            domain_counts = {}
            for sequence in self._sequence_cache.values():
                domain_counts[sequence.domain] = domain_counts.get(sequence.domain, 0) + 1
            
            # Remove least recently used sequences if over limit
            for domain, count in domain_counts.items():
                if count > self.max_sequences_per_domain:
                    domain_sequences = [
                        (seq_id, seq) for seq_id, seq in self._sequence_cache.items()
                        if seq.domain == domain
                    ]
                    # Sort by last used (oldest first)
                    domain_sequences.sort(key=lambda x: x[1].last_used)
                    
                    # Remove excess sequences
                    excess = count - self.max_sequences_per_domain
                    for i in range(excess):
                        seq_id = domain_sequences[i][0]
                        del self._sequence_cache[seq_id]
            
            self.logger.info(f"Cache cleanup completed. Removed {len(sequences_to_remove)} old sequences")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")


# Global cache manager instance
_cache_manager = None

def get_action_cache() -> ActionCacheManager:
    """Get the global action cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ActionCacheManager()
    return _cache_manager