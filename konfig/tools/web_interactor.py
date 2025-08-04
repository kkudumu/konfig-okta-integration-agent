"""
WebInteractor Tool for Konfig.

This tool provides browser automation capabilities using Playwright,
enabling the agent to interact with web-based user interfaces during
SSO integration processes.
"""

import asyncio
import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from konfig.config.settings import get_settings
from konfig.utils.logging import LoggingMixin
from konfig.services.action_cache import get_action_cache


class WebInteractorError(Exception):
    """Base exception for WebInteractor errors."""
    pass


class ElementNotFoundError(WebInteractorError):
    """Raised when an element cannot be found."""
    pass


class NavigationError(WebInteractorError):
    """Raised when navigation fails."""
    pass


class WebInteractor(LoggingMixin):
    """
    Web automation tool using Playwright.
    
    This tool provides comprehensive browser automation capabilities
    for interacting with vendor admin interfaces and web forms during
    SSO integration processes.
    """
    
    def __init__(self):
        """Initialize the WebInteractor."""
        super().__init__()
        self.setup_logging("web_interactor")
        
        self.settings = get_settings()
        self.action_cache = get_action_cache()
        
        # Playwright instances
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        
        # Configuration
        self.headless = self.settings.browser.headless
        self.timeout = self.settings.browser.timeout_ms
        self.viewport = {
            "width": self.settings.browser.viewport_width,
            "height": self.settings.browser.viewport_height
        }
        self.user_agent = self.settings.browser.user_agent
        
        # State tracking
        self._current_url = None
        self._session_active = False
        self._dom_utils_injected = False
        
        self.logger.info("WebInteractor initialized", headless=self.headless)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()
    
    async def start_browser(self) -> None:
        """Start the browser and create a new page."""
        if self._session_active:
            self.logger.warning("Browser session already active")
            return
        
        try:
            self.logger.info("Starting browser session")
            
            # Start Playwright
            self._playwright = await async_playwright().start()
            
            # Launch browser
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-features=TranslateUI',
                    '--no-first-run',
                ]
            )
            
            # Create browser context
            self._context = await self._browser.new_context(
                viewport=self.viewport,
                user_agent=self.user_agent,
                ignore_https_errors=True,  # For development environments
            )
            
            # Create new page
            self._page = await self._context.new_page()
            
            # Set default timeout
            self._page.set_default_timeout(self.timeout)
            
            # Set up page event handlers
            self._page.on("response", self._on_response)
            self._page.on("pageerror", self._on_page_error)
            
            # Set up console monitoring for error context collection
            await self.setup_console_monitoring()
            
            self._session_active = True
            self.logger.info("Browser session started successfully")
            
        except Exception as e:
            self.log_error("start_browser", e)
            await self.close_browser()
            raise WebInteractorError(f"Failed to start browser: {e}")
    
    async def close_browser(self) -> None:
        """Close the browser and clean up resources."""
        if not self._session_active:
            return
        
        try:
            self.logger.info("Closing browser session")
            
            if self._page:
                await self._page.close()
                self._page = None
            
            if self._context:
                await self._context.close()
                self._context = None
            
            if self._browser:
                await self._browser.close()
                self._browser = None
            
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            
            self._session_active = False
            self._current_url = None
            
            self.logger.info("Browser session closed")
            
        except Exception as e:
            self.log_error("close_browser", e)
    
    def _ensure_browser_active(self) -> None:
        """Ensure browser session is active."""
        if not self._session_active or not self._page:
            raise WebInteractorError("Browser session not active. Call start_browser() first.")
    
    async def _on_response(self, response) -> None:
        """Handle page responses."""
        if response.status >= 400:
            self.logger.warning(
                "HTTP error response",
                url=response.url,
                status=response.status,
                status_text=response.status_text
            )
    
    async def _on_page_error(self, error) -> None:
        """Handle page JavaScript errors."""
        self.logger.warning("Page JavaScript error", error=str(error))
    
    # ========== NAVIGATION METHODS ==========
    
    async def navigate(self, url: str, wait_until: str = "networkidle") -> Dict[str, Any]:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete
                       ('load', 'domcontentloaded', 'networkidle')
        
        Returns:
            Navigation result with status and timing info
        """
        self._ensure_browser_active()
        self.log_method_call("navigate", url=url, wait_until=wait_until)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            response = await self._page.goto(url, wait_until=wait_until)
            
            end_time = asyncio.get_event_loop().time()
            duration_ms = (end_time - start_time) * 1000
            
            self._current_url = url
            
            # Inject DOM utilities after navigation
            await self.inject_dom_utils()
            
            result = {
                "success": True,
                "url": url,
                "final_url": self._page.url,
                "status_code": response.status if response else None,
                "duration_ms": round(duration_ms, 2),
                "title": await self._page.title(),
            }
            
            self.log_method_result("navigate", result, duration_ms)
            return result
            
        except Exception as e:
            self.log_error("navigate", e, url=url)
            raise NavigationError(f"Failed to navigate to {url}: {e}")
    
    async def get_current_url(self) -> str:
        """Get the current page URL."""
        self._ensure_browser_active()
        return self._page.url
    
    async def get_page_title(self) -> str:
        """Get the current page title."""
        self._ensure_browser_active()
        return await self._page.title()
    
    # ========== ELEMENT INTERACTION METHODS ==========
    
    async def click(
        self,
        selector: str,
        text_content: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Click on an element with advanced fallback strategies.
        
        Args:
            selector: CSS selector for the element
            text_content: Optional text content to match for disambiguation
            timeout: Optional timeout in milliseconds
        
        Returns:
            Click result with success status and timing
        """
        self._ensure_browser_active()
        self.log_method_call("click", selector=selector, text_content=text_content)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Use advanced chain clicking with multiple fallback strategies
            result = await self._chain_click(selector, text_content, timeout)
            
            end_time = asyncio.get_event_loop().time()
            duration_ms = (end_time - start_time) * 1000
            result["duration_ms"] = round(duration_ms, 2)
            
            self.log_method_result("click", result, duration_ms)
            return result
            
        except Exception as e:
            self.log_error("click", e, selector=selector, text_content=text_content)
            raise ElementNotFoundError(f"Failed to click element '{selector}': {e}")

    async def _chain_click(self, selector: str, text_content: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Advanced clicking with fallback strategies inspired by Skyvern.
        
        This method tries multiple approaches to click an element:
        1. Direct click on target element
        2. Click on associated label (for inputs)
        3. Click on parent element if target is blocked
        4. Click on first clickable child
        5. Force click using JavaScript
        6. Coordinate-based click as last resort
        """
        action_results = []
        
        # Strategy 1: Direct click
        try:
            if text_content:
                xpath_selector = f"//*[contains(text(), '{text_content}')]"
                element = self._page.locator(xpath_selector)
            else:
                element = self._page.locator(selector)
            
            await element.wait_for(state="visible", timeout=timeout or self.timeout)
            await element.click(timeout=timeout or self.timeout)
            
            return {
                "success": True,
                "selector": selector,
                "text_content": text_content,
                "strategy": "direct_click"
            }
            
        except Exception as e:
            action_results.append(f"direct_click: {str(e)}")
        
        # Strategy 2: Click associated label (for form inputs)
        try:
            label_selector = await self._find_associated_label(selector)
            if label_selector:
                label_element = self._page.locator(label_selector)
                await label_element.click(timeout=timeout or self.timeout)
                
                return {
                    "success": True,
                    "selector": selector,
                    "strategy": "label_click",
                    "label_selector": label_selector
                }
                
        except Exception as e:
            action_results.append(f"label_click: {str(e)}")
        
        # Strategy 3: Click parent if target is blocked
        try:
            parent_element = self._page.locator(selector).locator("..")
            if await parent_element.count() > 0:
                await parent_element.click(timeout=timeout or self.timeout)
                
                return {
                    "success": True,
                    "selector": selector,
                    "strategy": "parent_click"
                }
                
        except Exception as e:
            action_results.append(f"parent_click: {str(e)}")
        
        # Strategy 4: Click first clickable child
        try:
            child_selectors = ["button", "a", "input", "[onclick]", "[role='button']"]
            for child_sel in child_selectors:
                child_element = self._page.locator(selector).locator(child_sel).first
                if await child_element.count() > 0:
                    await child_element.click(timeout=timeout or self.timeout)
                    
                    return {
                        "success": True,
                        "selector": selector,
                        "strategy": "child_click",
                        "child_selector": child_sel
                    }
                    
        except Exception as e:
            action_results.append(f"child_click: {str(e)}")
        
        # Strategy 5: Force click using JavaScript
        try:
            clicked = await self._page.evaluate(f"""
                (selector) => {{
                    const element = document.querySelector(selector);
                    if (element) {{
                        element.click();
                        return true;
                    }}
                    return false;
                }}
            """, selector)
            
            if clicked:
                return {
                    "success": True,
                    "selector": selector,
                    "strategy": "javascript_click"
                }
            
        except Exception as e:
            action_results.append(f"javascript_click: {str(e)}")
        
        # Strategy 6: Coordinate-based click (last resort)
        try:
            element = self._page.locator(selector)
            if await element.count() > 0:
                box = await element.bounding_box()
                if box:
                    x = box['x'] + box['width'] / 2
                    y = box['y'] + box['height'] / 2
                    await self._page.mouse.click(x, y)
                    
                    return {
                        "success": True,
                        "selector": selector,
                        "strategy": "coordinate_click",
                        "coordinates": {"x": x, "y": y}
                    }
                    
        except Exception as e:
            action_results.append(f"coordinate_click: {str(e)}")
        
        # All strategies failed
        return {
            "success": False,
            "selector": selector,
            "error": "All clicking strategies failed",
            "failed_attempts": action_results
        }

    async def _find_associated_label(self, input_selector: str) -> Optional[str]:
        """Find label associated with an input element."""
        try:
            # Check for label with 'for' attribute
            element_id = await self._page.evaluate(f"""
                (selector) => {{
                    const element = document.querySelector(selector);
                    return element ? element.id : null;
                }}
            """, input_selector)
            
            if element_id:
                label_exists = await self._page.locator(f"label[for='{element_id}']").count()
                if label_exists > 0:
                    return f"label[for='{element_id}']"
            
            # Check for wrapping label
            wrapped_label = await self._page.evaluate(f"""
                (selector) => {{
                    const element = document.querySelector(selector);
                    if (element) {{
                        const label = element.closest('label');
                        if (label) {{
                            let labelSelector = 'label';
                            if (label.id) labelSelector += '#' + label.id;
                            else if (label.className) labelSelector += '.' + label.className.split(' ')[0];
                            return labelSelector;
                        }}
                    }}
                    return null;
                }}
            """, input_selector)
            
            return wrapped_label
            
        except Exception:
            return None
    
    async def type(
        self,
        selector: str,
        text: str,
        secret: bool = False,
        clear_first: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Type text into an input element.
        
        Args:
            selector: CSS selector for the input element
            text: Text to type
            secret: Whether the text is sensitive (won't be logged)
            clear_first: Whether to clear existing text first
            timeout: Optional timeout in milliseconds
        
        Returns:
            Type result with success status and timing
        """
        self._ensure_browser_active()
        log_text = "[REDACTED]" if secret else text
        self.log_method_call("type", selector=selector, text=log_text, clear_first=clear_first)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            element = self._page.locator(selector)
            
            # Wait for element to be visible
            await element.wait_for(state="visible", timeout=timeout or self.timeout)
            
            # Clear existing text if requested
            if clear_first:
                await element.clear(timeout=timeout or self.timeout)
            
            # Type the text
            await element.fill(text, timeout=timeout or self.timeout)
            
            end_time = asyncio.get_event_loop().time()
            duration_ms = (end_time - start_time) * 1000
            
            result = {
                "success": True,
                "selector": selector,
                "text_length": len(text),
                "secret": secret,
                "duration_ms": round(duration_ms, 2),
            }
            
            self.log_method_result("type", result, duration_ms)
            return result
            
        except Exception as e:
            self.log_error("type", e, selector=selector, secret=secret)
            raise ElementNotFoundError(f"Failed to type into element '{selector}': {e}")
    
    async def select_option(
        self,
        selector: str,
        value: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select an option from a dropdown.
        
        Args:
            selector: CSS selector for the select element
            value: Value to select
            timeout: Optional timeout in milliseconds
        
        Returns:
            Selection result with success status
        """
        self._ensure_browser_active()
        self.log_method_call("select_option", selector=selector, value=value)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            element = self._page.locator(selector)
            
            # Wait for element to be visible
            await element.wait_for(state="visible", timeout=timeout or self.timeout)
            
            # Select the option
            await element.select_option(value, timeout=timeout or self.timeout)
            
            end_time = asyncio.get_event_loop().time()
            duration_ms = (end_time - start_time) * 1000
            
            result = {
                "success": True,
                "selector": selector,
                "value": value,
                "duration_ms": round(duration_ms, 2),
            }
            
            self.log_method_result("select_option", result, duration_ms)
            return result
            
        except Exception as e:
            self.log_error("select_option", e, selector=selector, value=value)
            raise ElementNotFoundError(f"Failed to select option '{value}' in '{selector}': {e}")
    
    # ========== ELEMENT QUERY METHODS ==========
    
    async def get_element_text(
        self,
        selector: str,
        timeout: Optional[int] = None
    ) -> str:
        """
        Get the visible text content of an element.
        
        Args:
            selector: CSS selector for the element
            timeout: Optional timeout in milliseconds
        
        Returns:
            Element text content
        """
        self._ensure_browser_active()
        self.log_method_call("get_element_text", selector=selector)
        
        try:
            element = self._page.locator(selector)
            await element.wait_for(state="visible", timeout=timeout or self.timeout)
            
            text = await element.text_content(timeout=timeout or self.timeout)
            
            self.log_method_result("get_element_text", {"text_length": len(text or "")})
            return text or ""
            
        except Exception as e:
            self.log_error("get_element_text", e, selector=selector)
            raise ElementNotFoundError(f"Failed to get text from element '{selector}': {e}")
    
    async def get_element_value(
        self,
        selector: str,
        timeout: Optional[int] = None
    ) -> str:
        """
        Get the value property of an input element.
        
        Args:
            selector: CSS selector for the input element
            timeout: Optional timeout in milliseconds
        
        Returns:
            Element value
        """
        self._ensure_browser_active()
        self.log_method_call("get_element_value", selector=selector)
        
        try:
            element = self._page.locator(selector)
            await element.wait_for(state="visible", timeout=timeout or self.timeout)
            
            value = await element.input_value(timeout=timeout or self.timeout)
            
            self.log_method_result("get_element_value", {"value_length": len(value or "")})
            return value or ""
            
        except Exception as e:
            self.log_error("get_element_value", e, selector=selector)
            raise ElementNotFoundError(f"Failed to get value from element '{selector}': {e}")
    
    async def get_element_attribute(
        self,
        selector: str,
        attribute: str,
        timeout: Optional[int] = None
    ) -> Optional[str]:
        """
        Get an attribute value from an element.
        
        Args:
            selector: CSS selector for the element
            attribute: Attribute name
            timeout: Optional timeout in milliseconds
        
        Returns:
            Attribute value or None
        """
        self._ensure_browser_active()
        self.log_method_call("get_element_attribute", selector=selector, attribute=attribute)
        
        try:
            element = self._page.locator(selector)
            await element.wait_for(state="visible", timeout=timeout or self.timeout)
            
            value = await element.get_attribute(attribute, timeout=timeout or self.timeout)
            
            self.log_method_result("get_element_attribute", {"has_value": value is not None})
            return value
            
        except Exception as e:
            self.log_error("get_element_attribute", e, selector=selector, attribute=attribute)
            raise ElementNotFoundError(f"Failed to get attribute '{attribute}' from element '{selector}': {e}")
    
    async def element_exists(self, selector: str, timeout: int = 1000) -> bool:
        """
        Check if an element exists on the page.
        
        Args:
            selector: CSS selector for the element
            timeout: Timeout in milliseconds
        
        Returns:
            True if element exists, False otherwise
        """
        self._ensure_browser_active()
        
        try:
            element = self._page.locator(selector)
            await element.wait_for(state="visible", timeout=timeout)
            return True
        except:
            return False
    
    # ========== PAGE ANALYSIS METHODS ==========
    
    async def get_current_dom(self, simplify: bool = True) -> str:
        """
        Get a simplified representation of the current page's DOM.
        
        Args:
            simplify: Whether to simplify the DOM for LLM analysis
        
        Returns:
            DOM representation as string
        """
        self._ensure_browser_active()
        self.log_method_call("get_current_dom", simplify=simplify)
        
        try:
            if simplify:
                # Get a simplified DOM optimized for LLM analysis
                dom_script = """
                () => {
                    const simplifyElement = (el) => {
                        if (!el || el.nodeType !== 1) return null;
                        
                        // Skip hidden elements
                        const style = window.getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden') {
                            return null;
                        }
                        
                        // Skip script, style, and meta tags
                        if (['SCRIPT', 'STYLE', 'META', 'LINK', 'NOSCRIPT'].includes(el.tagName)) {
                            return null;
                        }
                        
                        const result = {
                            tag: el.tagName.toLowerCase(),
                            text: '',
                            attributes: {}
                        };
                        
                        // Get important attributes
                        const importantAttrs = ['id', 'class', 'name', 'type', 'value', 'href', 'src', 'role', 'data-testid'];
                        for (const attr of importantAttrs) {
                            if (el.hasAttribute(attr)) {
                                result.attributes[attr] = el.getAttribute(attr);
                            }
                        }
                        
                        // Get text content for interactive elements
                        if (['INPUT', 'BUTTON', 'A', 'SPAN', 'DIV', 'LABEL', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(el.tagName)) {
                            const text = el.textContent?.trim();
                            if (text && text.length < 100) {
                                result.text = text;
                            }
                        }
                        
                        // Recursively process children for form elements
                        if (['FORM', 'DIV', 'SECTION', 'MAIN', 'ARTICLE'].includes(el.tagName)) {
                            result.children = [];
                            for (const child of el.children) {
                                const childResult = simplifyElement(child);
                                if (childResult) {
                                    result.children.push(childResult);
                                }
                            }
                            if (result.children.length === 0) {
                                delete result.children;
                            }
                        }
                        
                        return result;
                    };
                    
                    return JSON.stringify(simplifyElement(document.body), null, 2);
                }
                """
                
                dom = await self._page.evaluate(dom_script)
            else:
                # Get full HTML
                dom = await self._page.content()
            
            self.log_method_result("get_current_dom", {"dom_length": len(dom)})
            return dom
            
        except Exception as e:
            self.log_error("get_current_dom", e)
            raise WebInteractorError(f"Failed to get DOM: {e}")
    
    async def find_elements_by_text(self, text: str, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find elements containing specific text.
        
        Args:
            text: Text to search for
            tag: Optional tag name to filter by
        
        Returns:
            List of element info dictionaries
        """
        self._ensure_browser_active()
        self.log_method_call("find_elements_by_text", text=text, tag=tag)
        
        try:
            script = f"""
            () => {{
                const elements = [];
                const xpath = {repr(f"//{tag or '*'}[contains(text(), '{text}')]")};
                const result = document.evaluate(xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                
                for (let i = 0; i < result.snapshotLength; i++) {{
                    const el = result.snapshotItem(i);
                    elements.push({{
                        tag: el.tagName.toLowerCase(),
                        text: el.textContent.trim(),
                        id: el.id || null,
                        className: el.className || null,
                        selector: el.id ? `#${{el.id}}` : `.${{el.className.split(' ')[0]}}` || el.tagName.toLowerCase()
                    }});
                }}
                
                return elements;
            }}
            """
            
            elements = await self._page.evaluate(script)
            
            self.log_method_result("find_elements_by_text", {"num_elements": len(elements)})
            return elements
            
        except Exception as e:
            self.log_error("find_elements_by_text", e, text=text, tag=tag)
            return []
    
    # ========== UTILITY METHODS ==========
    
    async def wait_for_element(
        self,
        selector: str,
        timeout: int,
        state: str = "visible"
    ) -> bool:
        """
        Wait for an element to appear in the specified state.
        
        Args:
            selector: CSS selector for the element
            timeout: Timeout in milliseconds
            state: Element state to wait for ('visible', 'hidden', 'attached')
        
        Returns:
            True if element reached the state, False if timeout
        """
        self._ensure_browser_active()
        self.log_method_call("wait_for_element", selector=selector, timeout=timeout, state=state)
        
        try:
            element = self._page.locator(selector)
            await element.wait_for(state=state, timeout=timeout)
            
            self.log_method_result("wait_for_element", {"success": True})
            return True
            
        except Exception as e:
            self.log_error("wait_for_element", e, selector=selector, timeout=timeout, state=state)
            return False
    
    async def take_screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False
    ) -> str:
        """
        Take a screenshot of the current page.
        
        Args:
            path: Optional file path to save screenshot
            full_page: Whether to capture the full page
        
        Returns:
            Path to the saved screenshot
        """
        self._ensure_browser_active()
        self.log_method_call("take_screenshot", path=path, full_page=full_page)
        
        try:
            if not path:
                timestamp = asyncio.get_event_loop().time()
                path = f"screenshots/screenshot_{timestamp}.png"
            
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            await self._page.screenshot(path=path, full_page=full_page)
            
            self.log_method_result("take_screenshot", {"path": path})
            return path
            
        except Exception as e:
            self.log_error("take_screenshot", e, path=path)
            raise WebInteractorError(f"Failed to take screenshot: {e}")
    
    async def scroll_to_element(self, selector: str) -> bool:
        """
        Scroll to bring an element into view.
        
        Args:
            selector: CSS selector for the element
        
        Returns:
            True if successful
        """
        self._ensure_browser_active()
        
        try:
            element = self._page.locator(selector)
            await element.scroll_into_view_if_needed()
            return True
        except Exception as e:
            self.log_error("scroll_to_element", e, selector=selector)
            return False
    
    async def wait_for_navigation(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Wait for navigation to complete.
        
        Args:
            timeout: Optional timeout in milliseconds
        
        Returns:
            Navigation result
        """
        self._ensure_browser_active()
        
        try:
            response = await self._page.wait_for_load_state(
                "networkidle",
                timeout=timeout or self.timeout
            )
            
            return {
                "success": True,
                "url": self._page.url,
                "title": await self._page.title(),
            }
            
        except Exception as e:
            self.log_error("wait_for_navigation", e)
            return {"success": False, "error": str(e)}
    
    # ========== FORM HANDLING METHODS ==========
    
    async def fill_form(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill a form with multiple fields.
        
        Args:
            form_data: Dictionary mapping selectors to values
        
        Returns:
            Form filling result
        """
        self._ensure_browser_active()
        self.log_method_call("fill_form", num_fields=len(form_data))
        
        results = {}
        
        for selector, value in form_data.items():
            try:
                if isinstance(value, dict):
                    # Handle special field types
                    field_type = value.get("type", "text")
                    field_value = value.get("value")
                    secret = value.get("secret", False)
                    
                    if field_type == "select":
                        result = await self.select_option(selector, field_value)
                    elif field_type == "click":
                        result = await self.click(selector)
                    else:
                        result = await self.type(selector, field_value, secret=secret)
                else:
                    # Simple text input
                    result = await self.type(selector, str(value))
                
                results[selector] = result
                
            except Exception as e:
                self.log_error("fill_form", e, selector=selector)
                results[selector] = {"success": False, "error": str(e)}
        
        success_count = sum(1 for r in results.values() if r.get("success", False))
        
        final_result = {
            "success": success_count == len(form_data),
            "total_fields": len(form_data),
            "successful_fields": success_count,
            "results": results
        }
        
        self.log_method_result("fill_form", final_result)
        return final_result
    
    # ========== ERROR CONTEXT METHODS ==========
    
    async def screenshot(self, path: Optional[str] = None) -> str:
        """
        Alias for take_screenshot for compatibility with error context collector.
        
        Args:
            path: Optional file path to save screenshot
            
        Returns:
            Path to the saved screenshot
        """
        return await self.take_screenshot(path, full_page=True)
    
    async def get_console_logs(self) -> List[Dict[str, Any]]:
        """
        Get browser console logs.
        
        Returns:
            List of console log entries
        """
        self._ensure_browser_active()
        self.log_method_call("get_console_logs")
        
        try:
            # Get console messages from the page
            # Note: We need to collect these during page lifecycle
            logs = []
            
            # Try to get any stored console messages
            script = """
            () => {
                // Get any console messages if available
                if (window.console && window.console._messages) {
                    return window.console._messages;
                }
                return [];
            }
            """
            
            try:
                stored_logs = await self._page.evaluate(script)
                logs.extend(stored_logs)
            except:
                pass
            
            # If no stored logs, return basic page info
            if not logs:
                logs = [{
                    "level": "info",
                    "text": f"Page loaded: {await self.get_current_url()}",
                    "timestamp": asyncio.get_event_loop().time()
                }]
            
            self.log_method_result("get_console_logs", {"num_logs": len(logs)})
            return logs
            
        except Exception as e:
            self.log_error("get_console_logs", e)
            return []
    
    async def get_viewport_size(self) -> Dict[str, int]:
        """
        Get current viewport size.
        
        Returns:
            Dictionary with width and height
        """
        self._ensure_browser_active()
        self.log_method_call("get_viewport_size")
        
        try:
            viewport = await self._page.evaluate("""
                () => ({
                    width: window.innerWidth,
                    height: window.innerHeight
                })
            """)
            
            self.log_method_result("get_viewport_size", viewport)
            return viewport
            
        except Exception as e:
            self.log_error("get_viewport_size", e)
            return {"width": 0, "height": 0}
    
    async def inject_dom_utils(self):
        """Inject enhanced DOM utilities into the current page."""
        if not self._page:
            return
        
        try:
            # Read the DOM utilities JavaScript file
            dom_utils_path = Path(__file__).parent / "skyvern_dom_utils.js"
            if dom_utils_path.exists():
                with open(dom_utils_path, 'r') as f:
                    dom_utils_js = f.read()
                
                # Check if already injected
                already_injected = await self._page.evaluate("""
                    () => typeof window.skyvernDomUtils !== 'undefined'
                """)
                
                if not already_injected:
                    # Inject the utilities into the page using evaluate
                    await self._page.evaluate(dom_utils_js)
                    self._dom_utils_injected = True
                    self.logger.debug("DOM utilities injected successfully")
                else:
                    self.logger.debug("DOM utilities already injected")
            else:
                self.logger.warning("DOM utilities file not found")
                
        except Exception as e:
            self.logger.error(f"Failed to inject DOM utilities: {e}")
    
    async def find_interactable_elements(self) -> List[Dict[str, Any]]:
        """Find all interactable elements on the current page using enhanced detection."""
        self._ensure_browser_active()
        await self.inject_dom_utils()
        
        try:
            # Use the enhanced DOM utilities to find interactable elements
            elements = await self._page.evaluate("""
                () => {
                    if (typeof window.skyvernDomUtils === 'undefined') {
                        return [];
                    }
                    return window.skyvernDomUtils.findInteractableElementsWithContext();
                }
            """)
            
            # Filter and enhance the results
            filtered_elements = []
            for element in elements:
                # Skip elements that are too small or off-screen
                rect = element.get('rect', {})
                if rect.get('width', 0) < 3 or rect.get('height', 0) < 3:
                    continue
                
                # Add additional metadata
                element['confidence'] = self._calculate_element_confidence(element)
                filtered_elements.append(element)
            
            # Sort by confidence score (highest first)
            filtered_elements.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            self.logger.debug(f"Found {len(filtered_elements)} interactable elements")
            return filtered_elements
            
        except Exception as e:
            self.logger.error(f"Failed to find interactable elements: {e}")
            return []
    
    async def find_best_element_for_action(self, action_description: str, element_type: str = None) -> Optional[Dict[str, Any]]:
        """Find the best element for a given action using LLM-enhanced matching."""
        elements = await self.find_interactable_elements()
        
        if not elements:
            return None
        
        # Filter by element type if specified
        if element_type:
            elements = [e for e in elements if e.get('tagName') == element_type.lower()]
        
        # Use simple scoring for now (could be enhanced with LLM later)
        best_element = None
        best_score = 0
        
        action_lower = action_description.lower()
        
        for element in elements:
            score = 0
            
            # Check text content
            text = element.get('text', '').lower()
            if action_lower in text or text in action_lower:
                score += 50
            
            # Check context
            context = element.get('context', '').lower()
            if action_lower in context or any(word in context for word in action_lower.split()):
                score += 30
            
            # Check attributes
            attrs = element.get('attributes', {})
            for attr_name, attr_value in attrs.items():
                if isinstance(attr_value, str) and action_lower in attr_value.lower():
                    score += 20
            
            # Boost score for common action elements
            tag = element.get('tagName', '')
            if 'login' in action_lower and tag in ['button', 'input']:
                score += 25
            if 'submit' in action_lower and tag == 'button':
                score += 25
            if 'click' in action_lower and attrs.get('type') == 'submit':
                score += 25
            
            # Add confidence score
            score += element.get('confidence', 0)
            
            if score > best_score:
                best_score = score
                best_element = element
        
        if best_element:
            self.logger.debug(f"Best element for '{action_description}': {best_element.get('selector')} (score: {best_score})")
        
        return best_element
    
    async def smart_click(self, action_description: str, element_type: str = None) -> Dict[str, Any]:
        """Intelligently click an element based on description."""
        self._ensure_browser_active()
        
        try:
            # Find the best element for this action
            element = await self.find_best_element_for_action(action_description, element_type)
            
            if not element:
                return {
                    "success": False,
                    "error": f"No suitable element found for action: {action_description}"
                }
            
            selector = element.get('selector')
            if not selector:
                return {
                    "success": False,
                    "error": "Element found but no valid selector generated"
                }
            
            # Attempt to click using the selector
            click_result = await self.click(selector)
            
            if click_result.get('success'):
                click_result['element_info'] = {
                    'text': element.get('text', ''),
                    'context': element.get('context', ''),
                    'confidence': element.get('confidence', 0)
                }
            
            return click_result
            
        except Exception as e:
            self.logger.error(f"Smart click failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def smart_fill(self, field_description: str, value: str) -> Dict[str, Any]:
        """Intelligently fill a form field with auto-completion support."""
        self._ensure_browser_active()
        
        try:
            # Look for input/textarea elements
            element = await self.find_best_element_for_action(field_description, "input")
            
            if not element:
                # Try textarea as fallback
                element = await self.find_best_element_for_action(field_description, "textarea")
            
            if not element:
                return {
                    "success": False,
                    "error": f"No suitable input field found for: {field_description}"
                }
            
            selector = element.get('selector')
            if not selector:
                return {
                    "success": False,
                    "error": "Field found but no valid selector generated"
                }
            
            # Check if this might be an auto-completion field
            is_autocomplete = await self._detect_autocomplete_field(selector)
            
            if is_autocomplete:
                # Use enhanced auto-completion handling
                result = await self._handle_autocomplete_input(selector, value, field_description)
                result["element_info"] = {
                    "text": element.get('text', ''),
                    "context": element.get('context', ''),
                    "confidence": element.get('confidence', 0)
                }
                return result
            else:
                # Regular text input
                fill_result = await self.type(selector, value, clear_first=True)
                
                if fill_result.get('success'):
                    fill_result['element_info'] = {
                        'text': element.get('text', ''),
                        'context': element.get('context', ''),
                        'confidence': element.get('confidence', 0)
                    }
                
                return fill_result
            
        except Exception as e:
            self.logger.error(f"Smart fill failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _detect_autocomplete_field(self, selector: str) -> bool:
        """Detect if an input field has auto-completion capabilities."""
        try:
            # Check for common auto-completion indicators
            result = await self._page.evaluate(f"""
                (selector) => {{
                    const element = document.querySelector(selector);
                    if (!element) return false;
                    
                    // Check for autocomplete attributes
                    if (element.hasAttribute('autocomplete') && element.getAttribute('autocomplete') !== 'off') {{
                        return true;
                    }}
                    
                    // Check for datalist association
                    if (element.hasAttribute('list')) {{
                        return true;
                    }}
                    
                    // Check for common auto-completion classes
                    const className = element.className || '';
                    const autoClasses = ['autocomplete', 'typeahead', 'suggest', 'search'];
                    if (autoClasses.some(cls => className.toLowerCase().includes(cls))) {{
                        return true;
                    }}
                    
                    // Check parent for auto-completion containers
                    const parent = element.parentElement;
                    if (parent) {{
                        const parentClass = parent.className || '';
                        if (autoClasses.some(cls => parentClass.toLowerCase().includes(cls))) {{
                            return true;
                        }}
                    }}
                    
                    return false;
                }}
            """, selector)
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Failed to detect autocomplete field: {e}")
            return False

    async def _handle_autocomplete_input(self, selector: str, target_text: str, field_context: str) -> Dict[str, Any]:
        """Handle auto-completion input with intelligent option selection."""
        try:
            # Clear the field first
            await self._page.locator(selector).clear()
            
            # Type partial text to trigger auto-completion
            partial_text = target_text[:3] if len(target_text) > 3 else target_text
            await self._page.locator(selector).type(partial_text)
            
            # Wait for auto-completion dropdown to appear
            await asyncio.sleep(2)
            
            # Look for newly appeared dropdown options
            dropdown_options = await self._find_autocomplete_options()
            
            if dropdown_options:
                # Find best matching option
                best_match = await self._select_best_autocomplete_option(
                    dropdown_options, target_text, field_context
                )
                
                if best_match:
                    # Click the best matching option
                    await self._page.locator(best_match['selector']).click()
                    return {
                        "success": True,
                        "method": "autocomplete_selection",
                        "selected_option": best_match['text'],
                        "target_text": target_text
                    }
            
            # If no good auto-completion option found, type the full text
            await self._page.locator(selector).clear()
            await self._page.locator(selector).type(target_text)
            
            # Try pressing Enter or Tab to confirm
            await self._page.locator(selector).press("Tab")
            
            return {
                "success": True,
                "method": "direct_typing",
                "text": target_text
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Auto-completion handling failed: {e}"
            }

    async def _find_autocomplete_options(self) -> List[Dict[str, Any]]:
        """Find auto-completion dropdown options that appeared after typing."""
        try:
            # Look for common dropdown option patterns
            options = await self._page.evaluate("""
                () => {
                    const options = [];
                    
                    // Common selectors for auto-completion options
                    const selectors = [
                        '[role="option"]',
                        '.autocomplete-option',
                        '.suggestion',
                        '.typeahead-option',
                        'ul.dropdown-menu li',
                        '.search-results li',
                        '[data-value]'
                    ];
                    
                    selectors.forEach(sel => {
                        const elements = document.querySelectorAll(sel);
                        elements.forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0 && rect.top >= 0) {
                                options.push({
                                    text: el.textContent.trim(),
                                    selector: sel + ':nth-of-type(' + (index + 1) + ')',
                                    visible: true
                                });
                            }
                        });
                    });
                    
                    return options;
                }
            """)
            
            return options or []
            
        except Exception as e:
            self.logger.debug(f"Failed to find autocomplete options: {e}")
            return []

    async def _select_best_autocomplete_option(self, options: List[Dict[str, Any]], target_text: str, context: str) -> Optional[Dict[str, Any]]:
        """Select the best matching auto-completion option using intelligent scoring."""
        if not options:
            return None
        
        best_option = None
        best_score = 0
        
        target_lower = target_text.lower()
        context_lower = context.lower()
        
        for option in options:
            option_text = option.get('text', '').lower()
            score = 0
            
            # Exact match gets highest score
            if option_text == target_lower:
                score = 100
            # Starts with target text
            elif option_text.startswith(target_lower):
                score = 80
            # Contains target text
            elif target_lower in option_text:
                score = 60
            # Fuzzy matching for partial matches
            else:
                # Simple fuzzy matching based on common characters
                common_chars = sum(1 for c in target_lower if c in option_text)
                score = (common_chars / len(target_lower)) * 40 if target_lower else 0
            
            # Boost score if option contains context keywords
            if any(word in option_text for word in context_lower.split()):
                score += 10
            
            if score > best_score:
                best_score = score
                best_option = option
        
        # Only return option if it has a reasonable confidence score
        return best_option if best_score >= 50 else None

    async def smart_select(self, field_description: str, option_text: str, max_levels: int = 3) -> Dict[str, Any]:
        """Intelligently select from dropdown menus with multi-level support."""
        self._ensure_browser_active()
        
        try:
            # Find the select element or trigger element
            element = await self.find_best_element_for_action(field_description, "select")
            
            if not element:
                # Look for clickable elements that might trigger dropdowns
                element = await self.find_best_element_for_action(field_description)
            
            if not element:
                return {
                    "success": False,
                    "error": f"No suitable select element found for: {field_description}"
                }
            
            selector = element.get('selector')
            if not selector:
                return {
                    "success": False,
                    "error": "Element found but no valid selector generated"
                }
            
            # Handle multi-level selection
            return await self._handle_multilevel_selection(selector, option_text, field_description, max_levels)
            
        except Exception as e:
            self.logger.error(f"Smart select failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_multilevel_selection(self, selector: str, target_option: str, context: str, max_levels: int) -> Dict[str, Any]:
        """Handle multi-level dropdown selection scenarios."""
        selection_history = []
        
        for level in range(max_levels):
            try:
                # Click to open the dropdown (or continue to next level)
                await self._page.locator(selector).click()
                await asyncio.sleep(1)  # Wait for dropdown to appear
                
                # Find available options
                options = await self._find_dropdown_options()
                
                if not options:
                    if level == 0:
                        return {
                            "success": False,
                            "error": "No dropdown options found after clicking element"
                        }
                    else:
                        # No more levels available, we're done
                        break
                
                # Find the best matching option for this level
                best_match = await self._select_best_multilevel_option(
                    options, target_option, context, selection_history
                )
                
                if not best_match:
                    return {
                        "success": False,
                        "error": f"No suitable option found at level {level + 1}",
                        "available_options": [opt.get('text', '') for opt in options[:5]]
                    }
                
                # Click the selected option
                await self._page.locator(best_match['selector']).click()
                selection_history.append({
                    "level": level + 1,
                    "selected": best_match['text'],
                    "selector": best_match['selector']
                })
                
                # Wait to see if another level opens
                await asyncio.sleep(2)
                
                # Check if this was the final selection
                if await self._is_selection_complete(selector, target_option):
                    return {
                        "success": True,
                        "method": "multilevel_selection",
                        "levels_used": level + 1,
                        "selection_history": selection_history,
                        "final_value": best_match['text']
                    }
                
                # If exact match found, we're likely done
                if best_match['text'].lower() == target_option.lower():
                    return {
                        "success": True,
                        "method": "exact_match",
                        "levels_used": level + 1,
                        "selection_history": selection_history,
                        "final_value": best_match['text']
                    }
                
            except Exception as e:
                self.logger.warning(f"Error at selection level {level + 1}: {e}")
                break
        
        # Return partial success if we made some selections
        if selection_history:
            return {
                "success": True,
                "method": "partial_multilevel_selection",
                "levels_used": len(selection_history),
                "selection_history": selection_history,
                "warning": f"Completed {len(selection_history)} levels, may need manual verification"
            }
        
        return {
            "success": False,
            "error": "Failed to make any selections",
            "max_levels_attempted": max_levels
        }

    async def _find_dropdown_options(self) -> List[Dict[str, Any]]:
        """Find dropdown options that are currently visible."""
        try:
            # Look for various dropdown option patterns
            options = await self._page.evaluate("""
                () => {
                    const options = [];
                    
                    // Various selectors for dropdown options
                    const selectors = [
                        'option',  // Standard select options
                        '[role="option"]',
                        '.dropdown-item',
                        '.select-option',
                        '.menu-item',
                        'li[data-value]',
                        '.option',
                        '[role="menuitem"]',
                        '.choice'
                    ];
                    
                    selectors.forEach(sel => {
                        const elements = document.querySelectorAll(sel);
                        elements.forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            // Only include visible options
                            if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.bottom <= window.innerHeight * 1.5) {
                                const text = el.textContent?.trim() || '';
                                if (text) {
                                    options.push({
                                        text: text,
                                        selector: sel + ':nth-of-type(' + (index + 1) + ')',
                                        element_selector: sel,
                                        index: index,
                                        value: el.getAttribute('value') || el.getAttribute('data-value') || text,
                                        visible: true
                                    });
                                }
                            }
                        });
                    });
                    
                    return options;
                }
            """)
            
            return options or []
            
        except Exception as e:
            self.logger.debug(f"Failed to find dropdown options: {e}")
            return []

    async def _select_best_multilevel_option(self, options: List[Dict[str, Any]], target: str, context: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """Select best option considering multi-level context."""
        if not options:
            return None
        
        best_option = None
        best_score = 0
        
        target_lower = target.lower()
        context_lower = context.lower()
        
        # Keywords from previous selections for context
        history_keywords = set()
        for selection in history:
            history_keywords.update(selection['selected'].lower().split())
        
        for option in options:
            option_text = option.get('text', '').lower()
            score = 0
            
            # Exact match gets highest score
            if option_text == target_lower:
                score = 100
            # Starts with target
            elif option_text.startswith(target_lower):
                score = 85
            # Contains target
            elif target_lower in option_text:
                score = 70
            # Target contains option (for partial matches)
            elif option_text in target_lower:
                score = 65
            # Fuzzy matching
            else:
                common_chars = sum(1 for c in target_lower if c in option_text)
                score = (common_chars / max(len(target_lower), 1)) * 50
            
            # Boost score for context relevance
            context_words = context_lower.split()
            matching_context = sum(1 for word in context_words if word in option_text)
            score += matching_context * 5
            
            # Reduce score if conflicts with history
            conflicting_history = sum(1 for word in history_keywords if word in option_text)
            if conflicting_history > 0 and score < 90:  # Don't penalize exact matches
                score -= conflicting_history * 3
            
            if score > best_score:
                best_score = score
                best_option = option
        
        # Only return if confidence is reasonable
        return best_option if best_score >= 40 else None

    async def _is_selection_complete(self, original_selector: str, target: str) -> bool:
        """Check if the selection process is complete."""
        try:
            # Check if the original element now shows the selected value
            current_value = await self._page.evaluate(f"""
                (selector) => {{
                    const element = document.querySelector(selector);
                    if (!element) return null;
                    
                    // For select elements
                    if (element.tagName === 'SELECT') {{
                        return element.value || element.options[element.selectedIndex]?.text;
                    }}
                    
                    // For other elements, check text content or value
                    return element.value || element.textContent?.trim() || null;
                }}
            """, original_selector)
            
            if current_value and target.lower() in current_value.lower():
                return True
            
            # Check if no dropdown is currently visible
            dropdown_visible = await self._page.evaluate("""
                () => {
                    const dropdownSelectors = [
                        '.dropdown-menu:visible',
                        '[role="listbox"]:visible',
                        '.select-dropdown:visible',
                        '.menu:visible'
                    ];
                    
                    return dropdownSelectors.some(sel => {
                        const elements = document.querySelectorAll(sel);
                        return Array.from(elements).some(el => {
                            const rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.height > 0;
                        });
                    });
                }
            """)
            
            return not dropdown_visible
            
        except Exception as e:
            self.logger.debug(f"Failed to check selection completion: {e}")
            return False
    
    def _calculate_element_confidence(self, element: Dict[str, Any]) -> float:
        """Calculate confidence score for an element's interactability."""
        score = 0.0
        
        # Base score for being detected as interactable
        score += 10.0
        
        # Boost for standard interactive elements
        tag = element.get('tagName', '')
        if tag in ['button', 'input', 'select', 'textarea', 'a']:
            score += 15.0
        
        # Boost for having meaningful text
        text = element.get('text', '').strip()
        if text:
            score += min(len(text) * 0.1, 10.0)
        
        # Boost for having context
        context = element.get('context', '').strip()
        if context:
            score += min(len(context) * 0.05, 5.0)
        
        # Boost for having ID or name attributes
        attrs = element.get('attributes', {})
        if attrs.get('id') or attrs.get('name'):
            score += 10.0
        
        # Check for specific attributes that indicate interactability
        if attrs.get('type') in ['submit', 'button', 'text', 'email', 'password']:
            score += 8.0
        
        if attrs.get('role') in ['button', 'link', 'textbox']:
            score += 6.0
        
        # Penalize for very large elements (might be containers)
        rect = element.get('rect', {})
        area = rect.get('width', 0) * rect.get('height', 0)
        if area > 100000:  # Very large elements
            score -= 5.0
        
        return max(score, 0.0)

    async def setup_console_monitoring(self):
        """Setup console monitoring for error context collection."""
        if self._page:
            # Store console messages for later retrieval
            await self._page.add_init_script("""
                window.console._messages = [];
                const originalLog = window.console.log;
                const originalError = window.console.error;
                const originalWarn = window.console.warn;
                
                window.console.log = function(...args) {
                    window.console._messages.push({
                        level: 'log',
                        text: args.join(' '),
                        timestamp: Date.now()
                    });
                    originalLog.apply(console, args);
                };
                
                window.console.error = function(...args) {
                    window.console._messages.push({
                        level: 'error',
                        text: args.join(' '),
                        timestamp: Date.now()
                    });
                    originalError.apply(console, args);
                };
                
                window.console.warn = function(...args) {
                    window.console._messages.push({
                        level: 'warn',
                        text: args.join(' '),
                        timestamp: Date.now()
                    });
                    originalWarn.apply(console, args);
                };
            """)

    async def handle_captcha_challenge(self, timeout: int = 30) -> Dict[str, Any]:
        """
        Handle CAPTCHA challenges intelligently.
        
        This method detects various types of CAPTCHAs and implements strategies
        to handle them, inspired by Skyvern's approach to dynamic challenges.
        """
        try:
            self.logger.info("Scanning for CAPTCHA challenges...")
            
            # Look for common CAPTCHA indicators
            captcha_elements = await self._detect_captcha_elements()
            
            if not captcha_elements:
                return {
                    "success": True,
                    "captcha_detected": False,
                    "message": "No CAPTCHA detected"
                }
            
            self.logger.info(f"Detected {len(captcha_elements)} CAPTCHA element(s)")
            
            for captcha_info in captcha_elements:
                captcha_type = captcha_info.get('type', 'unknown')
                
                if captcha_type == 'recaptcha':
                    result = await self._handle_recaptcha(captcha_info, timeout)
                elif captcha_type == 'hcaptcha':
                    result = await self._handle_hcaptcha(captcha_info, timeout)
                elif captcha_type == 'image_captcha':
                    result = await self._handle_image_captcha(captcha_info, timeout)
                elif captcha_type == 'text_captcha':
                    result = await self._handle_text_captcha(captcha_info, timeout)
                else:
                    result = await self._handle_generic_captcha(captcha_info, timeout)
                
                if result.get('success'):
                    return result
                else:
                    self.logger.warning(f"Failed to handle {captcha_type}: {result.get('error')}")
            
            return {
                "success": False,
                "captcha_detected": True,
                "error": "Could not solve any detected CAPTCHA challenges",
                "captcha_types": [info.get('type') for info in captcha_elements]
            }
            
        except Exception as e:
            self.logger.error(f"CAPTCHA handling failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _detect_captcha_elements(self) -> List[Dict[str, Any]]:
        """Detect CAPTCHA elements on the page."""
        try:
            captcha_elements = await self._page.evaluate("""
                () => {
                    const captchas = [];
                    
                    // Check for reCAPTCHA
                    const recaptcha = document.querySelector('.g-recaptcha, #recaptcha, iframe[src*="recaptcha"]');
                    if (recaptcha) {
                        captchas.push({
                            type: 'recaptcha',
                            element: recaptcha,
                            selector: '.g-recaptcha',
                            visible: recaptcha.offsetWidth > 0 && recaptcha.offsetHeight > 0
                        });
                    }
                    
                    // Check for hCaptcha
                    const hcaptcha = document.querySelector('.h-captcha, iframe[src*="hcaptcha"]');
                    if (hcaptcha) {
                        captchas.push({
                            type: 'hcaptcha',
                            element: hcaptcha,
                            selector: '.h-captcha',
                            visible: hcaptcha.offsetWidth > 0 && hcaptcha.offsetHeight > 0
                        });
                    }
                    
                    // Check for image CAPTCHAs
                    const imageCaptcha = document.querySelector('img[src*="captcha"], img[alt*="captcha"], .captcha-image');
                    if (imageCaptcha) {
                        captchas.push({
                            type: 'image_captcha',
                            element: imageCaptcha,
                            selector: 'img[src*="captcha"], img[alt*="captcha"], .captcha-image',
                            visible: imageCaptcha.offsetWidth > 0 && imageCaptcha.offsetHeight > 0
                        });
                    }
                    
                    // Check for text-based CAPTCHAs
                    const textElements = document.querySelectorAll('*');
                    for (const element of textElements) {
                        const text = (element.textContent || '').toLowerCase();
                        if (text.includes('enter the code') || 
                            text.includes('verification code') ||
                            text.includes('security code') ||
                            text.includes('captcha')) {
                            
                            // Look for nearby input field
                            const parent = element.closest('form, div, section');
                            const input = parent?.querySelector('input[type="text"], input[name*="captcha"], input[name*="code"]');
                            
                            if (input) {
                                captchas.push({
                                    type: 'text_captcha',
                                    element: input,
                                    selector: input.tagName + (input.name ? `[name="${input.name}"]` : ''),
                                    prompt: text,
                                    visible: input.offsetWidth > 0 && input.offsetHeight > 0
                                });
                                break;
                            }
                        }
                    }
                    
                    return captchas.filter(c => c.visible);
                }
            """)
            
            return captcha_elements or []
            
        except Exception as e:
            self.logger.error(f"Failed to detect CAPTCHA elements: {e}")
            return []

    async def _handle_recaptcha(self, captcha_info: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Handle reCAPTCHA challenges."""
        try:
            self.logger.info("Attempting to handle reCAPTCHA...")
            
            # Wait for reCAPTCHA to be fully loaded
            await asyncio.sleep(2)
            
            # Look for reCAPTCHA checkbox
            recaptcha_checkbox = '.recaptcha-checkbox-border'
            
            try:
                await self._page.wait_for_selector(recaptcha_checkbox, timeout=10000)
                await self._page.click(recaptcha_checkbox)
                
                # Wait to see if it's solved automatically (easy reCAPTCHA)
                await asyncio.sleep(3)
                
                # Check if reCAPTCHA is solved
                is_solved = await self._page.evaluate("""
                    () => {
                        const checkbox = document.querySelector('.recaptcha-checkbox');
                        return checkbox && checkbox.getAttribute('aria-checked') === 'true';
                    }
                """)
                
                if is_solved:
                    return {
                        "success": True,
                        "method": "recaptcha_checkbox",
                        "message": "reCAPTCHA solved automatically"
                    }
                
                # If challenge appears, we need user intervention
                challenge_frame = await self._page.wait_for_selector('iframe[src*="bframe"]', timeout=5000)
                if challenge_frame:
                    return {
                        "success": False,
                        "requires_manual_intervention": True,
                        "message": "reCAPTCHA challenge requires manual solving",
                        "action_needed": "Please solve the reCAPTCHA challenge manually"
                    }
                
            except Exception as e:
                self.logger.debug(f"reCAPTCHA checkbox not found or clickable: {e}")
                
            return {
                "success": False,
                "error": "Could not interact with reCAPTCHA"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"reCAPTCHA handling error: {e}"
            }

    async def _handle_hcaptcha(self, captcha_info: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Handle hCaptcha challenges."""
        try:
            self.logger.info("Attempting to handle hCaptcha...")
            
            # Similar approach to reCAPTCHA
            await asyncio.sleep(2)
            
            hcaptcha_checkbox = '.hcaptcha-checkbox'
            
            try:
                await self._page.wait_for_selector(hcaptcha_checkbox, timeout=10000)
                await self._page.click(hcaptcha_checkbox)
                
                await asyncio.sleep(3)
                
                # Check if solved
                is_solved = await self._page.evaluate("""
                    () => {
                        const checkbox = document.querySelector('[data-hcaptcha-response]');
                        return checkbox && checkbox.value && checkbox.value.length > 0;
                    }
                """)
                
                if is_solved:
                    return {
                        "success": True,
                        "method": "hcaptcha_checkbox", 
                        "message": "hCaptcha solved automatically"
                    }
                
                return {
                    "success": False,
                    "requires_manual_intervention": True,
                    "message": "hCaptcha challenge requires manual solving"
                }
                
            except Exception as e:
                self.logger.debug(f"hCaptcha interaction failed: {e}")
                
            return {
                "success": False,
                "error": "Could not interact with hCaptcha"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"hCaptcha handling error: {e}"
            }

    async def _handle_image_captcha(self, captcha_info: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Handle image-based CAPTCHAs."""
        try:
            self.logger.info("Detected image CAPTCHA - requires manual intervention")
            
            # Take screenshot of CAPTCHA for manual solving
            captcha_selector = captcha_info.get('selector', '')
            
            if captcha_selector:
                captcha_screenshot = await self._page.locator(captcha_selector).screenshot()
                captcha_b64 = base64.b64encode(captcha_screenshot).decode()
                
                return {
                    "success": False,
                    "requires_manual_intervention": True,
                    "captcha_type": "image",
                    "captcha_image": captcha_b64,
                    "message": "Image CAPTCHA detected - manual solving required",
                    "input_selector": self._find_captcha_input_field()
                }
            
            return {
                "success": False,
                "error": "Could not capture CAPTCHA image"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Image CAPTCHA handling error: {e}"
            }

    async def _handle_text_captcha(self, captcha_info: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Handle text-based CAPTCHAs."""
        try:
            self.logger.info("Detected text CAPTCHA - requires manual intervention")
            
            prompt = captcha_info.get('prompt', 'Enter verification code')
            input_selector = captcha_info.get('selector', '')
            
            return {
                "success": False,
                "requires_manual_intervention": True,
                "captcha_type": "text",
                "prompt": prompt,
                "input_selector": input_selector,
                "message": "Text CAPTCHA detected - manual input required"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Text CAPTCHA handling error: {e}"
            }

    async def _handle_generic_captcha(self, captcha_info: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Handle unknown CAPTCHA types."""
        return {
            "success": False,
            "requires_manual_intervention": True,
            "captcha_type": "unknown",
            "message": "Unknown CAPTCHA type detected - manual intervention required"
        }

    async def handle_verification_code(self, code_type: str = "email", timeout: int = 60) -> Dict[str, Any]:
        """
        Handle verification code challenges.
        
        This method manages various verification code scenarios commonly
        encountered during SSO setup processes.
        """
        try:
            self.logger.info(f"Handling {code_type} verification code challenge...")
            
            # Look for verification code input fields
            code_fields = await self._detect_verification_code_fields()
            
            if not code_fields:
                return {
                    "success": False,
                    "error": "No verification code input fields detected"
                }
            
            # Find the most likely verification code field
            best_field = self._select_best_verification_field(code_fields, code_type)
            
            if not best_field:
                return {
                    "success": False,
                    "error": "Could not identify verification code input field"
                }
            
            return {
                "success": False,
                "requires_manual_intervention": True,
                "verification_type": code_type,
                "input_selector": best_field.get('selector'),
                "input_description": best_field.get('description'),
                "message": f"Verification code input field ready - awaiting {code_type} code",
                "instructions": f"Please enter the {code_type} verification code in the detected field"
            }
            
        except Exception as e:
            self.logger.error(f"Verification code handling failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _detect_verification_code_fields(self) -> List[Dict[str, Any]]:
        """Detect verification code input fields."""
        try:
            fields = await self._page.evaluate("""
                () => {
                    const fields = [];
                    
                    // Look for inputs with verification-related attributes
                    const inputs = document.querySelectorAll('input[type="text"], input[type="number"], input[name*="code"], input[name*="verify"], input[id*="code"], input[id*="verify"]');
                    
                    for (const input of inputs) {
                        const name = input.name || '';
                        const id = input.id || '';
                        const placeholder = input.placeholder || '';
                        const label = input.labels?.[0]?.textContent || '';
                        
                        // Look for verification code indicators
                        const text = `${name} ${id} ${placeholder} ${label}`.toLowerCase();
                        
                        if (text.includes('code') || 
                            text.includes('verify') || 
                            text.includes('otp') ||
                            text.includes('2fa') ||
                            text.includes('authentication') ||
                            text.includes('token')) {
                            
                            fields.push({
                                selector: input.tagName.toLowerCase() + 
                                         (input.id ? `#${input.id}` : 
                                          input.name ? `[name="${input.name}"]` : 
                                          input.placeholder ? `[placeholder="${input.placeholder}"]` : ''),
                                name: name,
                                id: id,
                                placeholder: placeholder,
                                label: label,
                                description: text,
                                visible: input.offsetWidth > 0 && input.offsetHeight > 0
                            });
                        }
                    }
                    
                    return fields.filter(f => f.visible);
                }
            """)
            
            return fields or []
            
        except Exception as e:
            self.logger.error(f"Failed to detect verification code fields: {e}")
            return []

    def _select_best_verification_field(self, fields: List[Dict[str, Any]], code_type: str) -> Optional[Dict[str, Any]]:
        """Select the most appropriate verification code field."""
        if not fields:
            return None
        
        # Score fields based on relevance to code_type
        scored_fields = []
        
        for field in fields:
            score = 0
            description = field.get('description', '').lower()
            
            # Base score for being a verification field
            score += 10
            
            # Bonus for matching code type
            if code_type.lower() in description:
                score += 20
            
            # Bonus for common verification patterns
            if 'otp' in description:
                score += 15
            if 'authentication' in description:
                score += 12
            if '2fa' in description:
                score += 15
            if 'token' in description:
                score += 10
            
            # Bonus for having clear input characteristics
            if field.get('placeholder'):
                score += 5
            if field.get('label'):
                score += 5
            
            scored_fields.append((score, field))
        
        # Return the highest scoring field
        if scored_fields:
            scored_fields.sort(reverse=True, key=lambda x: x[0])
            return scored_fields[0][1]
        
        return fields[0]  # Fallback to first field

    async def _find_captcha_input_field(self) -> Optional[str]:
        """Find the input field associated with a CAPTCHA."""
        try:
            input_selector = await self._page.evaluate("""
                () => {
                    // Look for inputs near CAPTCHA elements
                    const captchaKeywords = ['captcha', 'code', 'verify', 'security'];
                    const inputs = document.querySelectorAll('input[type="text"]');
                    
                    for (const input of inputs) {
                        const name = (input.name || '').toLowerCase();
                        const id = (input.id || '').toLowerCase();
                        const placeholder = (input.placeholder || '').toLowerCase();
                        
                        if (captchaKeywords.some(keyword => 
                            name.includes(keyword) || 
                            id.includes(keyword) || 
                            placeholder.includes(keyword))) {
                            
                            return input.tagName.toLowerCase() + 
                                   (input.id ? `#${input.id}` : 
                                    input.name ? `[name="${input.name}"]` : '');
                        }
                    }
                    
                    return null;
                }
            """)
            
            return input_selector
            
        except Exception:
            return None

    async def submit_verification_code(self, code: str, field_selector: Optional[str] = None) -> Dict[str, Any]:
        """Submit a verification code to the detected field."""
        try:
            if field_selector:
                selector = field_selector
            else:
                # Try to find verification field automatically
                fields = await self._detect_verification_code_fields()
                if not fields:
                    return {
                        "success": False,
                        "error": "No verification code field found"
                    }
                selector = fields[0].get('selector')
            
            # Clear and type the code
            await self._page.locator(selector).clear()
            await self._page.locator(selector).type(code)
            
            # Try to submit automatically
            # Look for nearby submit button
            submit_result = await self._page.evaluate(f"""
                (selector) => {{
                    const field = document.querySelector(selector);
                    if (!field) return false;
                    
                    // Try to find submit button in the same form
                    const form = field.closest('form');
                    if (form) {{
                        const submitBtn = form.querySelector('[type="submit"], button[type="submit"], .submit-btn, .verify-btn');
                        if (submitBtn) {{
                            submitBtn.click();
                            return true;
                        }}
                    }}
                    
                    // Try pressing Enter
                    field.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', code: 'Enter'}}));
                    return true;
                }}
            """, selector)
            
            return {
                "success": True,
                "code_submitted": code,
                "method": "auto_submit" if submit_result else "manual_submit_needed",
                "message": "Verification code entered successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to submit verification code: {e}"
            }

    async def execute_cached_integration(self, goal: str, current_url: str) -> Dict[str, Any]:
        """
        Try to execute a cached action sequence for the given goal and URL.
        
        This method leverages the action caching system to speed up repeated
        vendor integrations by reusing proven patterns.
        """
        try:
            # Look for a cached sequence that matches our current context
            cached_sequence = await self.action_cache.find_cached_sequence(
                goal=goal,
                url=current_url,
                similarity_threshold=0.7
            )
            
            if not cached_sequence:
                return {
                    "success": False,
                    "cache_hit": False,
                    "message": "No suitable cached sequence found for this goal and URL"
                }
            
            self.logger.info(f"Found cached sequence for goal: {goal}")
            
            # Execute the cached sequence with adaptation
            result = await self.action_cache.execute_cached_sequence(
                sequence=cached_sequence,
                web_interactor=self,
                adaptation_mode=True
            )
            
            result["cache_hit"] = True
            result["cached_goal"] = cached_sequence.goal_description
            result["sequence_success_rate"] = cached_sequence.success_rate
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute cached integration: {e}")
            return {
                "success": False,
                "cache_hit": False,
                "error": str(e)
            }

    async def cache_successful_integration(
        self, 
        goal: str, 
        url: str, 
        actions: List[Dict[str, Any]], 
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Cache a successful integration sequence for future reuse.
        
        This should be called after completing a successful vendor integration
        to store the action sequence for similar future integrations.
        """
        try:
            sequence_id = await self.action_cache.cache_action_sequence(
                goal=goal,
                url=url,
                actions=actions,
                success=True,
                tags=tags
            )
            
            self.logger.info(f"Cached successful integration sequence: {sequence_id}")
            return sequence_id
            
        except Exception as e:
            self.logger.error(f"Failed to cache integration sequence: {e}")
            return ""