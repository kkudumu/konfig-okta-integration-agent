"""
WebInteractor Tool for Konfig.

This tool provides browser automation capabilities using Playwright,
enabling the agent to interact with web-based user interfaces during
SSO integration processes.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from konfig.config.settings import get_settings
from konfig.utils.logging import LoggingMixin


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
        Click on an element.
        
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
            
            # Find element with optional text content matching
            if text_content:
                # Use XPath to find element with specific text
                xpath_selector = f"//*[contains(text(), '{text_content}')]"
                element = self._page.locator(xpath_selector)
            else:
                element = self._page.locator(selector)
            
            # Wait for element to be visible and enabled
            await element.wait_for(state="visible", timeout=timeout or self.timeout)
            
            # Perform click
            await element.click(timeout=timeout or self.timeout)
            
            end_time = asyncio.get_event_loop().time()
            duration_ms = (end_time - start_time) * 1000
            
            result = {
                "success": True,
                "selector": selector,
                "text_content": text_content,
                "duration_ms": round(duration_ms, 2),
            }
            
            self.log_method_result("click", result, duration_ms)
            return result
            
        except Exception as e:
            self.log_error("click", e, selector=selector, text_content=text_content)
            raise ElementNotFoundError(f"Failed to click element '{selector}': {e}")
    
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