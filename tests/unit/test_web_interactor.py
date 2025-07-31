"""
Unit tests for the WebInteractor tool.

Tests browser automation capabilities and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from konfig.tools.web_interactor import WebInteractor, WebInteractorError, ElementNotFoundError


@pytest.mark.unit
@pytest.mark.requires_browser
class TestWebInteractor:
    """Test cases for WebInteractor functionality."""
    
    @pytest.fixture
    async def interactor(self):
        """Create WebInteractor instance for testing."""
        interactor = WebInteractor()
        interactor.headless = True  # Ensure headless for tests
        yield interactor
        
        # Cleanup
        if interactor._session_active:
            await interactor.close_browser()
    
    async def test_browser_startup_and_shutdown(self, interactor):
        """Test browser session lifecycle."""
        # Initially not active
        assert not interactor._session_active
        
        # Start browser
        await interactor.start_browser()
        assert interactor._session_active
        assert interactor._browser is not None
        assert interactor._page is not None
        
        # Close browser
        await interactor.close_browser()
        assert not interactor._session_active
        assert interactor._browser is None
        assert interactor._page is None
    
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with WebInteractor() as interactor:
            assert interactor._session_active
            assert interactor._browser is not None
            assert interactor._page is not None
        
        # Should be cleaned up automatically
        assert not interactor._session_active
    
    async def test_navigate_success(self, interactor):
        """Test successful navigation."""
        await interactor.start_browser()
        
        # Mock a simple page navigation
        with patch.object(interactor._page, 'goto') as mock_goto:
            mock_response = Mock()
            mock_response.status = 200
            mock_goto.return_value = mock_response
            
            with patch.object(interactor._page, 'url', 'https://example.com'):
                with patch.object(interactor._page, 'title', return_value=AsyncMock(return_value="Test Page")):
                    result = await interactor.navigate("https://example.com")
        
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["final_url"] == "https://example.com"
        assert result["status_code"] == 200
        assert "duration_ms" in result
    
    async def test_navigate_failure(self, interactor):
        """Test navigation failure handling."""
        await interactor.start_browser()
        
        # Mock navigation failure
        with patch.object(interactor._page, 'goto') as mock_goto:
            mock_goto.side_effect = Exception("Navigation failed")
            
            with pytest.raises(Exception):
                await interactor.navigate("https://invalid-url.com")
    
    async def test_click_element(self, interactor):
        """Test clicking an element."""
        await interactor.start_browser()
        
        # Mock successful click
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.click = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.click("#test-button")
        
        assert result["success"] is True
        assert result["selector"] == "#test-button"
        assert "duration_ms" in result
        
        # Verify the mock was called correctly
        mock_locator.wait_for.assert_called_once()
        mock_locator.click.assert_called_once()
    
    async def test_click_with_text_content(self, interactor):
        """Test clicking an element with text content matching."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.click = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.click("#test-button", text_content="Submit")
        
        assert result["success"] is True
        assert result["text_content"] == "Submit"
    
    async def test_click_element_not_found(self, interactor):
        """Test clicking non-existent element."""
        await interactor.start_browser()
        
        # Mock element not found
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock(side_effect=Exception("Element not found"))
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            with pytest.raises(ElementNotFoundError):
                await interactor.click("#non-existent")
    
    async def test_type_text(self, interactor):
        """Test typing text in an input field."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.clear = AsyncMock()
        mock_locator.fill = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.type("#username", "testuser")
        
        assert result["success"] is True
        assert result["selector"] == "#username"
        assert result["text_length"] == 8
        assert result["secret"] is False
        
        mock_locator.clear.assert_called_once()
        mock_locator.fill.assert_called_once_with("testuser", timeout=interactor.timeout)
    
    async def test_type_secret_text(self, interactor):
        """Test typing secret text (should not be logged)."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.clear = AsyncMock()
        mock_locator.fill = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.type("#password", "secret123", secret=True)
        
        assert result["success"] is True
        assert result["secret"] is True
        assert result["text_length"] == 9
    
    async def test_type_without_clear(self, interactor):
        """Test typing without clearing existing text."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.fill = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.type("#field", "text", clear_first=False)
        
        assert result["success"] is True
        # Should not call clear when clear_first=False
        assert not hasattr(mock_locator, 'clear') or not mock_locator.clear.called
    
    async def test_select_option(self, interactor):
        """Test selecting an option from dropdown."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.select_option = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.select_option("#country", "US")
        
        assert result["success"] is True
        assert result["selector"] == "#country"
        assert result["value"] == "US"
        
        mock_locator.select_option.assert_called_once_with("US", timeout=interactor.timeout)
    
    async def test_get_element_text(self, interactor):
        """Test getting element text content."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.text_content = AsyncMock(return_value="Sample text")
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            text = await interactor.get_element_text("#message")
        
        assert text == "Sample text"
    
    async def test_get_element_value(self, interactor):
        """Test getting input element value."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.input_value = AsyncMock(return_value="current_value")
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            value = await interactor.get_element_value("#input-field")
        
        assert value == "current_value"
    
    async def test_get_element_attribute(self, interactor):
        """Test getting element attribute."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.get_attribute = AsyncMock(return_value="button")
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            attr_value = await interactor.get_element_attribute("#submit", "type")
        
        assert attr_value == "button"
    
    async def test_element_exists_true(self, interactor):
        """Test checking if element exists (positive case)."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()  # No exception = element exists
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            exists = await interactor.element_exists("#existing-element")
        
        assert exists is True
    
    async def test_element_exists_false(self, interactor):
        """Test checking if element exists (negative case)."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock(side_effect=Exception("Timeout"))
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            exists = await interactor.element_exists("#non-existent", timeout=100)
        
        assert exists is False
    
    async def test_get_current_dom(self, interactor):
        """Test getting simplified DOM representation."""
        await interactor.start_browser()
        
        mock_dom_result = '{"tag": "body", "children": [{"tag": "div", "text": "Test"}]}'
        
        with patch.object(interactor._page, 'evaluate', return_value=mock_dom_result):
            dom = await interactor.get_current_dom(simplify=True)
        
        assert isinstance(dom, str)
        assert "body" in dom or "div" in dom
    
    async def test_take_screenshot(self, interactor, temp_screenshot_dir):
        """Test taking a screenshot."""
        await interactor.start_browser()
        
        screenshot_path = str(temp_screenshot_dir / "test.png")
        
        with patch.object(interactor._page, 'screenshot') as mock_screenshot:
            result_path = await interactor.take_screenshot(screenshot_path)
        
        assert result_path == screenshot_path
        mock_screenshot.assert_called_once_with(path=screenshot_path, full_page=False)
    
    async def test_wait_for_element_success(self, interactor):
        """Test waiting for element successfully."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.wait_for_element("#element", timeout=5000)
        
        assert result is True
        mock_locator.wait_for.assert_called_once_with(state="visible", timeout=5000)
    
    async def test_wait_for_element_timeout(self, interactor):
        """Test waiting for element with timeout."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock(side_effect=Exception("Timeout"))
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.wait_for_element("#element", timeout=1000)
        
        assert result is False
    
    async def test_fill_form(self, interactor):
        """Test filling a form with multiple fields."""
        await interactor.start_browser()
        
        # Mock successful form filling
        mock_locator = Mock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.fill = AsyncMock()
        mock_locator.select_option = AsyncMock()
        mock_locator.click = AsyncMock()
        
        form_data = {
            "#username": "testuser",
            "#email": "test@example.com",
            "#country": {"type": "select", "value": "US"},
            "#submit": {"type": "click"}
        }
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            # Mock the type method to return success results
            with patch.object(interactor, 'type', return_value={"success": True}):
                with patch.object(interactor, 'select_option', return_value={"success": True}):
                    with patch.object(interactor, 'click', return_value={"success": True}):
                        result = await interactor.fill_form(form_data)
        
        assert result["success"] is True
        assert result["total_fields"] == 4
        assert result["successful_fields"] == 4
    
    async def test_fill_form_partial_failure(self, interactor):
        """Test form filling with some field failures."""
        await interactor.start_browser()
        
        form_data = {
            "#username": "testuser",
            "#invalid-field": "value"
        }
        
        # Mock one success, one failure
        async def mock_type(selector, text, **kwargs):
            if selector == "#username":
                return {"success": True}
            else:
                raise ElementNotFoundError("Element not found")
        
        with patch.object(interactor, 'type', side_effect=mock_type):
            result = await interactor.fill_form(form_data)
        
        assert result["success"] is False
        assert result["total_fields"] == 2
        assert result["successful_fields"] == 1
        assert "#username" in result["results"]
        assert "#invalid-field" in result["results"]
        assert result["results"]["#username"]["success"] is True
        assert result["results"]["#invalid-field"]["success"] is False
    
    def test_browser_not_active_error(self, interactor):
        """Test operations without active browser session."""
        # Try to perform action without starting browser
        with pytest.raises(WebInteractorError, match="Browser session not active"):
            asyncio.run(interactor.click("#button"))
    
    async def test_find_elements_by_text(self, interactor):
        """Test finding elements by text content."""
        await interactor.start_browser()
        
        mock_elements = [
            {
                "tag": "button",
                "text": "Submit Form",
                "id": "submit-btn",
                "className": "btn primary",
                "selector": "#submit-btn"
            },
            {
                "tag": "a", 
                "text": "Submit Report",
                "id": "",
                "className": "link",
                "selector": ".link"
            }
        ]
        
        with patch.object(interactor._page, 'evaluate', return_value=mock_elements):
            elements = await interactor.find_elements_by_text("Submit")
        
        assert len(elements) == 2
        assert elements[0]["text"] == "Submit Form"
        assert elements[1]["text"] == "Submit Report"
    
    async def test_scroll_to_element(self, interactor):
        """Test scrolling to element."""
        await interactor.start_browser()
        
        mock_locator = Mock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        
        with patch.object(interactor._page, 'locator', return_value=mock_locator):
            result = await interactor.scroll_to_element("#bottom-element")
        
        assert result is True
        mock_locator.scroll_into_view_if_needed.assert_called_once()
    
    async def test_wait_for_navigation(self, interactor):
        """Test waiting for navigation to complete."""
        await interactor.start_browser()
        
        with patch.object(interactor._page, 'wait_for_load_state') as mock_wait:
            with patch.object(interactor._page, 'url', 'https://new-page.com'):
                with patch.object(interactor._page, 'title', return_value=AsyncMock(return_value="New Page")):
                    result = await interactor.wait_for_navigation()
        
        assert result["success"] is True
        assert result["url"] == "https://new-page.com"
        assert result["title"] == "New Page"
        mock_wait.assert_called_once_with("networkidle", timeout=interactor.timeout)