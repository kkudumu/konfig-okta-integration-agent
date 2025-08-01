"""
Intelligent Web Automation Service

Uses LLM reasoning to perform complex web automation tasks by analyzing
page content and determining the appropriate actions to take.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from konfig.utils.logging import LoggingMixin
from konfig.services.adaptive_intelligence import AdaptiveIntelligence


@dataclass
class WebAction:
    """Represents a web automation action."""
    action_type: str  # click, type, select, navigate, etc.
    selector: Optional[str] = None
    value: Optional[str] = None
    description: str = ""
    verification: Optional[str] = None  # How to verify success


class IntelligentWebAutomation(LoggingMixin):
    """
    Service that uses LLM reasoning to perform intelligent web automation.
    Instead of hard-coded selectors and actions, it analyzes page content
    and determines what actions to take dynamically.
    """
    
    def __init__(self, web_interactor):
        super().__init__()
        self.setup_logging("intelligent_web_automation")
        self.web_interactor = web_interactor
        
        # Initialize adaptive intelligence
        try:
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            self.adaptive_intelligence = AdaptiveIntelligence(web_interactor, llm_service)
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM service for adaptive intelligence: {e}")
            self.adaptive_intelligence = AdaptiveIntelligence(web_interactor, None)
    
    async def navigate_and_configure_saml(
        self,
        admin_url: str,
        sso_url: str,
        entity_id: str,
        certificate: Optional[str] = None,
        vendor_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Navigate to vendor admin console and configure SAML using LLM guidance.
        
        Args:
            admin_url: URL of the vendor's admin console
            sso_url: Okta SSO URL to configure
            entity_id: SAML Entity ID  
            certificate: SAML signing certificate (optional)
            vendor_name: Name of the vendor for context
            
        Returns:
            Result of the configuration process
        """
        self.logger.info(f"Starting intelligent SAML configuration for {vendor_name}")
        
        try:
            # Step 1: Navigate to admin console
            await self.web_interactor.navigate(admin_url)
            
            # Step 1.5: Handle authentication if needed (Google Workspace)
            if "google.com" in admin_url.lower() and vendor_name.lower() == "google workspace":
                from konfig.services.google_auth_service import GoogleAuthService
                auth_service = GoogleAuthService()
                auth_result = await auth_service.authenticate_to_admin_console(self.web_interactor, admin_url)
                
                if not auth_result["success"]:
                    return {
                        "status": "error",
                        "message": f"Authentication failed: {auth_result['message']}"
                    }
                
                self.logger.info("Successfully authenticated to Google Admin Console")
            
            # Step 2: Analyze page and find SSO/SAML settings
            sso_page_url = await self._find_sso_settings_page(vendor_name)
            
            if sso_page_url:
                await self.web_interactor.navigate(sso_page_url)
            
            # Step 3: Configure SAML settings using LLM guidance
            result = await self._configure_saml_settings(
                sso_url=sso_url,
                entity_id=entity_id,
                certificate=certificate,
                vendor_name=vendor_name
            )
            
            # Step 4: Test the configuration if possible
            test_result = await self._test_sso_configuration(vendor_name)
            result["test_result"] = test_result
            
            return {
                "status": "success",
                "message": f"SAML configuration completed for {vendor_name}",
                "details": result
            }
            
        except Exception as e:
            self.logger.error(f"SAML configuration failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to configure SAML for {vendor_name}: {str(e)}"
            }
    
    async def _find_sso_settings_page(self, vendor_name: str) -> Optional[str]:
        """Intelligently scan page to find SSO settings using multiple strategies."""
        
        try:
            # Strategy 1: Try direct navigation to known Google SSO paths
            known_google_sso_paths = [
                "https://admin.google.com/ac/security/sso",
                "https://admin.google.com/ac/security",
                "https://admin.google.com/ac/apps/unified"
            ]
            
            current_url = await self.web_interactor.get_current_url()
            self.logger.info(f"Current URL: {current_url}")
            
            # Strategy 2: Scan current page for navigation elements
            page_scan_result = await self._scan_page_for_navigation()
            
            if page_scan_result.get("security_found"):
                # Click on Security section
                security_clicked = await self._smart_click_by_text("Security")
                if security_clicked:
                    await self.web_interactor._page.wait_for_timeout(2000)  # Wait for navigation
                    
                    # Look for SSO options in Security section
                    sso_result = await self._scan_page_for_sso_options()
                    if sso_result.get("sso_found"):
                        await self._smart_click_by_text(sso_result["sso_text"])
                        await self.web_interactor._page.wait_for_timeout(2000)
                        return await self.web_interactor.get_current_url()
            
            # Strategy 3: Try direct path navigation if page scanning failed
            for path in known_google_sso_paths:
                try:
                    self.logger.info(f"Trying direct navigation to: {path}")
                    result = await self.web_interactor.navigate(path)
                    if result.get("success"):
                        await self.web_interactor._page.wait_for_timeout(3000)
                        
                        # Check if we reached a valid page
                        final_url = await self.web_interactor.get_current_url()
                        page_title = await self.web_interactor.get_page_title()
                        
                        if "sso" in final_url.lower() or "sso" in page_title.lower():
                            self.logger.info(f"Successfully navigated to SSO page: {final_url}")
                            return final_url
                        
                except Exception as e:
                    self.logger.warning(f"Failed to navigate to {path}: {e}")
                    continue
            
            self.logger.warning("Could not find SSO settings page using any strategy")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find SSO settings: {e}")
            return None
    
    async def _configure_saml_settings(
        self,
        sso_url: str,
        entity_id: str,
        certificate: Optional[str],
        vendor_name: str
    ) -> Dict[str, Any]:
        """Use LLM to configure SAML settings on the current page."""
        
        page_content = await self.web_interactor.get_current_dom(simplify=True)
        current_url = await self.web_interactor.get_current_url()
        
        # Check if we're on the SSO profiles list page (Google Workspace specific)
        if "google" in vendor_name.lower() and "/security/sso" in current_url:
            self.logger.info("Detected Google SSO profiles list page, using adaptive intelligence to navigate")
            
            # Use adaptive intelligence to click on Legacy SSO Profile
            navigation_action = {
                "name": "Click Legacy SSO Profile",
                "tool": "WebInteractor", 
                "action": "click",
                "params": {
                    "selector": "tr:has-text('Legacy SSO Profile')",
                    "text_content": "Legacy SSO Profile"
                }
            }
            
            nav_result = await self.adaptive_intelligence.execute_with_adaptive_intelligence(
                navigation_action,
                {"working_memory": {}, "vendor_name": vendor_name}
            )
            
            if nav_result.get("success"):
                self.logger.info("Successfully navigated to Legacy SSO Profile configuration")
                # Wait for page load and get updated content
                await self.web_interactor._page.wait_for_timeout(3000)
                page_content = await self.web_interactor.get_current_dom(simplify=True)
            else:
                self.logger.warning(f"Failed to navigate to Legacy SSO Profile: {nav_result.get('error')}")
                # Continue with original page content for LLM analysis
        
        config_prompt = f"""
You are configuring SAML SSO settings for {vendor_name}. 

CURRENT PAGE CONTENT:
{page_content[:3000]}...

CURRENT URL: {current_url}

SAML VALUES TO CONFIGURE:
- SSO URL (Sign-in page URL): {sso_url}
- Sign-out page URL: {sso_url.replace('/sso/saml', '/logout')}  # Typically the logout URL
- Entity ID: {entity_id}
- Certificate: {"Provided" if certificate else "Not provided"}

For Google Workspace, we need to:
1. If we see a list of SSO profiles, click on "Legacy SSO Profile" first
2. Then fill in the form fields:
   - Sign-in page URL
   - Sign-out page URL  
   - Verification certificate
   - Check "Use a domain-specific issuer"
   - Change password URL

Analyze the page and respond with JSON:
{{
    "needs_navigation": true/false,
    "navigation_action": {{
        "action": "click",
        "selector": "CSS selector or text",
        "description": "what to click"
    }},
    "fields_found": [
        {{
            "field_type": "sign_in_url|sign_out_url|certificate|domain_issuer|password_url",
            "selector": "CSS selector or label text",
            "current_value": "current field value if visible",
            "action": "type|select|click|check",
            "new_value": "value to set"
        }}
    ],
    "submit_button": {{
        "selector": "CSS selector for SAVE button",
        "text": "SAVE"
    }},
    "confidence": 0.0-1.0
}}

Be precise with selectors - use ID, name, label text, or unique attributes when possible.
"""

        try:
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=config_prompt,
                max_tokens=1500,
                temperature=0.1
            )
            
            # Use robust JSON extraction
            from konfig.services.intelligent_planner import IntelligentPlanner
            planner = IntelligentPlanner()
            config = planner._extract_json_from_response(response)
            
            # Execute the configuration actions using adaptive intelligence
            configured_fields = []
            
            for field in config.get("fields_found", []):
                field_action = self._create_field_action(field, sso_url, entity_id, certificate)
                
                if field_action:
                    result = await self.adaptive_intelligence.execute_with_adaptive_intelligence(
                        field_action,
                        {"working_memory": {"sso_url": sso_url, "entity_id": entity_id}}
                    )
                    
                    configured_fields.append({
                        "field_type": field["field_type"],
                        "success": result.get("success", False),
                        "error": result.get("error") if not result.get("success") else None
                    })
            
            # Submit the form if submit button found
            if config.get("submit_button"):
                submit_action = {
                    "name": "Submit SAML Configuration",
                    "tool": "WebInteractor",
                    "action": "click", 
                    "params": {
                        "selector": config["submit_button"]["selector"],
                        "text_content": config["submit_button"].get("text")
                    }
                }
                
                submit_result = await self.adaptive_intelligence.execute_with_adaptive_intelligence(
                    submit_action,
                    {"working_memory": {}}
                )
                
                configured_fields.append({
                    "field_type": "submit",
                    "success": submit_result.get("success", False),
                    "error": submit_result.get("error") if not submit_result.get("success") else None
                })
            
            return {
                "configured_fields": configured_fields,
                "total_fields": len(config.get("fields_found", [])),
                "confidence": config.get("confidence", 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to configure SAML settings: {e}")
            return {"error": str(e)}
    
    def _create_field_action(
        self,
        field_config: Dict[str, Any],
        sso_url: str,
        entity_id: str,
        certificate: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Create an action for configuring a specific field."""
        
        field_type = field_config.get("field_type")
        selector = field_config.get("selector")
        action = field_config.get("action")
        
        # Determine the value to use
        value = None
        if field_type in ["sso_url", "sign_in_url"]:
            value = sso_url
        elif field_type == "sign_out_url":
            value = sso_url.replace('/sso/saml', '/logout') if sso_url else None
        elif field_type == "entity_id":
            value = entity_id
        elif field_type == "certificate" and certificate:
            value = certificate
        elif field_type == "password_url":
            value = sso_url.replace('/sso/saml', '/password') if sso_url else None
        
        if not selector:
            return None
        
        # Create appropriate action
        if action == "type" and value:
            return {
                "name": f"Fill {field_type} field",
                "tool": "WebInteractor",
                "action": "type",
                "params": {
                    "selector": selector,
                    "text": value,
                    "clear_first": True
                }
            }
        elif action in ["click", "check"]:
            return {
                "name": f"Click {field_type} field",
                "tool": "WebInteractor", 
                "action": "click",
                "params": {
                    "selector": selector
                }
            }
        elif action == "select" and value:
            return {
                "name": f"Select {field_type} option",
                "tool": "WebInteractor",
                "action": "select_option", 
                "params": {
                    "selector": selector,
                    "value": value
                }
            }
        
        return None
    
    async def _configure_field(
        self,
        field_config: Dict[str, Any],
        sso_url: str,
        entity_id: str,
        certificate: Optional[str]
    ) -> bool:
        """Configure a specific field based on its type (legacy method - kept for compatibility)."""
        
        try:
            field_type = field_config["field_type"]
            selector = field_config["selector"]
            action = field_config["action"]
            
            # Determine the value to use
            if field_type == "sso_url":
                value = sso_url
            elif field_type == "entity_id":
                value = entity_id
            elif field_type == "certificate" and certificate:
                value = certificate
            elif field_type == "enable_toggle":
                value = None  # Just click to enable
            else:
                return False
            
            # Perform the action
            if action == "type" and value:
                await self.web_interactor.type(selector, value)
                return True
            elif action == "click":
                return await self._smart_click(selector)
            elif action == "select" and value:
                await self.web_interactor.select_option(selector, value)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to configure field {field_config.get('field_type')}: {e}")
            return False
    
    async def _smart_click(self, selector_or_text: str) -> bool:
        """Intelligently click an element by selector or text."""
        try:
            # Try direct selector first
            if await self.web_interactor.element_exists(selector_or_text):
                await self.web_interactor.click(selector_or_text)
                return True
            
            # Try finding by text
            elements = await self.web_interactor.find_elements_by_text(selector_or_text)
            if elements:
                # Click the most likely element (first one for now)
                element = elements[0]
                if "selector" in element:
                    await self.web_interactor.click(element["selector"])
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _smart_click_by_text(self, text: str) -> bool:
        """Click element containing specific text using robust search."""
        try:
            # Strategy 1: Use Playwright's text search
            try:
                await self.web_interactor._page.click(f"text={text}", timeout=5000)
                return True
            except Exception:
                pass
            
            # Strategy 2: Use XPath to find text
            try:
                xpath = f"//*[contains(text(), '{text}')]"
                await self.web_interactor._page.click(f"xpath={xpath}", timeout=5000)
                return True
            except Exception:
                pass
            
            # Strategy 3: Search in various clickable elements
            clickable_selectors = [
                f"button:has-text('{text}')",
                f"a:has-text('{text}')",
                f"div[role='button']:has-text('{text}')",
                f"span:has-text('{text}')",
                f"li:has-text('{text}')"
            ]
            
            for selector in clickable_selectors:
                try:
                    await self.web_interactor._page.click(selector, timeout=2000)
                    return True
                except Exception:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to click element with text '{text}': {e}")
            return False
    
    async def _scan_page_for_navigation(self) -> Dict[str, Any]:
        """Scan current page for navigation elements."""
        try:
            # Get page content and look for navigation elements
            dom_content = await self.web_interactor.get_current_dom(simplify=True)
            
            # Look for common navigation keywords
            navigation_keywords = ["Security", "Apps", "Authentication", "Directory", "Admin"]
            found_navigation = {}
            
            for keyword in navigation_keywords:
                # Check if keyword exists in page content
                if keyword.lower() in dom_content.lower():
                    found_navigation[keyword.lower() + "_found"] = True
                    self.logger.info(f"Found '{keyword}' in page navigation")
                else:
                    found_navigation[keyword.lower() + "_found"] = False
            
            return found_navigation
            
        except Exception as e:
            self.logger.error(f"Failed to scan page for navigation: {e}")
            return {}
    
    async def _scan_page_for_sso_options(self) -> Dict[str, Any]:
        """Scan current page for SSO-related options."""
        try:
            # Look for SSO-related text on the page
            sso_keywords = [
                "Single sign-on",
                "SSO",
                "SAML",
                "Set up SSO",
                "Third party SSO",
                "Identity provider",
                "Configure SSO"
            ]
            
            dom_content = await self.web_interactor.get_current_dom(simplify=True)
            
            for keyword in sso_keywords:
                if keyword.lower() in dom_content.lower():
                    self.logger.info(f"Found SSO keyword: {keyword}")
                    return {
                        "sso_found": True,
                        "sso_text": keyword
                    }
            
            return {"sso_found": False}
            
        except Exception as e:
            self.logger.error(f"Failed to scan page for SSO options: {e}")
            return {"sso_found": False}
    
    async def _test_sso_configuration(self, vendor_name: str) -> Dict[str, Any]:
        """Test the SSO configuration if possible."""
        
        try:
            page_content = await self.web_interactor.get_current_dom(simplify=True)
            
            # Look for test buttons or validation indicators
            test_prompt = f"""
Look at this {vendor_name} admin page for SSO/SAML testing options:

{page_content[:1500]}...

Look for:
1. "Test SSO", "Test SAML", "Validate", "Test Connection" buttons
2. Status indicators showing if SSO is enabled/working
3. Error messages or validation warnings

Respond with JSON:
{{
    "test_available": true/false,
    "test_button_selector": "CSS selector if found",
    "status_indicators": ["list of status messages found"],
    "errors_found": ["list of error messages if any"]
}}
"""
            
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=test_prompt,
                max_tokens=800,
                temperature=0.1
            )
            
            # Use robust JSON extraction
            from konfig.services.intelligent_planner import IntelligentPlanner
            planner = IntelligentPlanner()
            test_info = planner._extract_json_from_response(response)
            
            # If test button available, click it
            if test_info.get("test_available") and test_info.get("test_button_selector"):
                test_success = await self._smart_click(test_info["test_button_selector"])
                test_info["test_executed"] = test_success
                
                if test_success:
                    # Wait a moment for results
                    await self.web_interactor._page.wait_for_timeout(3000)
                    
                    # Check for test results
                    updated_content = await self.web_interactor.get_current_dom(simplify=True)
                    test_info["test_results"] = "Test executed - check page for results"
            
            return test_info
            
        except Exception as e:
            self.logger.error(f"Failed to test SSO configuration: {e}")
            return {"error": str(e)}