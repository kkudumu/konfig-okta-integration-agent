"""
Intelligent Web Automation Service

Uses LLM reasoning to perform complex web automation tasks by analyzing
page content and determining the appropriate actions to take.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from konfig.utils.logging import LoggingMixin


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
        """Use LLM to analyze page and find SSO settings."""
        
        # Get current page content
        page_content = await self.web_interactor.get_current_dom(simplify=True)
        current_url = await self.web_interactor.get_current_url()
        
        # Analyze page with LLM
        analysis_prompt = f"""
You are analyzing the admin console for {vendor_name} to find SSO/SAML configuration settings.

CURRENT URL: {current_url}

PAGE CONTENT:
{page_content[:2000]}...

Look for:
1. Links or buttons related to SSO, SAML, Single Sign-On, Authentication, Security, Identity, or Login settings
2. Navigation menus that might contain these options
3. Settings or configuration areas

Respond with JSON:
{{
    "sso_found": true/false,
    "sso_url": "full URL if found or null",
    "navigation_needed": true/false,
    "actions_required": [
        {{
            "action": "click|navigate",
            "selector": "CSS selector or text to find",
            "description": "What this action does"
        }}
    ],
    "confidence": 0.0-1.0
}}

Focus on finding the most direct path to SAML/SSO configuration.
"""

        try:
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=analysis_prompt,
                max_tokens=1000,
                temperature=0.1
            )
            
            analysis = json.loads(response.strip())
            
            if analysis.get("sso_found") and analysis.get("sso_url"):
                self.logger.info(f"Found SSO settings page: {analysis['sso_url']}")
                return analysis["sso_url"]
            
            elif analysis.get("navigation_needed") and analysis.get("actions_required"):
                # Perform navigation actions
                for action in analysis["actions_required"]:
                    if action["action"] == "click":
                        success = await self._smart_click(action["selector"])
                        if success:
                            # Check if we're now on the SSO page
                            new_url = await self.web_interactor.get_current_url()
                            if "sso" in new_url.lower() or "saml" in new_url.lower():
                                return new_url
            
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
        
        config_prompt = f"""
You are configuring SAML SSO settings for {vendor_name}. 

CURRENT PAGE CONTENT:
{page_content[:3000]}...

SAML VALUES TO CONFIGURE:
- SSO URL: {sso_url}
- Entity ID: {entity_id}
- Certificate: {"Provided" if certificate else "Not provided"}

Analyze the page and identify form fields that need to be filled with these SAML values.
Look for fields like:
- SSO URL, Sign-on URL, Login URL, SAML Endpoint
- Entity ID, Issuer, IdP Entity ID
- Certificate, X.509 Certificate, Signing Certificate
- Any enable/activate SSO buttons or toggles

Respond with JSON:
{{
    "fields_found": [
        {{
            "field_type": "sso_url|entity_id|certificate|enable_toggle",
            "selector": "CSS selector or text to identify field",
            "current_value": "current field value if visible",
            "action": "type|select|click|toggle",
            "new_value": "value to set"
        }}
    ],
    "submit_button": {{
        "selector": "CSS selector for save/submit button",
        "text": "button text"
    }},
    "confidence": 0.0-1.0
}}

Be precise with selectors - use ID, name, or unique attributes when possible.
"""

        try:
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=config_prompt,
                max_tokens=1500,
                temperature=0.1
            )
            
            config = json.loads(response.strip())
            
            # Execute the configuration actions
            configured_fields = []
            
            for field in config.get("fields_found", []):
                success = await self._configure_field(field, sso_url, entity_id, certificate)
                configured_fields.append({
                    "field_type": field["field_type"],
                    "success": success
                })
            
            # Submit the form if submit button found
            if config.get("submit_button"):
                submit_success = await self._smart_click(config["submit_button"]["selector"])
                configured_fields.append({
                    "field_type": "submit",
                    "success": submit_success
                })
            
            return {
                "configured_fields": configured_fields,
                "total_fields": len(config.get("fields_found", [])),
                "confidence": config.get("confidence", 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to configure SAML settings: {e}")
            return {"error": str(e)}
    
    async def _configure_field(
        self,
        field_config: Dict[str, Any],
        sso_url: str,
        entity_id: str,
        certificate: Optional[str]
    ) -> bool:
        """Configure a specific field based on its type."""
        
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
            
            test_info = json.loads(response.strip())
            
            # If test button available, click it
            if test_info.get("test_available") and test_info.get("test_button_selector"):
                test_success = await self._smart_click(test_info["test_button_selector"])
                test_info["test_executed"] = test_success
                
                if test_success:
                    # Wait a moment for results
                    await self.web_interactor.page.wait_for_timeout(3000)
                    
                    # Check for test results
                    updated_content = await self.web_interactor.get_current_dom(simplify=True)
                    test_info["test_results"] = "Test executed - check page for results"
            
            return test_info
            
        except Exception as e:
            self.logger.error(f"Failed to test SSO configuration: {e}")
            return {"error": str(e)}