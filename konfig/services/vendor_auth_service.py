"""
Vendor Authentication Service

Handles authentication to any vendor admin console using LLM intelligence
to dynamically analyze login pages and determine appropriate actions.
"""

import asyncio
from typing import Dict, Any, Optional

from konfig.config.settings import get_settings
from konfig.utils.logging import LoggingMixin


class VendorAuthService(LoggingMixin):
    """Service for handling intelligent vendor admin console authentication."""
    
    def __init__(self):
        super().__init__()
        self.setup_logging("vendor_auth_service")
        self.settings = get_settings()
        
        # Initialize LLM service for intelligent analysis
        try:
            from konfig.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM service: {e}")
            self.llm_service = None
    
    async def authenticate_to_vendor_console(
        self,
        web_interactor,
        admin_url: str,
        vendor_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Intelligently authenticate to any vendor admin console using LLM analysis.
        
        Args:
            web_interactor: WebInteractor instance for browser automation
            admin_url: URL of vendor admin console
            vendor_name: Name of the vendor for context
            
        Returns:
            Dict with authentication result
        """
        self.logger.info(f"Starting intelligent authentication for {vendor_name}")
        
        try:
            # Navigate to admin console
            nav_result = await web_interactor.navigate(admin_url)
            current_url = await web_interactor.get_current_url()
            
            self.logger.info(f"Navigated to: {current_url}")
            
            # Use LLM to intelligently analyze and authenticate
            return await self._intelligent_authenticate(web_interactor, vendor_name)
                
        except Exception as e:
            self.logger.error(f"{vendor_name} authentication failed: {e}")
            return {
                "success": False,
                "message": f"Authentication failed: {str(e)}"
            }
    
    async def _intelligent_authenticate(
        self,
        web_interactor,
        vendor_name: str
    ) -> Dict[str, Any]:
        """
        Use LLM intelligence to analyze the login page and determine authentication actions.
        
        Args:
            web_interactor: WebInteractor instance
            vendor_name: Name of the vendor for context
            
        Returns:
            Dict with authentication result
        """
        self.logger.info(f"Starting intelligent authentication analysis for {vendor_name}")
        
        # Get credentials from settings
        username = self.settings.vendor.username
        password = self.settings.vendor.password
        
        if not username or not password:
            self.logger.warning("Vendor credentials not configured")
            return {
                "success": False,
                "message": "Vendor credentials not found in configuration"
            }
        
        if not self.llm_service:
            self.logger.warning("LLM service not available, falling back to basic authentication")
            return await self._basic_authenticate(web_interactor, username, password, vendor_name)
        
        try:
            # Get current page content for analysis
            await asyncio.sleep(2)  # Let page load
            
            # Get simplified DOM for LLM analysis
            dom_content = await web_interactor.get_current_dom(simplify=True)
            current_url = await web_interactor.get_current_url()
            
            # Create prompt for LLM to analyze login page
            analysis_prompt = f"""
            You are analyzing a {vendor_name} login page to determine the correct authentication steps.
            
            Current URL: {current_url}
            Page DOM: {dom_content}
            
            I have credentials:
            - Username: {username}
            - Password: [REDACTED]
            
            Analyze this login page and provide a step-by-step plan to authenticate. 
            Return your response as JSON with this structure:
            {{
                "steps": [
                    {{
                        "action": "type",
                        "selector": "css_selector_for_username_field",
                        "value": "username",
                        "description": "Enter username in the email/username field"
                    }},
                    {{
                        "action": "type", 
                        "selector": "css_selector_for_password_field",
                        "value": "password",
                        "description": "Enter password in the password field"
                    }},
                    {{
                        "action": "click",
                        "selector": "css_selector_for_login_button",
                        "description": "Click the login/submit button"
                    }}
                ],
                "reasoning": "Explanation of how you identified the login elements"
            }}
            
            CRITICAL REQUIREMENTS:
            1. NEVER use generic selectors like "input", "button", "div" - they will fail
            2. You MUST analyze the actual DOM content provided and use EXACT selectors that exist
            3. For Salesforce specifically:
               - Username field is typically: input[id="username"] or input[name="username"]
               - Password field is typically: input[id="password"] or input[type="password"] 
               - Login button is typically: input[id="Login"] or input[value="Log In"]
            4. Look for these specific patterns in the DOM:
               - id attributes (preferred): #username, #password, #Login
               - name attributes: input[name="username"], input[name="password"]
               - type attributes: input[type="email"], input[type="password"]
               - value attributes for buttons: input[value="Log In"]
            
            REASONING PROCESS:
            1. Search the DOM for password-related elements first (most distinctive)
            2. Then find username/email elements  
            3. Finally locate the submit/login button
            4. Verify each selector exists in the provided DOM before suggesting it
            
            If you cannot find specific selectors in the DOM, say so explicitly in your reasoning.
            """
            
            # Get LLM analysis
            analysis_response = await self.llm_service.generate_response(analysis_prompt)
            
            try:
                import json
                analysis = json.loads(analysis_response)
                self.logger.info(f"LLM analysis: {analysis.get('reasoning', 'No reasoning provided')}")
                
                # Execute the planned authentication steps
                return await self._execute_authentication_steps(
                    web_interactor, 
                    analysis['steps'], 
                    username, 
                    password, 
                    vendor_name
                )
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Could not parse LLM response: {e}")
                self.logger.info("Falling back to basic authentication")
                return await self._basic_authenticate(web_interactor, username, password, vendor_name)
                
        except Exception as e:
            self.logger.error(f"Intelligent authentication failed: {e}")
            self.logger.info("Falling back to basic authentication")
            return await self._basic_authenticate(web_interactor, username, password, vendor_name)
    
    async def _execute_authentication_steps(
        self,
        web_interactor,
        steps: list,
        username: str,
        password: str,
        vendor_name: str
    ) -> Dict[str, Any]:
        """
        Execute the authentication steps determined by LLM analysis.
        
        Args:
            web_interactor: WebInteractor instance
            steps: List of authentication steps from LLM
            username: Username to enter
            password: Password to enter
            vendor_name: Vendor name for logging
            
        Returns:
            Dict with authentication result
        """
        try:
            for step in steps:
                action = step.get('action')
                selector = step.get('selector')
                value = step.get('value')
                description = step.get('description', f"{action} action")
                
                self.logger.info(f"Executing step: {description}")
                
                if action == 'type':
                    if value == 'username':
                        actual_value = username
                        secret = False
                    elif value == 'password':
                        actual_value = password
                        secret = True
                    else:
                        actual_value = value
                        secret = False
                    
                    await web_interactor.type(selector, actual_value, secret=secret, clear_first=True)
                    
                elif action == 'click':
                    await web_interactor.click(selector)
                    
                elif action == 'wait':
                    wait_time = step.get('duration', 2)
                    await asyncio.sleep(wait_time)
                
                # Small delay between actions
                await asyncio.sleep(1)
            
            # Wait for authentication to complete
            await asyncio.sleep(5)
            
            # Check authentication success
            current_url = await web_interactor.get_current_url()
            
            # Basic success check - if we're still on a login page, it likely failed
            login_indicators = ['login', 'signin', 'sign-in', 'auth', 'authenticate']
            is_still_login_page = any(indicator in current_url.lower() for indicator in login_indicators)
            
            if not is_still_login_page:
                self.logger.info(f"Authentication appears successful for {vendor_name}")
                return {
                    "success": True,
                    "message": f"Successfully authenticated to {vendor_name}",
                    "current_url": current_url
                }
            else:
                self.logger.warning(f"Still on login page, authentication may have failed")
                return {
                    "success": False,
                    "message": f"Authentication uncertain - still on login page: {current_url}"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute authentication steps: {e}")
            return {
                "success": False,
                "message": f"Authentication execution failed: {str(e)}"
            }
    
    async def _basic_authenticate(
        self,
        web_interactor,
        username: str,
        password: str,
        vendor_name: str
    ) -> Dict[str, Any]:
        """
        Basic authentication fallback using common selectors.
        
        Args:
            web_interactor: WebInteractor instance
            username: Username to enter
            password: Password to enter
            vendor_name: Vendor name for logging
            
        Returns:
            Dict with authentication result
        """
        self.logger.info(f"Attempting basic authentication for {vendor_name}")
        
        try:
            # Common username/email field selectors
            username_selectors = [
                'input[type="email"]',
                'input[type="text"]',
                'input[name="username"]',
                'input[name="email"]',
                'input[id="username"]',
                'input[id="email"]',
                'input[autocomplete="username"]'
            ]
            
            username_entered = False
            for selector in username_selectors:
                if await web_interactor.element_exists(selector):
                    await web_interactor.type(selector, username, clear_first=True)
                    username_entered = True
                    self.logger.info(f"Username entered using selector: {selector}")
                    break
            
            if not username_entered:
                return {
                    "success": False,
                    "message": "Could not find username input field"
                }
            
            # Common password field selectors
            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[id="password"]',
                'input[autocomplete="current-password"]'
            ]
            
            password_entered = False
            for selector in password_selectors:
                if await web_interactor.element_exists(selector):
                    await web_interactor.type(selector, password, secret=True, clear_first=True)
                    password_entered = True
                    self.logger.info(f"Password entered using selector: {selector}")
                    break
            
            if not password_entered:
                return {
                    "success": False,
                    "message": "Could not find password input field"
                }
            
            # Common submit button selectors
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:contains("Login")',
                'button:contains("Sign In")',
                'button:contains("Log In")',
                'button:contains("Submit")',
                '#login',
                '#submit'
            ]
            
            submit_clicked = False
            for selector in submit_selectors:
                try:
                    if await web_interactor.element_exists(selector):
                        await web_interactor.click(selector)
                        submit_clicked = True
                        self.logger.info(f"Submit button clicked using selector: {selector}")
                        break
                except:
                    continue
            
            if not submit_clicked:
                # Try pressing Enter as fallback
                await web_interactor.page.keyboard.press('Enter')
                self.logger.info("Pressed Enter as fallback for submit")
            
            await asyncio.sleep(5)
            
            current_url = await web_interactor.get_current_url()
            return {
                "success": True,
                "message": f"Basic authentication completed for {vendor_name}",
                "current_url": current_url
            }
            
        except Exception as e:
            self.logger.error(f"Basic authentication failed: {e}")
            return {
                "success": False,
                "message": f"Authentication failed: {str(e)}"
            }