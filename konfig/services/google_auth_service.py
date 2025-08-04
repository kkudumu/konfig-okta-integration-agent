"""
Google Authentication Service

Handles authentication to Google Admin Console for automated SAML configuration.
"""

import asyncio
from typing import Dict, Any, Optional

from konfig.config.settings import get_settings
from konfig.utils.logging import LoggingMixin


class GoogleAuthService(LoggingMixin):
    """Service for handling Google Admin Console authentication."""
    
    def __init__(self):
        super().__init__()
        self.setup_logging("google_auth_service")
        self.settings = get_settings()
    
    async def authenticate_to_admin_console(
        self,
        web_interactor,
        admin_url: str = "https://admin.google.com"
    ) -> Dict[str, Any]:
        """
        Authenticate to Google Admin Console using stored credentials.
        
        Args:
            web_interactor: WebInteractor instance for browser automation
            admin_url: URL of Google Admin Console
            
        Returns:
            Dict with authentication result
        """
        self.logger.info("Starting Google Admin Console authentication")
        
        try:
            # Navigate to admin console
            nav_result = await web_interactor.navigate(admin_url)
            current_url = await web_interactor.get_current_url()
            
            self.logger.info(f"Navigated to: {current_url}")
            
            # Check if we're on Google sign-in page
            if "accounts.google.com" in current_url:
                self.logger.info("Detected Google sign-in page, attempting authentication")
                
                # Get credentials from settings (prefer vendor settings, fallback to google settings)
                username = self.settings.vendor.username or self.settings.google.admin_username
                password = self.settings.vendor.password or self.settings.google.admin_password
                
                if not username or not password:
                    self.logger.warning("Google admin credentials not configured")
                    return {
                        "success": False,
                        "message": "Google admin credentials not found in configuration"
                    }
                
                # Perform login
                return await self._perform_google_login(web_interactor, username, password)
            
            else:
                # Already authenticated or different page
                return {
                    "success": True,
                    "message": "Already authenticated or different auth flow",
                    "current_url": current_url
                }
                
        except Exception as e:
            self.logger.error(f"Google authentication failed: {e}")
            return {
                "success": False,
                "message": f"Authentication failed: {str(e)}"
            }
    
    async def _perform_google_login(
        self,
        web_interactor,
        username: str,
        password: str
    ) -> Dict[str, Any]:
        """
        Perform actual Google login with credentials.
        
        Args:
            web_interactor: WebInteractor instance
            username: Google admin username
            password: Google admin password
            
        Returns:
            Dict with login result
        """
        try:
            # Wait for page to load
            await asyncio.sleep(2)
            
            # Step 1: Enter email
            self.logger.info("Entering email address")
            email_selectors = [
                'input[type="email"]',
                '#identifierId',
                'input[name="identifier"]',
                'input[autocomplete="username"]'
            ]
            
            email_entered = False
            for selector in email_selectors:
                if await web_interactor.element_exists(selector):
                    await web_interactor.type(selector, username)
                    email_entered = True
                    self.logger.info(f"Email entered using selector: {selector}")
                    break
            
            if not email_entered:
                return {
                    "success": False,
                    "message": "Could not find email input field"
                }
            
            # Click Next button
            next_selectors = [
                '#identifierNext',
                'button[jsname="LgbsSe"]',
                'button:contains("Next")',
                'input[type="submit"]'
            ]
            
            next_clicked = False
            for selector in next_selectors:
                try:
                    if await web_interactor.element_exists(selector):
                        await web_interactor.click(selector)
                        next_clicked = True
                        self.logger.info(f"Next button clicked using selector: {selector}")
                        break
                except:
                    continue
            
            if not next_clicked:
                # Try pressing Enter as fallback
                await web_interactor.page.keyboard.press('Enter')
                self.logger.info("Pressed Enter as fallback for Next button")
            
            # Wait for password page
            await asyncio.sleep(3)
            
            # Step 2: Enter password
            self.logger.info("Entering password")
            password_selectors = [
                'input[type="password"]',
                '#password input',
                'input[name="password"]',
                'input[autocomplete="current-password"]'
            ]
            
            password_entered = False
            for selector in password_selectors:
                if await web_interactor.element_exists(selector):
                    await web_interactor.type(selector, password, secret=True)
                    password_entered = True
                    self.logger.info(f"Password entered using selector: {selector}")
                    break
            
            if not password_entered:
                return {
                    "success": False,
                    "message": "Could not find password input field"
                }
            
            # Click Sign In button
            signin_selectors = [
                '#passwordNext',
                'button[jsname="LgbsSe"]',
                'button:contains("Next")',
                'input[type="submit"]'
            ]
            
            signin_clicked = False
            for selector in signin_selectors:
                try:
                    if await web_interactor.element_exists(selector):
                        await web_interactor.click(selector)
                        signin_clicked = True
                        self.logger.info(f"Sign in button clicked using selector: {selector}")
                        break
                except:
                    continue
            
            if not signin_clicked:
                # Try pressing Enter as fallback
                await web_interactor.page.keyboard.press('Enter')
                self.logger.info("Pressed Enter as fallback for Sign In button")
            
            # Wait for authentication to complete
            await asyncio.sleep(5)
            
            # Check if we're now on admin console
            current_url = await web_interactor.get_current_url()
            
            if "admin.google.com" in current_url:
                self.logger.info("Successfully authenticated to Google Admin Console")
                return {
                    "success": True,
                    "message": "Successfully authenticated to Google Admin Console",
                    "current_url": current_url
                }
            else:
                self.logger.warning(f"Authentication may have failed, current URL: {current_url}")
                return {
                    "success": False,
                    "message": f"Authentication uncertain, current URL: {current_url}"
                }
                
        except Exception as e:
            self.logger.error(f"Login process failed: {e}")
            return {
                "success": False,
                "message": f"Login process failed: {str(e)}"
            }