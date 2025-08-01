"""
OktaAPIClient Tool for Konfig.

This tool provides programmatic access to the Okta Management API
for creating and configuring SAML applications during SSO integrations.
"""

import asyncio
import base64
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx
from cryptography import x509
from cryptography.x509.oid import NameOID

from konfig.config.settings import get_settings
from konfig.utils.logging import LoggingMixin


class OktaAPIError(Exception):
    """Base exception for Okta API errors."""
    pass


class OktaAuthenticationError(OktaAPIError):
    """Raised when API authentication fails."""
    pass


class OktaRateLimitError(OktaAPIError):
    """Raised when API rate limit is exceeded."""
    pass


class OktaResourceNotFoundError(OktaAPIError):
    """Raised when a requested resource is not found."""
    pass


class OktaAPIClient(LoggingMixin):
    """
    Okta Management API client for SAML application management.
    
    This client provides high-level methods for creating, configuring,
    and managing SAML applications in an Okta organization.
    """
    
    def __init__(
        self,
        okta_domain: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """
        Initialize the Okta API client.
        
        Args:
            okta_domain: Okta domain (e.g., company.okta.com)
            api_token: Okta API token for authentication
        """
        super().__init__()
        self.setup_logging("okta_api_client")
        
        settings = get_settings()
        
        # Use provided values or fall back to settings
        self.okta_domain = okta_domain or settings.okta.domain
        self.api_token = api_token or settings.okta.api_token
        
        # Validate required configuration
        if not self.okta_domain:
            raise OktaAPIError("Okta domain is required")
        if not self.api_token:
            raise OktaAPIError("Okta API token is required")
        
        # Build base URL
        if not self.okta_domain.startswith('https://'):
            self.base_url = f"https://{self.okta_domain}"
        else:
            self.base_url = self.okta_domain
            
        self.api_base_url = urljoin(self.base_url, "/api/v1/")
        
        # Rate limiting configuration
        self.rate_limit_max_requests = settings.okta.rate_limit_max_requests
        self.rate_limit_window_seconds = settings.okta.rate_limit_window_seconds
        self._rate_limit_requests = []
        
        # HTTP client configuration
        self.timeout = httpx.Timeout(30.0)
        self.headers = {
            "Authorization": f"SSWS {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Konfig-Agent/1.0"
        }
        
        self.logger.info("Okta API client initialized", domain=self.okta_domain)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, '_client'):
            await self._client.aclose()
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove old requests outside the window
        cutoff_time = current_time - self.rate_limit_window_seconds
        self._rate_limit_requests = [
            req_time for req_time in self._rate_limit_requests
            if req_time > cutoff_time
        ]
        
        # Check if we're at the limit
        if len(self._rate_limit_requests) >= self.rate_limit_max_requests:
            sleep_time = self._rate_limit_requests[0] + self.rate_limit_window_seconds - current_time
            if sleep_time > 0:
                raise OktaRateLimitError(f"Rate limit exceeded. Wait {sleep_time:.2f} seconds.")
        
        # Record this request
        self._rate_limit_requests.append(current_time)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the Okta API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
        
        Returns:
            Response data as dictionary
        """
        # Check rate limiting
        self._check_rate_limit()
        
        url = urljoin(self.api_base_url, endpoint)
        
        self.log_method_call(
            "_make_request",
            method=method,
            endpoint=endpoint,
            has_data=data is not None
        )
        
        try:
            if not hasattr(self, '_client'):
                raise OktaAPIError("Client not initialized. Use async context manager.")
            
            response = await self._client.request(
                method=method,
                url=url,
                json=data,
                params=params
            )
            
            # Handle different response types
            if response.status_code == 401:
                raise OktaAuthenticationError("Invalid API token or insufficient permissions")
            elif response.status_code == 403:
                raise OktaAuthenticationError("Access forbidden - check API token permissions")
            elif response.status_code == 404:
                raise OktaResourceNotFoundError("Resource not found")
            elif response.status_code == 429:
                # Rate limited by Okta
                retry_after = response.headers.get("X-Rate-Limit-Reset", "60")
                raise OktaRateLimitError(f"Okta rate limit exceeded. Retry after {retry_after} seconds.")
            elif response.status_code >= 400:
                error_data = response.json() if response.content else {}
                error_message = error_data.get("errorSummary", f"HTTP {response.status_code}")
                raise OktaAPIError(f"API request failed: {error_message}")
            
            # Parse response
            if response.content:
                result = response.json()
            else:
                result = {}
            
            self.log_method_result(
                "_make_request",
                {"status_code": response.status_code, "has_content": bool(response.content)}
            )
            
            return result
            
        except httpx.RequestError as e:
            self.log_error("_make_request", e, method=method, endpoint=endpoint)
            raise OktaAPIError(f"Request failed: {e}")
    
    # ========== APPLICATION MANAGEMENT ==========
    
    async def create_saml_app(
        self,
        label: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new SAML 2.0 application in Okta.
        
        Args:
            label: Display name for the application
            settings: Optional application settings
        
        Returns:
            Created application object
        """
        self.log_method_call("create_saml_app", label=label, has_settings=settings is not None)
        
        # Default SAML 2.0 application configuration
        app_config = {
            "label": label,
            "signOnMode": "SAML_2_0",
            "visibility": {
                "autoSubmitToolbar": False,
                "hide": {
                    "iOS": False,
                    "web": False
                }
            },
            "settings": {
                "app": settings.get("app", {}) if settings else {},
                "signOn": {
                    # Basic SAML settings - will be updated with vendor-specific information
                    "defaultRelayState": "",
                    "ssoAcsUrl": settings.get("ssoAcsUrl") if settings else "https://example.com/sso/saml",
                    "audience": settings.get("audience") if settings else "https://example.com",
                    "recipient": settings.get("recipient") if settings else "https://example.com/sso/saml",
                    "destination": settings.get("destination") if settings else "https://example.com/sso/saml",
                    "subjectNameIdTemplate": "${user.userName}",
                    "subjectNameIdFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified",
                    "responseSigned": True,
                    "assertionSigned": True,
                    "signatureAlgorithm": "RSA_SHA256",
                    "digestAlgorithm": "SHA256",
                    "honorForceAuthn": True,
                    "authnContextClassRef": "urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport"
                }
            }
        }
        
        # Merge any custom settings
        if settings:
            if "signOn" in settings:
                app_config["settings"]["signOn"].update(settings["signOn"])
        
        try:
            result = await self._make_request("POST", "apps", data=app_config)
            
            app_id = result.get("id")
            sso_url = self._extract_sso_url(result)
            entity_id = result.get("settings", {}).get("signOn", {}).get("audience")
            
            self.logger.info(
                "SAML application created",
                label=label,
                app_id=app_id,
                sso_url=sso_url
            )
            
            return result
            
        except Exception as e:
            self.log_error("create_saml_app", e, label=label)
            raise
    
    async def get_app_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Get an application by its label.
        
        Args:
            label: Application label to search for
        
        Returns:
            Application object or None if not found
        """
        self.log_method_call("get_app_by_label", label=label)
        
        try:
            # Search for applications with the given label
            params = {"q": label, "limit": 10}
            result = await self._make_request("GET", "apps", params=params)
            
            # Find exact match
            for app in result:
                if app.get("label") == label:
                    self.log_method_result("get_app_by_label", {"found": True})
                    return app
            
            self.log_method_result("get_app_by_label", {"found": False})
            return None
            
        except Exception as e:
            self.log_error("get_app_by_label", e, label=label)
            raise
    
    async def get_app_by_id(self, app_id: str) -> Dict[str, Any]:
        """
        Get an application by its ID.
        
        Args:
            app_id: Application ID
        
        Returns:
            Application object
        """
        self.log_method_call("get_app_by_id", app_id=app_id)
        
        try:
            result = await self._make_request("GET", f"apps/{app_id}")
            self.log_method_result("get_app_by_id", {"found": True})
            return result
            
        except OktaResourceNotFoundError:
            self.log_method_result("get_app_by_id", {"found": False})
            raise
        except Exception as e:
            self.log_error("get_app_by_id", e, app_id=app_id)
            raise
    
    async def update_app(
        self,
        app_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an existing application.
        
        Args:
            app_id: Application ID
            updates: Fields to update
        
        Returns:
            Updated application object
        """
        self.log_method_call("update_app", app_id=app_id, num_updates=len(updates))
        
        try:
            result = await self._make_request("PUT", f"apps/{app_id}", data=updates)
            
            self.logger.info("Application updated", app_id=app_id)
            return result
            
        except Exception as e:
            self.log_error("update_app", e, app_id=app_id)
            raise
    
    async def update_saml_settings(
        self,
        app_id: str,
        saml_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update SAML-specific settings for an application.
        
        Args:
            app_id: Application ID
            saml_settings: SAML configuration updates
        
        Returns:
            Updated application object
        """
        self.log_method_call("update_saml_settings", app_id=app_id)
        
        try:
            # Get current application
            current_app = await self.get_app_by_id(app_id)
            
            # Update SAML settings
            current_settings = current_app.get("settings", {})
            current_sign_on = current_settings.get("signOn", {})
            current_sign_on.update(saml_settings)
            
            # Prepare update payload
            updates = {
                "settings": {
                    **current_settings,
                    "signOn": current_sign_on
                }
            }
            
            result = await self.update_app(app_id, updates)
            
            self.logger.info("SAML settings updated", app_id=app_id)
            return result
            
        except Exception as e:
            self.log_error("update_saml_settings", e, app_id=app_id)
            raise
    
    async def get_app_metadata(self, app_id: str) -> Dict[str, Any]:
        """
        Get SAML metadata for an application.
        
        Args:
            app_id: Application ID
        
        Returns:
            SAML metadata including Entity ID, SSO URL, and certificate
        """
        self.log_method_call("get_app_metadata", app_id=app_id)
        
        try:
            # Get application details
            app = await self.get_app_by_id(app_id)
            
            # Extract SAML metadata
            sign_on_settings = app.get("settings", {}).get("signOn", {})
            
            # Get the actual SSO URL from Okta
            sso_url = self._extract_sso_url(app)
            
            # Get signing certificate
            cert_response = await self._make_request("GET", f"apps/{app_id}/credentials/keys")
            certificate = None
            if cert_response and len(cert_response) > 0:
                # Get the first active certificate
                for cert_info in cert_response:
                    if cert_info.get("status") == "ACTIVE":
                        certificate = cert_info.get("x5c", [None])[0]
                        break
            
            metadata = {
                "app_id": app_id,
                "entity_id": f"http://www.okta.com/{app_id}",
                "sso_url": sso_url,
                "slo_url": None,  # Okta doesn't typically provide SLO for SAML apps
                "certificate": certificate,
                "audience": sign_on_settings.get("audience"),
                "recipient": sign_on_settings.get("recipient"),
                "destination": sign_on_settings.get("destination"),
                "name_id_format": sign_on_settings.get("subjectNameIdFormat"),
                "signature_algorithm": sign_on_settings.get("signatureAlgorithm"),
                "digest_algorithm": sign_on_settings.get("digestAlgorithm"),
            }
            
            self.log_method_result("get_app_metadata", {"has_certificate": certificate is not None})
            return metadata
            
        except Exception as e:
            self.log_error("get_app_metadata", e, app_id=app_id)
            raise
    
    def _extract_sso_url(self, app: Dict[str, Any]) -> Optional[str]:
        """Extract the SSO URL from an application object."""
        app_id = app.get("id")
        if app_id:
            return f"{self.base_url}/app/{app_id}/sso/saml"
        return None
    
    # ========== USER ASSIGNMENT ==========
    
    async def assign_user_to_app(
        self,
        user_id: str,
        app_id: str,
        profile_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assign a user to an application.
        
        Args:
            user_id: Okta user ID
            app_id: Application ID
            profile_attributes: Optional user profile attributes for the app
        
        Returns:
            Assignment result
        """
        self.log_method_call("assign_user_to_app", user_id=user_id, app_id=app_id)
        
        assignment_data = {
            "id": user_id,
            "scope": "USER",
            "profile": profile_attributes or {}
        }
        
        try:
            result = await self._make_request(
                "POST",
                f"apps/{app_id}/users",
                data=assignment_data
            )
            
            self.logger.info("User assigned to application", user_id=user_id, app_id=app_id)
            return result
            
        except Exception as e:
            self.log_error("assign_user_to_app", e, user_id=user_id, app_id=app_id)
            raise
    
    async def assign_group_to_app(
        self,
        group_id: str,
        app_id: str,
        profile_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assign a group to an application.
        
        Args:
            group_id: Okta group ID
            app_id: Application ID
            profile_attributes: Optional group profile attributes
        
        Returns:
            Assignment result
        """
        self.log_method_call("assign_group_to_app", group_id=group_id, app_id=app_id)
        
        assignment_data = {
            "id": group_id,
            "profile": profile_attributes or {}
        }
        
        try:
            result = await self._make_request(
                "POST",
                f"apps/{app_id}/groups",
                data=assignment_data
            )
            
            self.logger.info("Group assigned to application", group_id=group_id, app_id=app_id)
            return result
            
        except Exception as e:
            self.log_error("assign_group_to_app", e, group_id=group_id, app_id=app_id)
            raise
    
    # ========== CERTIFICATE MANAGEMENT ==========
    
    async def upload_saml_certificate(
        self,
        app_id: str,
        certificate: str,
        certificate_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a SAML signing certificate to an application.
        
        Args:
            app_id: Application ID
            certificate: PEM-encoded certificate
            certificate_name: Optional certificate name
        
        Returns:
            Certificate upload result
        """
        self.log_method_call("upload_saml_certificate", app_id=app_id)
        
        try:
            # Parse certificate to validate and extract information
            cert_obj = x509.load_pem_x509_certificate(certificate.encode())
            
            # Extract certificate details
            subject = cert_obj.subject
            issuer = cert_obj.issuer
            serial_number = cert_obj.serial_number
            
            # Convert to DER format for upload
            der_cert = cert_obj.public_bytes(serialization.Encoding.DER)
            der_b64 = base64.b64encode(der_cert).decode()
            
            # Prepare certificate data
            cert_data = {
                "name": certificate_name or f"Certificate-{serial_number}",
                "x5c": [der_b64],
                "kty": "RSA",  # Assuming RSA certificate
                "use": "sig",
                "key_ops": ["verify"]
            }
            
            result = await self._make_request(
                "POST",
                f"apps/{app_id}/credentials/keys",
                data=cert_data
            )
            
            self.logger.info("Certificate uploaded", app_id=app_id, serial_number=serial_number)
            return result
            
        except Exception as e:
            self.log_error("upload_saml_certificate", e, app_id=app_id)
            raise
    
    async def get_app_certificates(self, app_id: str) -> List[Dict[str, Any]]:
        """
        Get all certificates for an application.
        
        Args:
            app_id: Application ID
        
        Returns:
            List of certificate objects
        """
        self.log_method_call("get_app_certificates", app_id=app_id)
        
        try:
            result = await self._make_request("GET", f"apps/{app_id}/credentials/keys")
            
            self.log_method_result("get_app_certificates", {"num_certificates": len(result)})
            return result
            
        except Exception as e:
            self.log_error("get_app_certificates", e, app_id=app_id)
            raise
    
    # ========== USER AND GROUP MANAGEMENT ==========
    
    async def get_user_by_login(self, login: str) -> Optional[Dict[str, Any]]:
        """
        Get a user by their login name.
        
        Args:
            login: User login name
        
        Returns:
            User object or None if not found
        """
        self.log_method_call("get_user_by_login", login=login)
        
        try:
            result = await self._make_request("GET", f"users/{login}")
            self.log_method_result("get_user_by_login", {"found": True})
            return result
            
        except OktaResourceNotFoundError:
            self.log_method_result("get_user_by_login", {"found": False})
            return None
        except Exception as e:
            self.log_error("get_user_by_login", e, login=login)
            raise
    
    async def get_group_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a group by name.
        
        Args:
            name: Group name
        
        Returns:
            Group object or None if not found
        """
        self.log_method_call("get_group_by_name", name=name)
        
        try:
            params = {"q": name, "limit": 10}
            result = await self._make_request("GET", "groups", params=params)
            
            # Find exact match
            for group in result:
                if group.get("profile", {}).get("name") == name:
                    self.log_method_result("get_group_by_name", {"found": True})
                    return group
            
            self.log_method_result("get_group_by_name", {"found": False})
            return None
            
        except Exception as e:
            self.log_error("get_group_by_name", e, name=name)
            raise
    
    # ========== UTILITY METHODS ==========
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the API connection and authentication.
        
        Returns:
            Connection test result
        """
        self.log_method_call("test_connection")
        
        try:
            # Test with a simple API call
            result = await self._make_request("GET", "org")
            
            org_info = {
                "success": True,
                "org_id": result.get("id"),
                "org_name": result.get("companyName"),
                "domain": self.okta_domain,
                "api_version": "v1"
            }
            
            self.logger.info("API connection test successful", org_name=org_info["org_name"])
            return org_info
            
        except Exception as e:
            self.log_error("test_connection", e)
            return {
                "success": False,
                "error": str(e),
                "domain": self.okta_domain
            }
    
    async def list_apps(
        self,
        app_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List applications in the Okta org.
        
        Args:
            app_type: Optional application type filter
            limit: Maximum number of apps to return
        
        Returns:
            List of application objects
        """
        self.log_method_call("list_apps", app_type=app_type, limit=limit)
        
        try:
            params = {"limit": limit}
            if app_type:
                params["filter"] = f'signOnMode eq "{app_type}"'
            
            result = await self._make_request("GET", "apps", params=params)
            
            self.log_method_result("list_apps", {"num_apps": len(result)})
            return result
            
        except Exception as e:
            self.log_error("list_apps", e, app_type=app_type)
            raise
    
    async def delete_app(self, app_id: str) -> bool:
        """
        Delete an application.
        
        Args:
            app_id: Application ID to delete
        
        Returns:
            True if deleted successfully
        """
        self.log_method_call("delete_app", app_id=app_id)
        
        try:
            # First deactivate the app
            await self._make_request("POST", f"apps/{app_id}/lifecycle/deactivate")
            
            # Then delete it
            await self._make_request("DELETE", f"apps/{app_id}")
            
            self.logger.info("Application deleted", app_id=app_id)
            return True
            
        except Exception as e:
            self.log_error("delete_app", e, app_id=app_id)
            raise