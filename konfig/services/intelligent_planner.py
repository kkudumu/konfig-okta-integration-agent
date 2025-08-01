"""
Intelligent Planning Service

This service uses LLM reasoning to analyze documentation and generate
dynamic integration plans instead of relying on hard-coded vendor plans.
"""

import json
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from konfig.config.settings import get_settings
from konfig.utils.logging import LoggingMixin


@dataclass
class IntegrationStep:
    """Represents a single step in an integration plan."""
    name: str
    tool: str
    action: str
    params: Dict[str, Any]
    description: str
    prerequisites: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tool": self.tool,
            "action": self.action,
            "params": self.params,
            "description": self.description,
            "prerequisites": self.prerequisites or []
        }


@dataclass
class IntegrationPlan:
    """Represents a complete integration plan."""
    vendor_name: str
    steps: List[IntegrationStep]
    estimated_duration: int  # minutes
    complexity: str  # simple, moderate, complex
    requirements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor_name": self.vendor_name,
            "steps": [step.to_dict() for step in self.steps],
            "estimated_duration": self.estimated_duration,
            "complexity": self.complexity,
            "requirements": self.requirements
        }


class IntelligentPlanner(LoggingMixin):
    """
    LLM-driven service that analyzes documentation and generates
    dynamic integration plans.
    """
    
    def __init__(self):
        super().__init__()
        self.setup_logging("intelligent_planner")
        self.settings = get_settings()
        
    async def generate_integration_plan(
        self,
        vendor_info: Dict[str, Any],
        documentation_content: str,
        okta_domain: str
    ) -> IntegrationPlan:
        """
        Generate a dynamic integration plan based on documentation analysis.
        
        Args:
            vendor_info: Parsed vendor information from documentation
            documentation_content: Full documentation content
            okta_domain: Target Okta domain
            
        Returns:
            IntegrationPlan with dynamically generated steps
        """
        self.logger.info("Generating dynamic integration plan using LLM analysis")
        
        # Analyze the documentation to understand requirements
        analysis = await self._analyze_documentation(
            vendor_info, documentation_content, okta_domain
        )
        
        # Generate integration steps based on analysis
        steps = await self._generate_integration_steps(analysis)
        
        # Create the integration plan
        plan = IntegrationPlan(
            vendor_name=analysis["vendor_name"],
            steps=steps,
            estimated_duration=analysis["estimated_duration"],
            complexity=analysis["complexity"],
            requirements=analysis["requirements"]
        )
        
        self.logger.info(
            f"Generated {len(steps)} steps for {plan.vendor_name} integration",
            complexity=plan.complexity,
            duration=plan.estimated_duration
        )
        
        return plan
    
    async def _analyze_documentation(
        self,
        vendor_info: Dict[str, Any],
        documentation_content: str,
        okta_domain: str
    ) -> Dict[str, Any]:
        """Analyze documentation to understand integration requirements."""
        
        analysis_prompt = f"""
You are an expert SSO integration specialist. Analyze this documentation and determine what steps are needed for a complete SAML SSO integration.

VENDOR INFO:
{json.dumps(vendor_info, indent=2)}

DOCUMENTATION CONTENT:
{documentation_content[:4000]}...

OKTA DOMAIN: {okta_domain}

Analyze this documentation and provide a structured response in JSON format:

{{
    "vendor_name": "The actual vendor/service name (e.g., Google Workspace, Slack, etc.)",
    "admin_console_url": "URL pattern for the vendor's admin console",
    "saml_requirements": {{
        "acs_url_pattern": "Pattern for ACS URL",
        "entity_id_pattern": "Pattern for Entity ID", 
        "certificate_required": true/false,
        "attributes_required": ["list", "of", "required", "attributes"]
    }},
    "integration_phases": [
        {{
            "phase": "okta_configuration",
            "description": "Configure SAML app in Okta",
            "required_data": ["app_name", "acs_url", "entity_id"]
        }},
        {{
            "phase": "vendor_configuration", 
            "description": "Configure SSO in vendor admin console",
            "steps": ["Navigate to admin", "Find SSO settings", "Enter Okta details", "Test connection"]
        }}
    ],
    "complexity": "simple|moderate|complex",
    "estimated_duration": 15,
    "requirements": ["Admin access to vendor", "Okta admin rights", "Domain verification"]
}}

Focus on:
1. What is the actual vendor/service being configured?
2. What specific steps are needed in their admin console?
3. What SAML configuration values are required?
4. What are the prerequisite requirements?
"""

        try:
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=analysis_prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse JSON response - extract JSON from LLM response
            analysis = self._extract_json_from_response(response)
            self.logger.debug("Documentation analysis completed", vendor=analysis.get("vendor_name"))
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze documentation: {e}")
            # Fallback to basic analysis
            return self._fallback_analysis(vendor_info, okta_domain)
    
    async def _generate_integration_steps(self, analysis: Dict[str, Any]) -> List[IntegrationStep]:
        """Generate specific integration steps based on analysis."""
        
        steps_prompt = f"""
Based on this integration analysis, generate specific executable steps for SSO integration:

ANALYSIS:
{json.dumps(analysis, indent=2)}

Generate a JSON array of integration steps. Each step should be executable using these available tools:

AVAILABLE TOOLS:
- OktaAPIClient: create_saml_app, get_app_metadata, update_app_settings
- WebInteractor: navigate, click, type, select_option, fill_form, wait_for_element

Example step formats:

OKTA STEP:
{{
    "name": "Create SAML Application in Okta",
    "tool": "OktaAPIClient", 
    "action": "create_saml_app",
    "params": {{
        "label": "{{app_name}}",
        "settings": {{
            "signOn": {{
                "ssoAcsUrl": "https://example.com/saml/acs",
                "audience": "example.com",
                "recipient": "https://example.com/saml/acs"
            }}
        }}
    }},
    "description": "Creates a new SAML 2.0 application in Okta with vendor-specific settings",
    "prerequisites": []
}}

WEB INTERACTION STEP (with CSS selector):
{{
    "name": "Click Security Menu",
    "tool": "WebInteractor",
    "action": "click", 
    "params": {{
        "selector": "nav a[href*='security']"
    }},
    "description": "Navigate to security settings",
    "prerequisites": []
}}

WEB INTERACTION STEP (with text content):
{{
    "name": "Click Set up SSO Button",
    "tool": "WebInteractor",
    "action": "click",
    "params": {{
        "selector": "button",
        "text_content": "Set up SSO"
    }},
    "description": "Click the SSO setup button",
    "prerequisites": []
}}

Generate steps that cover:
1. Okta SAML app creation with vendor-specific settings
2. Retrieving Okta metadata (SSO URL, Entity ID, Certificate)
3. Navigating to vendor admin console  
4. Configuring SAML settings in vendor system
5. Testing the SSO connection

IMPORTANT: For WebInteractor steps that need selectors:
- Use VALID CSS selectors like 'button[aria-label="Set up SSO"]' or '#sso-setup-button'
- NEVER use ':contains()' pseudo-class - it's not valid CSS!
- For text-based clicking, use the "text_content" parameter instead of selectors with :contains()
- Use proper CSS selectors: 'button', 'a', 'input[type="submit"]', '.class-name', '#id-name'
- Provide meaningful parameter values, never use null or empty strings
- For click actions targeting text, use "text_content" parameter with the text to find

Return ONLY a JSON array of steps, no other text.
"""

        try:
            from konfig.services.llm_service import LLMService
            llm_service = LLMService()
            
            response = await llm_service.generate_response(
                prompt=steps_prompt,
                max_tokens=3000,
                temperature=0.1
            )
            
            # Parse JSON response - extract JSON from LLM response
            steps_data = self._extract_json_from_response(response)
            
            # Convert to IntegrationStep objects
            steps = []
            for step_data in steps_data:
                step = IntegrationStep(
                    name=step_data["name"],
                    tool=step_data["tool"],
                    action=step_data["action"],
                    params=step_data["params"],
                    description=step_data["description"],
                    prerequisites=step_data.get("prerequisites", [])
                )
                steps.append(step)
            
            self.logger.info(f"Generated {len(steps)} integration steps")
            return steps
            
        except Exception as e:
            self.logger.error(f"Failed to generate integration steps: {e}")
            # Fallback to basic steps
            return self._fallback_steps(analysis)
    
    def _fallback_analysis(self, vendor_info: Dict[str, Any], okta_domain: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        return {
            "vendor_name": vendor_info.get("vendor_name", "Unknown Vendor"),
            "admin_console_url": "https://admin.example.com",
            "saml_requirements": {
                "acs_url_pattern": "https://example.com/saml/acs",
                "entity_id_pattern": "example.com",
                "certificate_required": True,
                "attributes_required": ["email", "name"]
            },
            "integration_phases": [
                {
                    "phase": "okta_configuration",
                    "description": "Configure SAML app in Okta"
                },
                {
                    "phase": "vendor_configuration",
                    "description": "Configure SSO in vendor system"
                }
            ],
            "complexity": "moderate",
            "estimated_duration": 20,
            "requirements": ["Admin access", "Domain verification"]
        }
    
    def _fallback_steps(self, analysis: Dict[str, Any]) -> List[IntegrationStep]:
        """Fallback steps when LLM fails."""
        return [
            IntegrationStep(
                name="Create SAML Application in Okta",
                tool="OktaAPIClient",
                action="create_saml_app",
                params={"label": "{app_name}"},
                description="Creates a SAML application in Okta"
            ),
            IntegrationStep(
                name="Get Okta SAML Metadata", 
                tool="OktaAPIClient",
                action="get_app_metadata",
                params={"app_id": "{okta_app_id}"},
                description="Retrieves SAML metadata from Okta"
            )
        ]
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response that might contain extra text."""
        import re
        
        # Try to find JSON in the response
        response = response.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON blocks in code fences
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for JSON objects in the text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Look for JSON arrays in the text
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If no JSON found, raise error with original response
        raise json.JSONDecodeError(f"No valid JSON found in response: {response[:200]}...", response, 0)