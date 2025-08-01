"""
LLM Service

Provides interface to language models for intelligent decision making.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

from konfig.config.settings import get_settings
from konfig.utils.logging import LoggingMixin


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    content: str
    tokens_used: int
    model: str
    reasoning: Optional[str] = None


class LLMService(LoggingMixin):
    """Service for interacting with language models."""
    
    def __init__(self):
        super().__init__()
        self.setup_logging("llm_service")
        self.settings = get_settings()
        
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            model: Specific model to use (optional)
            
        Returns:
            Generated response text
        """
        try:
            # Use OpenAI as the primary LLM provider
            if self.settings.llm.openai_api_key:
                return await self._generate_openai_response(
                    prompt, max_tokens, temperature, model
                )
            
            # Fallback to Gemini if available
            elif self.settings.llm.gemini_api_key:
                return await self._generate_gemini_response(
                    prompt, max_tokens, temperature, model
                )
            
            else:
                raise ValueError("No LLM API key configured")
                
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise
    
    async def _generate_openai_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        model: Optional[str]
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
            
            # Configure OpenAI client
            client = openai.AsyncOpenAI(
                api_key=self.settings.llm.openai_api_key
            )
            
            # Use specified model or default
            model_name = model or self.settings.llm.openai_model or "gpt-4o-mini"
            
            self.logger.debug(f"Generating response with {model_name}")
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SSO integration specialist with deep knowledge of SAML, OAuth, and enterprise authentication systems."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "text"}
            )
            
            content = response.choices[0].message.content
            
            self.logger.info(
                "OpenAI response generated",
                model=model_name,
                tokens_used=response.usage.total_tokens,
                response_length=len(content)
            )
            
            return content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _generate_gemini_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        model: Optional[str]
    ) -> str:
        """Generate response using Gemini API."""
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.settings.llm.gemini_api_key)
            
            model_name = model or self.settings.llm.gemini_model or "gemini-2.5-flash"
            model = genai.GenerativeModel(model_name)
            
            self.logger.debug(f"Generating response with {model_name}")
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            
            content = response.text
            
            self.logger.info(
                "Gemini response generated",
                model=model_name,
                response_length=len(content)
            )
            
            return content
            
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
    
    async def analyze_integration_requirements(
        self,
        documentation: str,
        vendor_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze documentation to understand integration requirements.
        
        This is a specialized method for SSO integration analysis.
        """
        prompt = f"""
        Analyze this SSO integration documentation and extract key requirements:
        
        VENDOR INFO: {json.dumps(vendor_info, indent=2)}
        
        DOCUMENTATION:
        {documentation[:3000]}...
        
        Provide a JSON response with:
        1. Vendor identification and admin console details
        2. SAML configuration requirements
        3. Step-by-step integration process
        4. Prerequisites and dependencies
        
        Focus on actionable information that can be used to automate the integration.
        """
        
        response = await self.generate_response(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("LLM response was not valid JSON, returning text")
            return {"analysis": response}