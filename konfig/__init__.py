"""
Konfig - Autonomous Okta SSO Integration Agent

An intelligent AI agent that automates end-to-end SAML SSO integrations
between third-party applications and Okta.
"""

__version__ = "0.1.0"
__author__ = "Konfig Team"
__email__ = "team@konfig.ai"

from konfig.orchestrator.agent import KonfigAgent
from konfig.config.settings import Settings

__all__ = [
    "KonfigAgent",
    "Settings",
    "__version__",
]