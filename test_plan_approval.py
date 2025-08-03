#!/usr/bin/env python3
"""
Test script for the plan approval system.
"""

import asyncio
import sys
from pathlib import Path

# Add the konfig package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from konfig.services.plan_approval import PlanApprovalService


async def test_plan_approval():
    """Test the plan approval functionality."""
    print("🔧 Testing Plan Approval System...")
    
    # Create a sample integration plan
    sample_plan = [
        {
            "name": "Create Okta Application",
            "tool": "OktaAPI",
            "action": "create_application",
            "params": {
                "name": "Google Workspace SAML",
                "type": "saml"
            }
        },
        {
            "name": "Configure SAML Settings",
            "tool": "OktaAPI", 
            "action": "configure_saml",
            "params": {
                "sso_url": "https://accounts.google.com/samlrp/...",
                "entity_id": "https://accounts.google.com/samlrp/..."
            }
        },
        {
            "name": "Navigate to Google Admin Console",
            "tool": "WebInteractor",
            "action": "navigate",
            "params": {
                "url": "https://admin.google.com"
            }
        },
        {
            "name": "Login to Google Admin",
            "tool": "WebInteractor",
            "action": "type",
            "params": {
                "selector": "input[type='email']",
                "text": "admin@company.com",
                "secret": False
            }
        },
        {
            "name": "Enter Admin Password",
            "tool": "WebInteractor",
            "action": "type",
            "params": {
                "selector": "input[type='password']",
                "text": "admin_password",
                "secret": True
            }
        },
        {
            "name": "Configure Google SSO Settings",
            "tool": "WebInteractor",
            "action": "click",
            "params": {
                "selector": "button[aria-label='Set up SSO']"
            }
        },
        {
            "name": "Test SSO Connection",
            "tool": "WebInteractor",
            "action": "click",
            "params": {
                "selector": "button:has-text('Test Connection')"
            }
        },
        {
            "name": "Clean Up Test Data (Optional)",
            "tool": "OktaAPI",
            "action": "cleanup_test_data",
            "params": {
                "test_users": ["test@company.com"]
            }
        }
    ]
    
    integration_context = {
        "vendor_name": "Google Workspace",
        "okta_domain": "company.okta.com",
        "app_name": "Google Workspace SAML",
        "dry_run": False,
        "job_id": "test-job-123"
    }
    
    # Test the approval service
    approval_service = PlanApprovalService()
    
    try:
        approved_steps, user_approved = await approval_service.request_plan_approval(
            raw_plan=sample_plan,
            integration_context=integration_context
        )
        
        print(f"\n✅ Plan approval completed!")
        print(f"User approved: {user_approved}")
        print(f"Approved steps: {len(approved_steps)}")
        
        if approved_steps:
            print("\nApproved steps:")
            for i, step in enumerate(approved_steps, 1):
                print(f"{i}. {step.name} ({step.tool})")
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting Plan Approval Test...\n")
    asyncio.run(test_plan_approval())
    print("\n✅ Plan approval test completed!")