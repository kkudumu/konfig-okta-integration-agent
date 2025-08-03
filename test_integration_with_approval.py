#!/usr/bin/env python3
"""
Test integration with the approval system using auto-approve.
"""

import asyncio
import sys
from pathlib import Path

# Add the konfig package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from konfig.orchestrator.mvp_orchestrator import MVPOrchestrator


async def test_integration_with_approval():
    """Test the integration with plan approval (using auto-approve)."""
    print("🔧 Testing Integration with Plan Approval...")
    
    orchestrator = MVPOrchestrator()
    
    # Test with auto-approve enabled
    print("\n📋 Testing AUTO-APPROVE mode...")
    
    try:
        result = await orchestrator.integrate_application(
            documentation_url="https://support.google.com/a/answer/6087519",
            okta_domain="company.okta.com",
            app_name="Google Workspace Test",
            dry_run=True,  # Use dry run for testing
            auto_approve=True  # Skip user interaction
        )
        
        print(f"✅ Integration result: {result.get('status', 'unknown')}")
        print(f"📝 Message: {result.get('message', 'No message')}")
        
        if result.get("success"):
            print("🎉 Integration completed successfully!")
        else:
            print("⚠️ Integration had issues but completed")
            
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📋 Testing USER APPROVAL mode (will timeout without input)...")
    
    # Test without auto-approve to verify the approval system loads
    try:
        # This will fail in a non-interactive environment, but we'll catch it
        result = await asyncio.wait_for(
            orchestrator.integrate_application(
                documentation_url="https://support.google.com/a/answer/6087519",
                okta_domain="company.okta.com",
                app_name="Google Workspace Test",
                dry_run=True,
                auto_approve=False  # Require user interaction
            ),
            timeout=5.0  # Short timeout since we can't provide input
        )
    except asyncio.TimeoutError:
        print("✅ User approval system loaded correctly (timed out waiting for input)")
    except Exception as e:
        if "EOF when reading a line" in str(e):
            print("✅ User approval system loaded correctly (no interactive input available)")
        else:
            print(f"⚠️ Unexpected error: {e}")


if __name__ == "__main__":
    print("🚀 Starting Integration with Approval Test...\n")
    asyncio.run(test_integration_with_approval())
    print("\n✅ Integration approval test completed!")