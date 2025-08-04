#!/usr/bin/env python3
"""
Test Salesforce authentication and recovery system.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_salesforce_auth():
    """Test Salesforce authentication."""
    print("🧪 Testing Salesforce Authentication System")
    print("=" * 50)
    
    try:
        from konfig.orchestrator.mvp_orchestrator import MVPOrchestrator
        from konfig.tools.web_interactor import WebInteractor
        
        orchestrator = MVPOrchestrator()
        print("✅ MVPOrchestrator instantiated")
        
        # Test authentication directly
        async with WebInteractor() as web_interactor:
            print("✅ Browser started")
            
            # Navigate to Salesforce login
            print("🌐 Navigating to Salesforce login page...")
            await web_interactor.navigate("https://login.salesforce.com")
            
            # Test the authentication method
            working_memory = {}
            success = await orchestrator._attempt_authentication(
                web_interactor, working_memory, dry_run=False
            )
            
            if success:
                print("✅ Authentication successful!")
                print(f"   Working memory: {working_memory}")
                
                # Check where we ended up
                current_url = await web_interactor.get_current_url()
                current_title = await web_interactor.get_page_title()
                print(f"   Current URL: {current_url}")
                print(f"   Current Title: {current_title}")
                
                # Take a screenshot to verify
                screenshot_path = await web_interactor.take_screenshot()
                print(f"   Screenshot saved: {screenshot_path}")
            else:
                print("❌ Authentication failed!")
                
                # Get error details
                current_url = await web_interactor.get_current_url()
                current_title = await web_interactor.get_page_title()
                print(f"   Still on URL: {current_url}")
                print(f"   Page Title: {current_title}")
                
                # Take a screenshot of the failure
                screenshot_path = await web_interactor.take_screenshot()
                print(f"   Screenshot saved: {screenshot_path}")
            
            # Keep browser open for a moment to see results
            print("\n⏳ Keeping browser open for 5 seconds...")
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("📋 Prerequisites:")
    print("   - VENDOR_USERNAME and VENDOR_PASSWORD must be set in .env")
    print("   - Credentials should be valid Salesforce admin credentials")
    print("")
    
    asyncio.run(test_salesforce_auth())