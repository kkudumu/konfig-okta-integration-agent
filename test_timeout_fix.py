#!/usr/bin/env python3
"""
Test script to verify timeout fixes are working.
"""

import asyncio
import sys
from pathlib import Path

# Add the konfig package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from konfig.tools.web_interactor import WebInteractor


async def test_timeout_behavior():
    """Test that timeouts are now 10 seconds instead of 30."""
    print("🔧 Testing timeout behavior...")
    
    web_interactor = None
    try:
        # Initialize web interactor
        web_interactor = WebInteractor()
        print(f"✅ WebInteractor timeout setting: {web_interactor.timeout}ms")
        
        # Start browser
        await web_interactor.start_browser()
        print("✅ Browser started")
        
        # Navigate to a simple page
        test_url = "https://httpbin.org/html"
        result = await web_interactor.navigate(test_url)
        
        if result.get("success"):
            print(f"✅ Successfully navigated to {test_url}")
        else:
            print(f"❌ Failed to navigate: {result}")
            return
        
        # Test clicking a non-existent element (should timeout in 10s not 30s)
        print("⏱️  Testing timeout on non-existent element (should timeout in ~10s)...")
        
        import time
        start_time = time.time()
        
        try:
            # This should timeout quickly now
            await web_interactor.click("input[name='nonexistent-field']")
            print("❌ Unexpected success - element should not exist")
        except Exception as e:
            end_time = time.time()
            timeout_duration = end_time - start_time
            
            print(f"✅ Timeout occurred after {timeout_duration:.1f} seconds")
            
            if timeout_duration < 15:  # Should be ~10s, allowing some buffer
                print("✅ Fast timeout confirmed - fix is working!")
            else:
                print("❌ Timeout still too slow - fix may not be working")
        
        # Test with a generic selector that would cause problems
        print("⚠️  Testing generic selector behavior...")
        start_time = time.time()
        
        try:
            # Generic selector that could match multiple elements or none
            await web_interactor.click("input")  # This is what was causing the 30s timeouts
        except Exception as e:
            end_time = time.time()
            timeout_duration = end_time - start_time
            print(f"✅ Generic selector timed out after {timeout_duration:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if web_interactor:
            try:
                await web_interactor.close_browser()
                print("✅ Browser closed")
            except Exception as e:
                print(f"⚠️  Error closing browser: {e}")


if __name__ == "__main__":
    print("🚀 Starting timeout fix verification...\n")
    asyncio.run(test_timeout_behavior())
    print("\n✅ Timeout test completed!")