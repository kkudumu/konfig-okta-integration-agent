#!/usr/bin/env python3
"""
Simple test script to debug screenshot functionality.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the konfig package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from konfig.tools.web_interactor import WebInteractor


async def test_screenshot():
    """Test the screenshot functionality."""
    print("🔧 Testing screenshot functionality...")
    
    web_interactor = None
    try:
        # Initialize web interactor
        web_interactor = WebInteractor()
        
        print("✅ WebInteractor initialized")
        
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
        
        # Wait a moment for page to load
        await asyncio.sleep(2)
        
        # Test screenshot with timeout
        print("📸 Taking screenshot (with 10s timeout)...")
        
        try:
            screenshot_path = await asyncio.wait_for(
                web_interactor.take_screenshot(),
                timeout=10.0
            )
            print(f"✅ Screenshot saved to: {screenshot_path}")
            
            # Check if file exists and has content
            if Path(screenshot_path).exists():
                file_size = Path(screenshot_path).stat().st_size
                print(f"✅ Screenshot file exists, size: {file_size} bytes")
                
                if file_size > 1000:  # Reasonable size check
                    print("✅ Screenshot appears to be valid")
                else:
                    print("⚠️  Screenshot file is very small, might be corrupted")
            else:
                print("❌ Screenshot file does not exist")
                
        except asyncio.TimeoutError:
            print("❌ Screenshot timed out after 10 seconds")
        except Exception as e:
            print(f"❌ Screenshot failed with error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
        
        # Test screenshot with full_page=True
        print("📸 Taking full page screenshot...")
        try:
            full_screenshot_path = await asyncio.wait_for(
                web_interactor.take_screenshot(full_page=True),
                timeout=15.0
            )
            print(f"✅ Full page screenshot saved to: {full_screenshot_path}")
        except Exception as e:
            print(f"❌ Full page screenshot failed: {e}")
        
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


async def test_screenshot_on_complex_page():
    """Test screenshot on a more complex page that might have font loading issues."""
    print("\n🔧 Testing screenshot on complex page...")
    
    web_interactor = None
    try:
        web_interactor = WebInteractor()
        await web_interactor.start_browser()
        
        # Navigate to Google (might have font loading)
        result = await web_interactor.navigate("https://www.google.com")
        
        if result.get("success"):
            print("✅ Successfully navigated to Google")
            
            # Wait for page to fully load
            await asyncio.sleep(3)
            
            # Test screenshot
            print("📸 Taking screenshot of Google...")
            try:
                screenshot_path = await asyncio.wait_for(
                    web_interactor.take_screenshot(),
                    timeout=10.0
                )
                print(f"✅ Google screenshot saved to: {screenshot_path}")
            except asyncio.TimeoutError:
                print("❌ Google screenshot timed out - likely font loading issue")
            except Exception as e:
                print(f"❌ Google screenshot failed: {e}")
        
    except Exception as e:
        print(f"❌ Complex page test failed: {e}")
    
    finally:
        if web_interactor:
            try:
                await web_interactor.close_browser()
            except Exception:
                pass


if __name__ == "__main__":
    print("🚀 Starting screenshot debug tests...\n")
    
    # Run basic test
    asyncio.run(test_screenshot())
    
    # Run complex page test
    asyncio.run(test_screenshot_on_complex_page())
    
    print("\n✅ Screenshot debug tests completed!")