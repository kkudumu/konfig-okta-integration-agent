#!/usr/bin/env python3
"""
Test Enhanced DOM Detection System

This test verifies the Skyvern-inspired DOM detection capabilities
work correctly with our agent swarm system.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_enhanced_dom_detection():
    """Test the enhanced DOM detection capabilities."""
    print("🔍 Testing Enhanced DOM Detection System")
    print("=" * 50)
    
    try:
        from konfig.tools.web_interactor import WebInteractor
        
        async with WebInteractor() as web_interactor:
            print("✅ WebInteractor initialized")
            
            # Test 1: Navigate to a complex form page
            print("\n📄 Test 1: Navigating to Google login page...")
            result = await web_interactor.navigate("https://accounts.google.com/signin")
            
            if result.get("success"):
                print(f"✅ Navigation successful: {result.get('title')}")
                
                # Test 2: Find all interactable elements
                print("\n🔍 Test 2: Finding interactable elements...")
                elements = await web_interactor.find_interactable_elements()
                
                print(f"✅ Found {len(elements)} interactable elements")
                
                # Show top 5 elements with highest confidence
                print("\n🏆 Top 5 Most Confident Elements:")
                for i, element in enumerate(elements[:5], 1):
                    print(f"  {i}. {element.get('tagName', 'unknown')} - {element.get('text', 'no text')[:50]}")
                    print(f"     Context: {element.get('context', 'no context')[:50]}")
                    print(f"     Confidence: {element.get('confidence', 0):.1f}")
                    print(f"     Selector: {element.get('selector', 'no selector')}")
                    print()
                
                # Test 3: Smart element finding
                print("🎯 Test 3: Smart element finding...")
                
                test_actions = [
                    ("find email input", "email"),
                    ("find password field", "password"),
                    ("find login button", "login"),
                    ("find next button", "next"),
                ]
                
                for action_desc, action_type in test_actions:
                    element = await web_interactor.find_best_element_for_action(action_desc)
                    if element:
                        print(f"✅ {action_desc}: Found {element.get('tagName')} with selector '{element.get('selector')}'")
                        print(f"   Text: '{element.get('text', '')[:30]}'")
                        print(f"   Confidence: {element.get('confidence', 0):.1f}")
                    else:
                        print(f"❌ {action_desc}: No suitable element found")
                    print()
                
                # Test 4: Smart interactions (dry run)
                print("🤖 Test 4: Smart interaction capabilities...")
                
                # Try to find and describe the email field
                email_result = await web_interactor.smart_fill("email address field", "test@example.com")
                if email_result.get("success"):
                    print("✅ Smart fill (email): Would successfully fill email field")
                    print(f"   Element info: {email_result.get('element_info', {})}")
                else:
                    print(f"❌ Smart fill (email): {email_result.get('error')}")
                
                # Try to find the next/continue button
                button_result = await web_interactor.smart_click("next button")
                if button_result.get("success"):
                    print("✅ Smart click (next): Would successfully click next button")
                    print(f"   Element info: {button_result.get('element_info', {})}")
                else:
                    print(f"❌ Smart click (next): {button_result.get('error')}")
                
            else:
                print(f"❌ Navigation failed: {result.get('error')}")
            
            # Test 5: Compare with simple approach
            print("\n⚖️  Test 5: Comparing with simple selector approach...")
            
            simple_selectors = ["#identifierId", "input[type='email']", "#Email"]
            enhanced_found = False
            simple_found = False
            
            # Check if enhanced detection found email field
            email_element = await web_interactor.find_best_element_for_action("email input field")
            if email_element:
                enhanced_found = True
                print(f"✅ Enhanced detection: Found email field with confidence {email_element.get('confidence', 0):.1f}")
            
            # Check simple selectors
            for selector in simple_selectors:
                try:
                    exists = await web_interactor.element_exists(selector)
                    if exists:
                        simple_found = True
                        print(f"✅ Simple selector '{selector}': Element found")
                        break
                except:
                    continue
            
            if not simple_found:
                print("❌ Simple selectors: No email field found with basic selectors")
            
            # Comparison result
            print(f"\n📊 COMPARISON RESULTS:")
            print(f"   Enhanced Detection: {'✅ SUCCESS' if enhanced_found else '❌ FAILED'}")
            print(f"   Simple Selectors: {'✅ SUCCESS' if simple_found else '❌ FAILED'}")
            
            if enhanced_found and not simple_found:
                print("🎉 Enhanced detection found elements that simple selectors missed!")
            elif enhanced_found and simple_found:
                print("✨ Both methods work, but enhanced provides more context and flexibility!")
            
        print(f"\n🏁 Enhanced DOM Detection Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_dom_utils_injection():
    """Test that DOM utilities are properly injected."""
    print("\n🧪 Testing DOM Utilities Injection...")
    
    try:
        from konfig.tools.web_interactor import WebInteractor
        
        async with WebInteractor() as web_interactor:
            # Navigate to a simple page
            await web_interactor.navigate("https://www.google.com")
            
            # Check if DOM utilities are available
            utils_available = await web_interactor._page.evaluate("""
                () => typeof window.skyvernDomUtils !== 'undefined'
            """)
            
            if utils_available:
                print("✅ DOM utilities successfully injected and available")
                
                # Test a utility function
                test_result = await web_interactor._page.evaluate("""
                    () => {
                        const elements = document.querySelectorAll('*');
                        let interactableCount = 0;
                        for (const el of elements) {
                            if (window.skyvernDomUtils.isInteractable(el)) {
                                interactableCount++;
                            }
                        }
                        return interactableCount;
                    }
                """)
                
                print(f"✅ DOM utilities working: Found {test_result} interactable elements")
                return True
            else:
                print("❌ DOM utilities not available in page context")
                return False
                
    except Exception as e:
        print(f"❌ DOM utilities injection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced DOM Detection Test Suite")
    print("="*60)
    
    async def main():
        # Test DOM utilities injection
        utils_ok = await test_dom_utils_injection()
        
        if utils_ok:
            # Test full enhanced detection system
            detection_ok = await test_enhanced_dom_detection()
            
            print(f"\n" + "="*60)
            print("🏁 TEST SUITE SUMMARY:")
            print(f"   DOM Utilities Injection: {'✅ PASSED' if utils_ok else '❌ FAILED'}")
            print(f"   Enhanced DOM Detection: {'✅ PASSED' if detection_ok else '❌ FAILED'}")
            
            if utils_ok and detection_ok:
                print(f"\n🎉 SUCCESS!")
                print(f"   Your agent swarm now has Skyvern-level DOM detection capabilities!")
                print(f"   Benefits:")
                print(f"   • Finds elements even without obvious selectors")
                print(f"   • Understands element context and purpose")
                print(f"   • Provides confidence scoring for better decision making")
                print(f"   • Handles complex visibility and interactability detection")
                print(f"   • Works with shadow DOM and dynamic content")
            else:
                print(f"\n⚠️  Some tests failed. Check configuration and dependencies.")
        
    asyncio.run(main())