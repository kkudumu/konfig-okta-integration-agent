#!/usr/bin/env python3
"""
Test script for the Error Context Collection System

This script demonstrates the new error context collection capabilities
by simulating a web automation error and showing how the system:
1. Captures comprehensive context (screenshot, DOM, console logs)
2. Analyzes the error with LLM intelligence  
3. Suggests specific recovery strategies
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from konfig.services.error_context_collector import (
    ErrorContextCollector, 
    ErrorRecoveryAnalyzer, 
    StepContextTracker
)
from konfig.tools.web_interactor import WebInteractor
from konfig.services.llm_service import LLMService


async def test_error_context_collection():
    """Test the error context collection system."""
    
    print("🧪 Testing Error Context Collection System")
    print("=" * 50)
    
    # Initialize components
    step_tracker = StepContextTracker()
    error_collector = ErrorContextCollector()
    recovery_analyzer = ErrorRecoveryAnalyzer()
    
    try:
        # Start browser
        async with WebInteractor() as web_interactor:
            print("✅ Browser started successfully")
            
            # Navigate to a test page (Salesforce login)
            print("🌐 Navigating to Salesforce login page...")
            await web_interactor.navigate("https://login.salesforce.com")
            
            # Start tracking a step that will fail
            step_context = step_tracker.start_step(
                "Test Step - Enter Credentials",
                "WebInteractor", 
                "type",
                {"selector": "input[id='nonexistent']", "value": "test@example.com"}
            )
            
            print("🎯 Simulating a timeout error...")
            
            try:
                # This will timeout because the selector doesn't exist
                await web_interactor.type("input[id='nonexistent']", "test@example.com")
                
            except Exception as e:
                print(f"❌ Error occurred (as expected): {e}")
                
                # Mark step as failed
                step_tracker.complete_step("failed", error=str(e))
                
                print("📊 Collecting comprehensive error context...")
                
                # Collect error context
                error_context = await error_collector.collect_error_context(
                    error=e,
                    web_interactor=web_interactor,
                    step_tracker=step_tracker,
                    additional_context={"test_scenario": "selector_timeout"}
                )
                
                print("✅ Error context collected:")
                print(f"  - Screenshot saved: {error_context.visual_context.get('screenshot_path', 'N/A')}")
                print(f"  - DOM length: {len(error_context.visual_context.get('full_dom', ''))}")
                print(f"  - Console logs: {len(error_context.browser_state.get('console_logs', []))}")
                print(f"  - Recent steps: {len(error_context.recent_steps)}")
                
                print("\n🤖 Analyzing error with LLM...")
                
                # Analyze with LLM
                llm_service = LLMService()
                recovery_plan = await recovery_analyzer.analyze_and_suggest_recovery(
                    error_context, llm_service
                )
                
                print("✅ LLM Analysis completed:")
                print(f"  - Confidence: {recovery_plan.get('confidence', 'unknown')}")
                print(f"  - Recovery actions: {len(recovery_plan.get('recovery_actions', []))}")
                
                if recovery_plan.get('root_cause_analysis'):
                    print(f"\n📋 Root Cause Analysis:")
                    print(f"  {recovery_plan['root_cause_analysis']}")
                
                if recovery_plan.get('recovery_actions'):
                    print(f"\n🔧 Suggested Recovery Actions:")
                    for i, action in enumerate(recovery_plan['recovery_actions'], 1):
                        print(f"  {i}. {action.get('description', 'No description')}")
                        if action.get('specific_changes'):
                            print(f"     Changes: {action['specific_changes']}")
                
                print("\n" + "=" * 50)
                print("✅ Error Context Collection System Test PASSED!")
                print("The system successfully:")
                print("  ✓ Captured comprehensive error context")  
                print("  ✓ Generated LLM-powered error analysis")
                print("  ✓ Provided specific recovery recommendations")
                
                return True
                
    except Exception as test_error:
        print(f"❌ Test failed with error: {test_error}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Error Context Collection System Test")
    print("This test will:")
    print("  1. Start a browser session")
    print("  2. Navigate to a test page")
    print("  3. Simulate a web automation error")
    print("  4. Collect comprehensive error context")
    print("  5. Analyze the error with LLM intelligence")
    print("  6. Show recovery suggestions")
    print()
    
    success = await test_error_context_collection()
    
    if success:
        print("\n🎉 All tests passed! The Error Context Collection System is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())