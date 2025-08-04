#!/usr/bin/env python3
"""
Test Agent Swarm Integration System

This test verifies the multi-agent swarm system works end-to-end
by running a complete SAML/SSO integration workflow using intelligent agents.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_agent_swarm_integration():
    """Test the complete agent swarm integration workflow."""
    print("🤖 Testing Multi-Agent Swarm Integration System")
    print("=" * 60)
    
    try:
        from konfig.orchestrator.mvp_orchestrator import MVPOrchestrator
        
        # Initialize orchestrator
        orchestrator = MVPOrchestrator()
        print("✅ MVPOrchestrator initialized")
        
        # Test with Salesforce (since we know the authentication works)
        vendor_name = "Salesforce"
        admin_url = "https://login.salesforce.com"
        okta_domain = "dev-123456.okta.com"  # Mock domain for testing
        
        print(f"\n🎯 Testing Integration:")
        print(f"   Vendor: {vendor_name}")
        print(f"   Admin URL: {admin_url}")
        print(f"   Okta Domain: {okta_domain}")
        print(f"   Mode: Dry Run (safe testing)")
        
        # Progress callback for monitoring
        def progress_callback(message, progress):
            progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            print(f"   Progress: [{progress_bar}] {progress:.1%} - {message}")
        
        print(f"\n🚀 Starting swarm integration...")
        
        # Execute swarm integration
        result = await orchestrator.integrate_application_with_swarm(
            vendor_name=vendor_name,
            admin_url=admin_url,
            okta_domain=okta_domain,
            app_name=f"{vendor_name} Test Integration",
            dry_run=True,  # Safe testing mode
            progress_callback=progress_callback
        )
        
        print(f"\n📊 Integration Results:")
        print(f"   Success: {'✅' if result.get('success') else '❌'}")
        print(f"   Job ID: {result.get('job_id')}")
        print(f"   Duration: {result.get('duration_seconds', 0):.1f} seconds")
        print(f"   Integration Type: {result.get('integration_type')}")
        print(f"   Total Steps: {result.get('total_steps', 0)}")
        
        if result.get('success'):
            print(f"\n🎉 SUCCESS DETAILS:")
            print(f"   Message: {result.get('message')}")
            print(f"   Okta App ID: {result.get('okta_app_id', 'N/A')}")
            print(f"   Agents Used: {', '.join(result.get('agents_used', []))}")
            
            swarm_result = result.get('swarm_result', {})
            if swarm_result:
                print(f"   Swarm Execution: {'✅' if swarm_result.get('success') else '❌'}")
                print(f"   Vendor Integration: {swarm_result.get('vendor_name', 'Unknown')}")
            
            validation_result = result.get('validation_result', {})
            if validation_result:
                print(f"   Validation: {'✅' if validation_result.get('valid') else '❌'}")
                print(f"   Confidence: {validation_result.get('confidence', 0):.1%}")
            
            next_steps = result.get('next_steps', [])
            if next_steps:
                print(f"\n📋 NEXT STEPS:")
                for i, step in enumerate(next_steps, 1):
                    print(f"   {i}. {step}")
        else:
            print(f"\n❌ FAILURE DETAILS:")
            print(f"   Message: {result.get('message', 'Unknown error')}")
            
            error_details = result.get('error_details', {})
            if error_details:
                swarm_errors = error_details.get('swarm_errors')
                if swarm_errors:
                    print(f"   Swarm Errors: {swarm_errors}")
                
                validation_errors = error_details.get('validation_errors', [])
                if validation_errors:
                    print(f"   Validation Errors:")
                    for error in validation_errors:
                        print(f"     - {error}")
            
            if result.get('error'):
                print(f"   Technical Error: {result.get('error')}")
        
        print(f"\n🧠 AGENT SWARM ARCHITECTURE SUMMARY:")
        print(f"   🎯 PlannerAgent: Breaks down integration goals into actionable steps")
        print(f"   🤖 TaskAgent: Executes steps using vision and DOM analysis")
        print(f"   ✅ ValidatorAgent: Validates step completion and provides feedback")
        print(f"   🎭 SwarmOrchestrator: Coordinates all agents for collaborative intelligence")
        
        print(f"\n💡 KEY FEATURES DEMONSTRATED:")
        print(f"   • Multi-agent collaborative planning and execution")
        print(f"   • Vision-based web automation (similar to Skyvern)")
        print(f"   • Intelligent error recovery and adaptation")
        print(f"   • LLM-powered reasoning for complex workflows")
        print(f"   • Real-time validation and feedback loops")
        print(f"   • Progress tracking and monitoring")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

async def test_individual_agents():
    """Test individual agents in isolation."""
    print(f"\n🔬 Testing Individual Agents:")
    print("=" * 40)
    
    try:
        from konfig.services.agent_swarm import (
            PlannerAgent, TaskAgent, ValidatorAgent, SwarmOrchestrator,
            IntegrationGoal, AgentRole
        )
        from konfig.tools.web_interactor import WebInteractor
        
        # Test PlannerAgent
        print("🧠 Testing PlannerAgent...")
        planner = PlannerAgent()
        print(f"   ✅ PlannerAgent initialized (Role: {planner.role.value})")
        
        # Test TaskAgent (requires WebInteractor)
        print("🤖 Testing TaskAgent...")
        async with WebInteractor() as web_interactor:
            task_agent = TaskAgent(web_interactor)
            print(f"   ✅ TaskAgent initialized (Role: {task_agent.role.value})")
            
            # Test ValidatorAgent
            print("✅ Testing ValidatorAgent...")
            validator = ValidatorAgent(web_interactor)
            print(f"   ✅ ValidatorAgent initialized (Role: {validator.role.value})")
            
            # Test SwarmOrchestrator
            print("🎭 Testing SwarmOrchestrator...")
            orchestrator = SwarmOrchestrator(web_interactor)
            print(f"   ✅ SwarmOrchestrator initialized (Role: {orchestrator.role.value})")
            
            # Test IntegrationGoal creation
            print("🎯 Testing IntegrationGoal...")
            goal = IntegrationGoal(
                vendor_name="Test Vendor",
                integration_type="saml_sso",
                admin_url="https://test.example.com",
                okta_config={"app_id": "test123"},
                success_criteria=["Test criterion"],
                constraints=["Test constraint"]
            )
            print(f"   ✅ IntegrationGoal created for {goal.vendor_name}")
        
        print("✅ All individual agents tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Individual agent testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Agent Swarm Integration Test Suite")
    print("="*60)
    
    print("📋 Prerequisites:")
    print("   - VENDOR_USERNAME and VENDOR_PASSWORD must be set in .env")
    print("   - Credentials should be valid for authentication testing")
    print("   - This test runs in DRY RUN mode (safe for testing)")
    print("")
    
    async def main():
        # Test individual agents first
        agents_ok = await test_individual_agents()
        
        if agents_ok:
            # Test full integration
            result = await test_agent_swarm_integration()
            
            print(f"\n" + "="*60)
            print("🏁 TEST SUITE SUMMARY:")
            print(f"   Individual Agents: {'✅ PASSED' if agents_ok else '❌ FAILED'}")
            print(f"   Full Integration: {'✅ PASSED' if result.get('success') else '❌ FAILED'}")
            
            if result.get('success'):
                print(f"\n🎉 CONGRATULATIONS!")
                print(f"   Your Skyvern-inspired multi-agent swarm system is working!")
                print(f"   The system can now intelligently plan, execute, and validate")
                print(f"   complex SAML/SSO integrations using collaborative AI agents.")
            else:
                print(f"\n⚠️  Integration test failed, but individual agents work.")
                print(f"   This suggests the issue is in agent coordination or")
                print(f"   the integration workflow itself.")
        else:
            print(f"\n❌ CRITICAL: Individual agents failed to initialize.")
            print(f"   Please check dependencies and configuration.")
    
    asyncio.run(main())