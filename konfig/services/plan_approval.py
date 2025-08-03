"""
Plan Approval Service

Handles user approval of integration plans before execution.
Displays detailed step-by-step plans and allows user to accept/reject individual steps.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm, Prompt

from konfig.utils.logging import LoggingMixin


@dataclass
class PlanStep:
    """Represents a single step in the integration plan."""
    id: str
    name: str
    description: str
    tool: str
    action: str
    params: Dict[str, Any]
    category: str  # 'okta', 'vendor', 'verification', 'cleanup'
    risk_level: str  # 'low', 'medium', 'high'
    approved: bool = False
    skipped: bool = False
    required: bool = True


class PlanApprovalService(LoggingMixin):
    """
    Service for displaying integration plans to users and gathering approval.
    """
    
    def __init__(self):
        super().__init__()
        self.setup_logging("plan_approval")
        self.console = Console()
    
    async def request_plan_approval(
        self,
        raw_plan: List[Dict[str, Any]],
        integration_context: Dict[str, Any]
    ) -> Tuple[List[PlanStep], bool]:
        """
        Display the integration plan to the user and request approval.
        
        Args:
            raw_plan: Raw plan from intelligent planner
            integration_context: Context about the integration
            
        Returns:
            Tuple of (approved_steps, user_approved_execution)
        """
        self.logger.info("Requesting user approval for integration plan")
        
        # Convert raw plan to structured steps
        structured_steps = self._structure_plan(raw_plan, integration_context)
        
        # Display the plan
        self._display_integration_overview(integration_context)
        self._display_plan_summary(structured_steps)
        
        # Get user approval
        user_wants_to_proceed = self._get_initial_approval()
        
        if not user_wants_to_proceed:
            self.logger.info("User rejected the integration plan")
            return [], False
        
        # Allow step-by-step review if requested
        if self._ask_for_detailed_review():
            structured_steps = self._review_individual_steps(structured_steps)
        else:
            # Auto-approve all steps
            for step in structured_steps:
                step.approved = True
        
        approved_steps = [step for step in structured_steps if step.approved and not step.skipped]
        
        self.logger.info(f"Plan approved with {len(approved_steps)} steps")
        return approved_steps, True
    
    def _structure_plan(
        self,
        raw_plan: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[PlanStep]:
        """Convert raw plan steps into structured PlanStep objects."""
        
        structured_steps = []
        
        for i, step in enumerate(raw_plan):
            # Determine category based on step content
            category = self._categorize_step(step)
            
            # Determine risk level
            risk_level = self._assess_risk_level(step)
            
            # Determine if step is required
            required = self._is_step_required(step)
            
            structured_step = PlanStep(
                id=f"step_{i+1}",
                name=step.get("name", f"Step {i+1}"),
                description=self._generate_step_description(step),
                tool=step.get("tool", "Unknown"),
                action=step.get("action", "unknown"),
                params=step.get("params", {}),
                category=category,
                risk_level=risk_level,
                required=required
            )
            
            structured_steps.append(structured_step)
        
        return structured_steps
    
    def _categorize_step(self, step: Dict[str, Any]) -> str:
        """Categorize a step based on its content."""
        name = step.get("name", "").lower()
        tool = step.get("tool", "").lower()
        params = step.get("params", {})
        
        # Okta-related steps
        if "okta" in name or "tile" in name or "application" in name:
            return "okta"
        
        # Browser/vendor steps
        if "browser" in tool or "web" in tool or any(
            keyword in name for keyword in ["navigate", "login", "configure", "sso", "saml"]
        ):
            return "vendor"
        
        # Verification steps
        if any(keyword in name for keyword in ["test", "verify", "validate", "check"]):
            return "verification"
        
        # Default
        return "general"
    
    def _assess_risk_level(self, step: Dict[str, Any]) -> str:
        """Assess the risk level of a step."""
        name = step.get("name", "").lower()
        action = step.get("action", "").lower()
        
        # High risk actions
        high_risk_keywords = ["delete", "remove", "disable", "reset", "clear"]
        if any(keyword in name for keyword in high_risk_keywords):
            return "high"
        
        # Medium risk actions
        medium_risk_keywords = ["create", "configure", "update", "modify", "save"]
        if any(keyword in name for keyword in medium_risk_keywords):
            return "medium"
        
        # Low risk (read-only, navigation)
        return "low"
    
    def _is_step_required(self, step: Dict[str, Any]) -> bool:
        """Determine if a step is required for the integration to work."""
        name = step.get("name", "").lower()
        
        # Optional steps (usually testing/verification)
        optional_keywords = ["test", "verify", "validate", "optional"]
        if any(keyword in name for keyword in optional_keywords):
            return False
        
        return True
    
    def _generate_step_description(self, step: Dict[str, Any]) -> str:
        """Generate a human-readable description of what the step does."""
        name = step.get("name", "")
        tool = step.get("tool", "")
        action = step.get("action", "")
        params = step.get("params", {})
        
        # Special handling for specific step names to provide better context
        if "Create SAML Application" in name:
            app_label = params.get("label", "Google Workspace")
            return f"Create a new SAML 2.0 application in Okta named '{app_label}' with initial configuration"
        
        elif "Retrieve Okta" in name and "Metadata" in name:
            return "Extract SAML metadata (Entity ID, SSO URL, Certificate) from the newly created Okta app"
        
        elif "Fill SSO Profile" in name or "SSO Profile Values" in name:
            return "Configure Google Workspace with Okta's SAML metadata (SSO URL, Entity ID, Certificate)"
        
        elif "Security Menu" in name:
            return "Access the Security settings section in Google Admin Console"
        
        elif "Set up SSO" in name:
            return "Open the Single Sign-On configuration page in Google Workspace"
        
        elif "Save Button" in name and "vendor" in step.get("category", ""):
            return "Save the SAML configuration changes in Google Workspace"
        
        elif "Test SSO" in name:
            return "Verify the SSO integration by attempting a test login through Okta"
        
        # Enhanced tool-specific descriptions
        if tool == "OktaAPI":
            if action == "create_application":
                app_type = params.get("type", "SAML 2.0")
                return f"Create new {app_type} application in Okta with vendor-specific settings"
            elif action == "configure_saml":
                return "Configure SAML assertions, attributes, and SSO URLs in Okta"
            elif action == "get_metadata":
                return "Retrieve SAML metadata (Entity ID, SSO URL, X.509 Certificate) from Okta"
            elif action == "enable_application":
                return "Activate the Okta application and make it available to assigned users"
        
        elif tool == "WebInteractor":
            if action == "navigate":
                url = params.get("url", "")
                if "admin.google.com" in url:
                    return "Open Google Workspace Admin Console in browser"
                elif "accounts.google.com" in url:
                    return "Navigate to Google login page for SSO testing"
                elif url:
                    # Extract domain for cleaner display
                    domain = url.split('/')[2] if url.startswith('http') else url
                    return f"Navigate to {domain}"
                return "Navigate to vendor admin portal"
            
            elif action == "click":
                selector = params.get("selector", "")
                text = params.get("text_content", "")
                
                # Provide context based on common UI patterns
                if "security" in text.lower() or "security" in name.lower():
                    return "Click on 'Security' section to access authentication settings"
                elif "sso" in text.lower() or "single sign" in text.lower():
                    return "Click on SSO/Single Sign-On option to configure SAML settings"
                elif "save" in text.lower():
                    return "Click Save to apply and persist the SAML configuration changes"
                elif text:
                    return f"Click on '{text}' button/link in the interface"
                elif selector:
                    # Try to make selector more readable
                    if "button" in selector:
                        return "Click on the relevant button to proceed"
                    elif "link" in selector or "a" in selector:
                        return "Click on the navigation link"
                return "Click on the interface element to proceed"
            
            elif action == "type" or action == "fill":
                selector = params.get("selector", "field")
                text = params.get("text", "")
                
                # Provide meaningful context for form fields
                if "entity" in selector.lower() or "issuer" in selector.lower():
                    return "Enter Okta's Entity ID/Issuer URL for SAML authentication"
                elif "sso" in selector.lower() or "signin" in selector.lower():
                    return "Enter Okta's SSO/Sign-in URL where users will be redirected"
                elif "certificate" in selector.lower():
                    return "Upload or paste Okta's X.509 certificate for signature verification"
                elif "acs" in selector.lower():
                    return "Enter the Assertion Consumer Service URL from vendor"
                elif "password" in selector.lower() or params.get("secret"):
                    return "Enter admin credentials securely (will be masked)"
                elif text and len(text) < 50 and not params.get("secret"):
                    return f"Enter '{text}' into the form field"
                else:
                    return f"Fill in the required configuration value"
        
        # Fallback with more context
        if name:
            # Try to make the name more descriptive if it's generic
            if "click" in name.lower() and "button" in name.lower():
                return "Click the button to proceed with configuration"
            return name
        
        return f"Execute {action} operation{' using ' + tool if tool else ''}"
    
    def _display_integration_overview(self, context: Dict[str, Any]):
        """Display overview of the integration being performed."""
        vendor_name = context.get("vendor_name", "Unknown Vendor")
        okta_domain = context.get("okta_domain", "your-org.okta.com")
        
        overview = f"""
[bold blue]🔗 SSO Integration Plan[/bold blue]

[yellow]Vendor:[/yellow] {vendor_name}
[yellow]Okta Domain:[/yellow] {okta_domain}
[yellow]Integration Type:[/yellow] SAML 2.0 Single Sign-On

[dim]This plan will configure SSO between your Okta organization and {vendor_name}.
Both systems will be modified to enable seamless user authentication.[/dim]
"""
        
        self.console.print(Panel(overview, title="Integration Overview", border_style="blue"))
    
    def _display_plan_summary(self, steps: List[PlanStep]):
        """Display a summary table of all planned steps."""
        
        # Create summary table
        table = Table(
            title="📋 Integration Plan Summary", 
            show_header=True, 
            header_style="bold magenta",
            show_lines=True,  # Add lines between rows for better readability
            expand=True,  # Allow table to use full terminal width
            padding=(0, 1)  # Add some padding
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Step Name", style="cyan", min_width=25)
        table.add_column("Category", justify="center", width=10)
        table.add_column("Risk", justify="center", width=8)
        table.add_column("Required", justify="center", width=8)
        table.add_column("Description", style="dim", overflow="fold")  # Allow text wrapping
        
        for i, step in enumerate(steps, 1):
            # Style risk level
            risk_style = {
                "low": "[green]LOW[/green]",
                "medium": "[yellow]MED[/yellow]",
                "high": "[red]HIGH[/red]"
            }.get(step.risk_level, step.risk_level)
            
            # Style category
            category_style = {
                "okta": "[blue]OKTA[/blue]",
                "vendor": "[purple]VENDOR[/purple]",
                "verification": "[green]VERIFY[/green]",
                "general": "[dim]GENERAL[/dim]"
            }.get(step.category, step.category.upper())
            
            # Required indicator
            required_indicator = "✓" if step.required else "○"
            
            table.add_row(
                str(i),
                step.name,
                category_style,
                risk_style,
                required_indicator,
                step.description  # Show full description
            )
        
        self.console.print(table)
        
        # Display legend
        legend = """
[dim]Legend:[/dim]
• [blue]OKTA[/blue] = Okta configuration steps
• [purple]VENDOR[/purple] = Vendor system configuration 
• [green]VERIFY[/green] = Testing/verification steps
• ✓ = Required step, ○ = Optional step
• [green]LOW[/green]/[yellow]MED[/yellow]/[red]HIGH[/red] = Risk level
"""
        self.console.print(Panel(legend, title="Legend", border_style="dim"))
    
    def _get_initial_approval(self) -> bool:
        """Get initial user approval for the overall plan."""
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]📋 PLAN APPROVAL REQUIRED[/bold cyan]")
        self.console.print("\nPlease review the integration plan above.")
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("  • Type [green]'y' or 'yes'[/green] and press Enter to [bold green]APPROVE[/bold green] and proceed")
        self.console.print("  • Type [red]'n' or 'no'[/red] and press Enter to [bold red]REJECT[/bold red] and cancel")
        self.console.print("  • Press [green]Enter[/green] alone to use the default ([bold green]approve[/bold green])")
        self.console.print("\n" + "="*60)
        
        return Confirm.ask(
            "\n[bold yellow]Do you want to proceed with this integration plan?[/bold yellow]",
            default=True
        )
    
    def _ask_for_detailed_review(self) -> bool:
        """Ask if user wants to review individual steps."""
        self.console.print("\n" + "-"*60)
        self.console.print("[bold cyan]🔍 DETAILED REVIEW OPTION[/bold cyan]")
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("  • Type [green]'y' or 'yes'[/green] to review and approve/skip individual steps")
        self.console.print("  • Type [red]'n' or 'no'[/red] to approve ALL steps automatically")
        self.console.print("  • Press [green]Enter[/green] alone to use the default ([bold green]approve all[/bold green])")
        self.console.print("-"*60)
        
        return Confirm.ask(
            "\n[bold yellow]Would you like to review individual steps?[/bold yellow]",
            default=False
        )
    
    def _review_individual_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Allow user to review and approve/reject individual steps."""
        
        self.console.print(Panel(
            "[bold]Individual Step Review[/bold]\n"
            "For each step, you can:\n"
            "• [green]Approve[/green] - Include this step in execution\n"
            "• [yellow]Skip[/yellow] - Exclude this step from execution\n"
            "• [blue]View Details[/blue] - See technical details",
            title="Step Review Mode",
            border_style="yellow"
        ))
        
        for i, step in enumerate(steps, 1):
            self._review_single_step(step, i, len(steps))
        
        return steps
    
    def _review_single_step(self, step: PlanStep, step_num: int, total_steps: int):
        """Review a single step with the user."""
        
        # Display step details
        step_panel = f"""
[bold cyan]Step {step_num}/{total_steps}: {step.name}[/bold cyan]

[yellow]Description:[/yellow] {step.description}
[yellow]Tool:[/yellow] {step.tool}
[yellow]Action:[/yellow] {step.action}
[yellow]Category:[/yellow] {step.category.upper()}
[yellow]Risk Level:[/yellow] {step.risk_level.upper()}
[yellow]Required:[/yellow] {"Yes" if step.required else "No"}
"""
        
        if step.params:
            # Show sanitized params (hide sensitive data)
            sanitized_params = self._sanitize_params_for_display(step.params)
            step_panel += f"\n[yellow]Parameters:[/yellow] {json.dumps(sanitized_params, indent=2)}"
        
        self.console.print(Panel(step_panel, border_style="cyan"))
        
        # Get user decision
        if step.required:
            self.console.print("[dim]This is a required step and cannot be skipped.[/dim]")
            step.approved = True
        else:
            choices = ["approve", "skip", "details"]
            choice = Prompt.ask(
                "Decision",
                choices=choices,
                default="approve"
            )
            
            if choice == "approve":
                step.approved = True
                self.console.print("[green]✓ Step approved[/green]")
            elif choice == "skip":
                step.skipped = True
                self.console.print("[yellow]○ Step skipped[/yellow]")
            elif choice == "details":
                self._show_step_details(step)
                # Ask again after showing details
                self._review_single_step(step, step_num, total_steps)
        
        self.console.print()  # Add spacing
    
    def _sanitize_params_for_display(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from parameters for display."""
        sanitized = {}
        
        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"]):
                sanitized[key] = "***HIDDEN***"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _show_step_details(self, step: PlanStep):
        """Show detailed technical information about a step."""
        details = f"""
[bold]Technical Details for: {step.name}[/bold]

[yellow]Tool:[/yellow] {step.tool}
[yellow]Action:[/yellow] {step.action}
[yellow]Full Parameters:[/yellow]
{json.dumps(step.params, indent=2)}

[yellow]What this step does:[/yellow]
{step.description}

[yellow]Risk Assessment:[/yellow]
This step has {step.risk_level} risk level.
"""
        
        if step.risk_level == "high":
            details += "\n[red]⚠️  HIGH RISK: This step makes significant changes to your system.[/red]"
        elif step.risk_level == "medium":
            details += "\n[yellow]⚠️  MEDIUM RISK: This step will modify system configuration.[/yellow]"
        else:
            details += "\n[green]✓ LOW RISK: This step is read-only or makes minimal changes.[/green]"
        
        self.console.print(Panel(details, title="Step Details", border_style="yellow"))
    
    def display_execution_summary(self, approved_steps: List[PlanStep]):
        """Display summary of approved steps before execution begins."""
        
        if not approved_steps:
            self.console.print(Panel(
                "[red]No steps approved for execution.[/red]\n"
                "Integration cancelled.",
                title="Execution Summary",
                border_style="red"
            ))
            return
        
        # Count by category
        category_counts = {}
        for step in approved_steps:
            category_counts[step.category] = category_counts.get(step.category, 0) + 1
        
        summary = f"""
[bold green]✓ Integration Plan Approved[/bold green]

[yellow]Total Steps to Execute:[/yellow] {len(approved_steps)}

[yellow]Breakdown by Category:[/yellow]
"""
        
        for category, count in category_counts.items():
            summary += f"• {category.upper()}: {count} steps\n"
        
        summary += f"""
[dim]The integration will now begin executing the approved steps.
You can monitor progress in the logs.[/dim]
"""
        
        self.console.print(Panel(summary, title="🚀 Starting Execution", border_style="green"))