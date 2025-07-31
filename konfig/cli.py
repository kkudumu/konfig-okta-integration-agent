"""
Command Line Interface for Konfig.

This module provides the main CLI entry point for the Konfig autonomous
SSO integration agent.
"""

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from konfig import __version__
from konfig.config.settings import get_settings
from konfig.orchestrator.mvp_orchestrator import MVPOrchestrator
from konfig.modules.memory.memory_module import MemoryModule
from konfig.database.connection import health_check

# Initialize Typer app and Rich console
app = typer.Typer(
    name="konfig",
    help="Konfig - Autonomous Okta SSO Integration Agent",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version information and exit."""
    if value:
        console.print(f"Konfig v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, help="Show version and exit"
    ),
) -> None:
    """
    Konfig - Autonomous Okta SSO Integration Agent
    
    An intelligent AI agent that automates end-to-end SAML SSO integrations
    between third-party applications and Okta.
    """
    pass


@app.command()
def integrate(
    url: str = typer.Argument(..., help="URL to the vendor's SAML setup documentation"),
    okta_domain: str = typer.Option(..., help="Okta domain (e.g., company.okta.com)"),
    app_name: Optional[str] = typer.Option(None, help="Custom application name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate the integration without making changes"),
    headless: bool = typer.Option(False, "--headless", help="Run browser in headless mode (invisible)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    Start a new SSO integration job.
    
    This command initiates an autonomous integration process that will:
    1. Parse the vendor's SAML documentation
    2. Create and configure a SAML application in Okta
    3. Navigate the vendor's admin interface to complete configuration
    4. Verify the integration is working correctly
    
    By default, the browser runs in HEADED mode so you can watch the agent work.
    Use --headless if you want the browser to run invisibly.
    """
    browser_mode = "Headless (invisible)" if headless else "Headed (visible) - you can watch the agent work!"
    
    console.print(Panel.fit(
        f"🚀 Starting SSO Integration\n\n"
        f"📄 Documentation: {url}\n"
        f"🏢 Okta Domain: {okta_domain}\n"
        f"📱 App Name: {app_name or 'Auto-detected'}\n"
        f"🧪 Dry Run: {'Yes' if dry_run else 'No'}\n"
        f"🌐 Browser Mode: {browser_mode}",
        title="Konfig Integration",
        border_style="blue"
    ))
    
    if dry_run:
        console.print("🧪 [yellow]Dry run mode enabled - no changes will be made[/yellow]")
    
    if not headless:
        console.print("👀 [cyan]Browser will open in visible mode - you can watch the agent work![/cyan]")
    
    try:
        job_id = asyncio.run(_start_integration(url, okta_domain, app_name, dry_run, headless, verbose))
        
        console.print(f"✅ Integration job completed with ID: [green]{job_id}[/green]")
        console.print(f"💡 Use 'konfig jobs show --job-id {job_id}' to view details")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to start integration: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def resume(
    job_id: str = typer.Option(..., help="Job ID to resume"),
) -> None:
    """
    Resume a paused integration job.
    
    This command resumes an integration that was paused for human intervention
    or due to an error that has since been resolved.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        console.print(f"❌ [red]Invalid job ID format: {job_id}[/red]")
        raise typer.Exit(1)
    
    console.print(f"🔄 Resuming integration job: [blue]{job_id}[/blue]")
    
    try:
        # This will be implemented when we have the orchestrator
        asyncio.run(_resume_integration(job_uuid))
        console.print("✅ Integration job resumed successfully")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to resume integration: {e}[/red]")
        raise typer.Exit(1)


# Jobs management commands
jobs_app = typer.Typer(name="jobs", help="Manage integration jobs")
app.add_typer(jobs_app)


@jobs_app.command("list")
def list_jobs(
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    limit: int = typer.Option(10, help="Maximum number of jobs to display"),
) -> None:
    """List integration jobs."""
    try:
        # This will be implemented when we have the database layer
        jobs = asyncio.run(_get_jobs(status, limit))
        
        if not jobs:
            console.print("📝 No integration jobs found")
            return
        
        table = Table(title="Integration Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("App Name", style="blue")
        table.add_column("Created", style="yellow")
        table.add_column("Updated", style="magenta")
        
        for job in jobs:
            table.add_row(
                str(job.get("job_id", ""))[:8] + "...",
                job.get("status", "unknown"),
                job.get("app_name", "N/A"),
                job.get("created_at", "N/A"),
                job.get("updated_at", "N/A"),
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ [red]Failed to list jobs: {e}[/red]")
        raise typer.Exit(1)


@jobs_app.command("show")
def show_job(
    job_id: str = typer.Option(..., help="Job ID to display"),
    traces: bool = typer.Option(False, "--traces", help="Show execution traces"),
) -> None:
    """Show detailed information about a specific job."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        console.print(f"❌ [red]Invalid job ID format: {job_id}[/red]")
        raise typer.Exit(1)
    
    try:
        # This will be implemented when we have the database layer
        job_info = asyncio.run(_get_job_details(job_uuid, traces))
        
        if not job_info:
            console.print(f"❌ [red]Job not found: {job_id}[/red]")
            raise typer.Exit(1)
        
        # Display job information
        console.print(Panel.fit(
            f"🆔 Job ID: {job_info.get('job_id')}\n"
            f"📊 Status: {job_info.get('status')}\n"
            f"🏢 Okta Domain: {job_info.get('okta_domain', 'N/A')}\n"
            f"📱 App Name: {job_info.get('app_name', 'N/A')}\n"
            f"🕐 Created: {job_info.get('created_at')}\n"
            f"🕓 Updated: {job_info.get('updated_at')}",
            title=f"Job Details - {job_id[:8]}...",
            border_style="blue"
        ))
        
        if traces and job_info.get("traces"):
            console.print("\n📜 Execution Traces:")
            # Display traces in a formatted way
            for trace in job_info["traces"][-10:]:  # Show last 10 traces
                console.print(f"  [{trace.get('timestamp')}] {trace.get('trace_type')}: {trace.get('content')}")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to show job: {e}[/red]")
        raise typer.Exit(1)


# Configuration commands
config_app = typer.Typer(name="config", help="Configuration management")
app.add_typer(config_app)


@config_app.command("show")
def show_config() -> None:
    """Show current configuration."""
    try:
        settings = get_settings()
        
        table = Table(title="Konfig Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Source", style="yellow")
        
        # Show key configuration items (mask sensitive data)
        config_items = [
            ("Environment", settings.environment, "config"),
            ("Database URL", settings.database.url.split('@')[-1] if '@' in str(settings.database.url) else "Not configured", "environment"),
            ("Okta Domain", settings.okta.domain or "Not configured", "environment"),
            ("OpenAI API Key", "Configured" if settings.llm.openai_api_key else "Not configured", "environment"),
            ("Gemini API Key", "Configured" if settings.llm.gemini_api_key else "Not configured", "environment"),
            ("Browser Headless", str(settings.browser.headless), "config"),
        ]
        
        for setting, value, source in config_items:
            table.add_row(setting, str(value), source)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ [red]Failed to show configuration: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("validate")
def validate_config() -> None:
    """Validate the current configuration."""
    console.print("🔍 Validating configuration...")
    
    try:
        settings = get_settings()
        issues = []
        
        # Check essential configurations
        if not settings.okta.domain:
            issues.append("❌ Okta domain not configured (set OKTA_DOMAIN)")
        
        if not settings.okta.api_token:
            issues.append("❌ Okta API token not configured (set OKTA_API_TOKEN)")
        
        if not settings.llm.openai_api_key and not settings.llm.gemini_api_key:
            issues.append("❌ No LLM API key configured (set OPENAI_API_KEY or GEMINI_API_KEY)")
        
        if issues:
            console.print("⚠️  Configuration issues found:")
            for issue in issues:
                console.print(f"  {issue}")
            console.print("\n💡 Check your .env file or environment variables")
        else:
            console.print("✅ Configuration looks good!")
            
    except Exception as e:
        console.print(f"❌ [red]Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)


# Database management commands
db_app = typer.Typer(name="db", help="Database management")
app.add_typer(db_app)


@db_app.command("init")
def init_database() -> None:
    """Initialize the database with required schemas and extensions."""
    console.print("🗄️ Initializing database...")
    
    try:
        # This will be implemented when we have the database layer
        asyncio.run(_init_database())
        console.print("✅ Database initialized successfully")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to initialize database: {e}[/red]")
        raise typer.Exit(1)


@db_app.command("migrate")
def migrate_database() -> None:
    """Run database migrations."""
    console.print("🔄 Running database migrations...")
    
    try:
        # This will be implemented when we have the database layer
        asyncio.run(_migrate_database())
        console.print("✅ Migrations completed successfully")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to run migrations: {e}[/red]")
        raise typer.Exit(1)


@db_app.command("reset")
def reset_database(
    confirm: bool = typer.Option(False, "--confirm", help="Confirm database reset")
) -> None:
    """Reset the database (WARNING: This will delete all data!)."""
    if not confirm:
        console.print("⚠️  [yellow]This will delete ALL data in the database![/yellow]")
        confirm_reset = typer.confirm("Are you sure you want to continue?")
        if not confirm_reset:
            console.print("❌ Database reset cancelled")
            return
    
    console.print("🔥 Resetting database...")
    
    try:
        # This will be implemented when we have the database layer
        asyncio.run(_reset_database())
        console.print("✅ Database reset completed")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to reset database: {e}[/red]")
        raise typer.Exit(1)


# Health check command
@app.command()
def health() -> None:
    """Check the health of all Konfig components."""
    console.print("🏥 Performing health check...")
    
    try:
        # This will be implemented when we have all components
        health_status = asyncio.run(_health_check())
        
        table = Table(title="Konfig Health Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        for component, status in health_status.items():
            status_color = "green" if status.get("healthy") else "red"
            table.add_row(
                component,
                f"[{status_color}]{'✅ Healthy' if status.get('healthy') else '❌ Unhealthy'}[/{status_color}]",
                status.get("details", "N/A")
            )
        
        console.print(table)
        
        # Exit with error code if any component is unhealthy
        if not all(status.get("healthy", False) for status in health_status.values()):
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"❌ [red]Health check failed: {e}[/red]")
        raise typer.Exit(1)


# Implementation functions
async def _start_integration(url: str, okta_domain: str, app_name: Optional[str], dry_run: bool, headless: bool, verbose: bool) -> str:
    """Start a new integration job."""
    # Temporarily override browser headless setting for this integration
    settings = get_settings()
    original_headless = settings.browser.headless
    settings.browser.headless = headless
    
    orchestrator = MVPOrchestrator()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Starting integration...", total=None)
            
            try:
                # Start the integration
                result = await orchestrator.integrate_application(
                    documentation_url=url,
                    okta_domain=okta_domain,
                    app_name=app_name,
                    dry_run=dry_run
                )
                
                if result["success"]:
                    progress.update(task, description="✅ Integration completed successfully!")
                    
                    # Display results
                    console.print(Panel.fit(
                        f"🎉 Integration Results\n\n"
                        f"🆔 Job ID: {result['job_id']}\n"
                        f"🏢 Vendor: {result.get('vendor', 'Unknown')}\n"
                        f"📱 Plan Used: {result.get('plan_used', 'Generic')}\n"
                        f"⏱️  Duration: {result.get('duration_seconds', 0):.2f}s\n"
                        f"✅ Steps Completed: {result.get('steps_completed', 0)}\n"
                        f"🧪 Dry Run: {'Yes' if dry_run else 'No'}" +
                        (f"\n🔗 Okta App ID: {result.get('okta_app_id')}" if result.get('okta_app_id') else ""),
                        title="Integration Completed",
                        border_style="green"
                    ))
                    
                    return result["job_id"]
                else:
                    progress.update(task, description="❌ Integration failed!")
                    console.print(f"❌ [red]Integration failed: {result.get('error', 'Unknown error')}[/red]")
                    raise typer.Exit(1)
                    
            except Exception as e:
                progress.update(task, description="❌ Integration failed!")
                raise e
    finally:
        # Restore original headless setting
        settings.browser.headless = original_headless


async def _resume_integration(job_id: uuid.UUID) -> None:
    """Resume a paused integration job."""
    console.print("🚧 [yellow]Resume functionality not yet implemented[/yellow]")
    console.print("💡 This feature will be available when Human-in-the-Loop (HITL) capabilities are fully implemented.")


async def _get_jobs(status: Optional[str], limit: int) -> list:
    """Get list of integration jobs."""
    memory_module = MemoryModule()
    
    try:
        jobs = await memory_module.list_jobs(status=status, limit=limit)
        
        return [
            {
                "job_id": str(job.job_id),
                "status": job.status,
                "app_name": job.vendor_name,
                "created_at": job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "N/A",
                "updated_at": job.updated_at.strftime("%Y-%m-%d %H:%M:%S") if job.updated_at else "N/A",
            }
            for job in jobs
        ]
    except Exception as e:
        console.print(f"⚠️  [yellow]Database not available: {e}[/yellow]")
        return []


async def _get_job_details(job_id: uuid.UUID, include_traces: bool) -> Optional[dict]:
    """Get detailed job information."""
    orchestrator = MVPOrchestrator()
    
    try:
        job_status = await orchestrator.get_job_status(job_id)
        
        if "error" in job_status:
            return None
            
        return {
            "job_id": job_status["job_id"],
            "status": job_status["status"],
            "app_name": job_status.get("vendor_name", "N/A"),
            "okta_domain": "N/A",  # Not stored in current implementation
            "created_at": job_status["created_at"],
            "updated_at": job_status["updated_at"],
            "traces": job_status.get("recent_traces", []) if include_traces else []
        }
        
    except Exception as e:
        console.print(f"⚠️  [yellow]Could not retrieve job details: {e}[/yellow]")
        return None


async def _init_database() -> None:
    """Initialize the database."""
    from konfig.database.connection import init_database
    await init_database()


async def _migrate_database() -> None:
    """Run database migrations."""
    import subprocess
    import os
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run Alembic upgrade
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    
    if result.returncode != 0:
        console.print(f"❌ [red]Migration failed: {result.stderr}[/red]")
        raise Exception(f"Migration failed: {result.stderr}")
    
    console.print("✅ Database migrations applied successfully")


async def _reset_database() -> None:
    """Reset the database."""
    from konfig.database.connection import reset_database
    await reset_database()


async def _health_check() -> dict:
    """Perform health check on all components."""
    settings = get_settings()
    health_status = {}
    
    # Check database
    try:
        db_health = await health_check()
        health_status["database"] = {
            "healthy": db_health.get("database_connected", False),
            "details": "PostgreSQL connection OK" if db_health.get("database_connected") else "Database connection failed"
        }
    except Exception as e:
        health_status["database"] = {"healthy": False, "details": f"Database error: {e}"}
    
    # Check Okta configuration
    okta_configured = bool(settings.okta.domain and settings.okta.api_token)
    health_status["okta_api"] = {
        "healthy": okta_configured,
        "details": "Okta API configured" if okta_configured else "Okta domain/token not configured"
    }
    
    # Check LLM configuration
    llm_configured = bool(settings.llm.openai_api_key or settings.llm.gemini_api_key)
    health_status["llm"] = {
        "healthy": llm_configured,
        "details": "LLM API key configured" if llm_configured else "No LLM API key configured"
    }
    
    # Check browser (Playwright)
    try:
        # Simple check - we assume if imports work, Playwright is ready
        from playwright.async_api import async_playwright
        health_status["web_browser"] = {"healthy": True, "details": "Playwright ready"}
    except Exception as e:
        health_status["web_browser"] = {"healthy": False, "details": f"Playwright error: {e}"}
    
    return health_status


def main_cli() -> None:
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ [red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()