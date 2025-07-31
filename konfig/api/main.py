"""
FastAPI web interface for Konfig.

This module provides a REST API and web interface for interacting with
the Konfig autonomous SSO integration agent.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from konfig import __version__
from konfig.config.settings import get_settings
from konfig.orchestrator.mvp_orchestrator import MVPOrchestrator
from konfig.database.connection import health_check
from konfig.utils.logging import get_logger
from konfig.utils.metrics import metrics_registry

# Initialize FastAPI app
app = FastAPI(
    title="Konfig API",
    description="Autonomous Okta SSO Integration Agent",
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
settings = get_settings()
logger = get_logger(__name__, "api")
orchestrator = MVPOrchestrator()

# Background tasks for async job processing
background_jobs: Dict[str, Dict] = {}


# Pydantic models for API requests/responses
class IntegrationRequest(BaseModel):
    """Request model for starting an integration."""
    
    documentation_url: str = Field(..., description="URL to vendor SAML documentation")
    okta_domain: str = Field(..., description="Okta domain (e.g., company.okta.com)")
    app_name: Optional[str] = Field(None, description="Custom application name")
    vendor_hint: Optional[str] = Field(None, description="Hint about the vendor type")
    dry_run: bool = Field(False, description="Simulate integration without making changes")


class IntegrationResponse(BaseModel):
    """Response model for integration results."""
    
    job_id: str
    status: str
    message: str
    vendor: Optional[str] = None
    okta_app_id: Optional[str] = None
    duration_seconds: Optional[float] = None


class JobStatus(BaseModel):
    """Job status response model."""
    
    job_id: str
    status: str
    vendor_name: Optional[str] = None
    created_at: str
    updated_at: str
    okta_app_id: Optional[str] = None
    error_message: Optional[str] = None
    recent_traces: List[Dict] = []


class VendorInfo(BaseModel):
    """Known vendor information."""
    
    key: str
    name: str
    admin_url: str
    steps: int


# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Konfig - Autonomous SSO Integration</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 40px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
            .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .loading { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Konfig</h1>
            <p>Autonomous Okta SSO Integration Agent</p>
        </div>
        
        <form id="integrationForm">
            <div class="form-group">
                <label for="documentationUrl">Documentation URL *</label>
                <input type="url" id="documentationUrl" required 
                       placeholder="https://vendor.com/saml-setup-guide">
            </div>
            
            <div class="form-group">
                <label for="oktaDomain">Okta Domain *</label>
                <input type="text" id="oktaDomain" required 
                       placeholder="company.okta.com">
            </div>
            
            <div class="form-group">
                <label for="appName">Application Name</label>
                <input type="text" id="appName" 
                       placeholder="My Application SAML">
            </div>
            
            <div class="form-group">
                <label for="vendorHint">Vendor Type</label>
                <select id="vendorHint">
                    <option value="">Auto-detect</option>
                    <option value="google_workspace">Google Workspace</option>
                    <option value="slack">Slack</option>
                    <option value="atlassian">Atlassian</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" id="dryRun"> Dry Run (simulate only)
                </label>
            </div>
            
            <button type="submit">Start Integration</button>
        </form>
        
        <div id="result"></div>
        
        <script>
            document.getElementById('integrationForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<div class="result loading">🔄 Starting integration...</div>';
                
                const formData = {
                    documentation_url: document.getElementById('documentationUrl').value,
                    okta_domain: document.getElementById('oktaDomain').value,
                    app_name: document.getElementById('appName').value || null,
                    vendor_hint: document.getElementById('vendorHint').value || null,
                    dry_run: document.getElementById('dryRun').checked
                };
                
                try {
                    const response = await fetch('/api/integrations', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <div class="result success">
                                <h3>✅ Integration ${result.status === 'completed' ? 'Completed' : 'Started'}</h3>
                                <p><strong>Job ID:</strong> ${result.job_id}</p>
                                ${result.vendor ? `<p><strong>Vendor:</strong> ${result.vendor}</p>` : ''}
                                ${result.okta_app_id ? `<p><strong>Okta App ID:</strong> ${result.okta_app_id}</p>` : ''}
                                ${result.duration_seconds ? `<p><strong>Duration:</strong> ${result.duration_seconds.toFixed(2)}s</p>` : ''}
                                <p><strong>Message:</strong> ${result.message}</p>
                                <p><a href="/api/jobs/${result.job_id}" target="_blank">View Job Details</a></p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="result error">
                                <h3>❌ Integration Failed</h3>
                                <p><strong>Error:</strong> ${result.detail || result.message || 'Unknown error'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>❌ Request Failed</h3>
                            <p><strong>Error:</strong> ${error.message}</p>
                        </div>
                    `;
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/api/integrations", response_model=IntegrationResponse)
async def start_integration(request: IntegrationRequest, background_tasks: BackgroundTasks):
    """Start a new SSO integration."""
    
    logger.info("Integration requested", url=request.documentation_url, domain=request.okta_domain)
    
    try:
        # Start integration
        result = await orchestrator.integrate_application(
            documentation_url=request.documentation_url,
            okta_domain=request.okta_domain,
            app_name=request.app_name,
            vendor_hint=request.vendor_hint,
            dry_run=request.dry_run
        )
        
        if result["success"]:
            return IntegrationResponse(
                job_id=result["job_id"],
                status="completed",
                message="Integration completed successfully",
                vendor=result.get("vendor"),
                okta_app_id=result.get("okta_app_id"),
                duration_seconds=result.get("duration_seconds")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Integration failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a specific job."""
    
    try:
        job_uuid = uuid.UUID(job_id)
        status = await orchestrator.get_job_status(job_uuid)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return JobStatus(**status)
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vendors", response_model=List[VendorInfo])
async def list_known_vendors():
    """List all known vendor configurations."""
    
    try:
        vendors = await orchestrator.list_known_vendors()
        return [VendorInfo(**vendor) for vendor in vendors]
    except Exception as e:
        logger.error(f"Failed to list vendors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check_endpoint():
    """Health check endpoint."""
    
    try:
        # Check database connectivity
        db_health = await health_check()
        
        # Check orchestrator
        orchestrator_health = orchestrator is not None
        
        return {
            "status": "healthy" if db_health.get("database_connected") and orchestrator_health else "unhealthy",
            "version": __version__,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "healthy" if db_health.get("database_connected") else "unhealthy",
                "orchestrator": "healthy" if orchestrator_health else "unhealthy",
                "extensions": db_health.get("details", {}).get("installed_extensions", [])
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics in JSON format."""
    
    try:
        return metrics_registry.export_json()
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format."""
    
    try:
        prometheus_data = metrics_registry.export_prometheus()
        return Response(prometheus_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to export Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/version")
async def get_version():
    """Get application version information."""
    
    return {
        "version": __version__,
        "name": "Konfig",
        "description": "Autonomous Okta SSO Integration Agent"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {"error": "Not found", "path": str(request.url.path), "status": 404}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status": 500}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Konfig API starting up", version=__version__)
    
    # Initialize database if needed
    try:
        db_health = await health_check()
        if not db_health.get("database_connected"):
            logger.warning("Database not connected during startup")
    except Exception as e:
        logger.error(f"Database health check failed during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Konfig API shutting down")
    
    # Close perception module HTTP client
    if hasattr(orchestrator.perception_module, 'http_client'):
        await orchestrator.perception_module.close()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "konfig.api.main:app",
        host=settings.web.host,
        port=settings.web.port,
        reload=settings.web.reload,
        workers=settings.web.workers if not settings.web.reload else 1,
        log_level=settings.logging.level.lower()
    )