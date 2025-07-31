# Konfig - Autonomous Okta SSO Integration Agent

![Konfig Logo](https://img.shields.io/badge/Konfig-AI%20Agent-blue)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Konfig is an autonomous AI agent designed to automate end-to-end SAML SSO integrations between third-party applications and Okta. Built with advanced agentic capabilities, Konfig can understand vendor documentation, plan integration steps, execute configurations across multiple environments, and learn from experience to continuously improve its performance.

## 🎯 Vision

Transform weeks of manual SSO integration work into minutes of autonomous operation. Konfig operates as an intelligent system that can:

- **Comprehend**: Parse and understand unstructured vendor documentation
- **Plan**: Decompose complex integration tasks into executable workflows  
- **Act**: Interact with web UIs and APIs to perform configurations
- **Learn**: Self-heal from errors and improve through experience

## 🏗️ Architecture

Konfig follows a modular, 5-component architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                           │
│                  (LangGraph-based)                         │
│                ReAct Cognitive Loop                        │
└─────────┬───────────────────────────────────────┬─────────┘
          │                                       │
┌─────────▼─────────┐  ┌─────────────────┐  ┌─────▼─────────┐
│   PERCEPTION      │  │   COGNITION     │  │    ACTION     │
│   MODULE          │  │   MODULE        │  │   MODULE      │
│                   │  │                 │  │               │
│ • Doc parsing     │  │ • LLM reasoning │  │ • Web automation│
│ • Knowledge base  │  │ • Task planning │  │ • API clients │
│ • Vector search   │  │ • Decision making│  │ • Tool execution│
└───────────────────┘  └─────────────────┘  └───────────────┘
          │                                       │
┌─────────▼─────────────────────────────────────▼─────────────┐
│                  MEMORY MODULE                              │
│              (PostgreSQL + pgvector)                       │
│                                                             │
│ • Working Memory    • Episodic Memory    • Semantic Memory │
│ • State Management  • Execution Traces  • Knowledge Vectors│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  LEARNING MODULE                           │
│                                                             │
│ • Self-healing     • Pattern extraction   • Performance    │  
│ • Error recovery   • Procedural memory   • Optimization   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### Autonomous Operation
- **Zero-touch integration**: Complete SAML configurations without manual intervention
- **Multi-environment orchestration**: Seamlessly switches between web UIs and APIs
- **Dynamic planning**: Adapts strategies based on real-time observations

### Self-Healing Intelligence  
- **Error recovery**: Automatically detects and resolves common integration issues
- **Learning from experience**: Builds procedural memory from execution traces
- **Continuous improvement**: Gets more reliable and efficient with each integration

### Human-in-the-Loop (HITL)
- **Graceful fallback**: Pauses for human assistance when encountering unknown situations
- **State persistence**: Seamlessly resumes operations after human intervention
- **Web-based interface**: Simple UI for operators to provide guidance

### Enterprise-Ready
- **Security-first**: Encrypted credential management and secure API handling
- **Full observability**: Comprehensive execution tracing and audit trails
- **Scalable architecture**: Containerized design with Kubernetes support

## 🛠️ Technology Stack

- **Language**: Python 3.11+
- **AI Framework**: LangGraph + LangChain
- **Database**: PostgreSQL with pgvector extension
- **Web Automation**: Playwright
- **API Integration**: Custom Okta Management API client
- **Containerization**: Docker + Kubernetes
- **Testing**: pytest with comprehensive coverage

## 📋 Prerequisites

- Python 3.11 or higher
- PostgreSQL 14+ with pgvector extension
- Docker and Docker Compose
- Node.js (for Playwright browser installation)

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/konfig.git
cd konfig
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Playwright browsers
```bash
playwright install
```

### 4. Set up PostgreSQL with pgvector
```bash
# Using Docker Compose (recommended)
docker-compose up -d postgres

# Or install manually
createdb konfig
psql konfig -c "CREATE EXTENSION vector;"
```

### 5. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 6. Run database migrations
```bash
alembic upgrade head
```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/konfig

# Okta API
OKTA_DOMAIN=your-domain.okta.com
OKTA_API_TOKEN=your-api-token

# LLM Configuration
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key

# Security
SECRET_KEY=your-secret-key
VAULT_URL=http://localhost:8200
VAULT_TOKEN=your-vault-token

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true

```

## 🚀 Usage

### Command Line Interface

Konfig is a CLI-first tool. All interactions happen through the command line, with the browser running in visible mode so you can watch the agent work.

```bash
# Start a new SSO integration (browser will be visible by default)
konfig integrate "https://vendor.com/saml-setup-guide" --okta-domain "company.okta.com"

# Run integration with custom app name
konfig integrate "https://vendor.com/saml-setup-guide" --okta-domain "company.okta.com" --app-name "My Custom App"

# Run in dry-run mode (no actual changes)
konfig integrate "https://vendor.com/saml-setup-guide" --okta-domain "company.okta.com" --dry-run

# Run with invisible browser (headless mode)
konfig integrate "https://vendor.com/saml-setup-guide" --okta-domain "company.okta.com" --headless

# Check configuration
konfig config show
konfig config validate

# Manage jobs
konfig jobs list
konfig jobs show --job-id "550e8400-e29b-41d4-a716-446655440000" --traces

# Database management
konfig db init
konfig db migrate
konfig db reset --confirm

# Health check
konfig health
```

**Key Features:**
- 👀 **Visible Browser**: By default, the browser runs in headed mode so you can watch the agent work
- 🧪 **Dry Run Mode**: Test integrations without making actual changes
- 📊 **Rich CLI Output**: Beautiful progress indicators and result displays
- 🎯 **Job Management**: Track and manage integration jobs

### Python API

```python
from konfig import KonfigAgent

# Initialize the agent
agent = KonfigAgent(
    okta_domain="company.okta.com",
    okta_api_token="your-token"
)

# Start an integration
job_id = await agent.integrate_application(
    documentation_url="https://vendor.com/saml-setup",
    app_name="Vendor Application"
)

# Monitor progress
status = await agent.get_job_status(job_id)
print(f"Integration status: {status}")
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=konfig --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Code Quality

```bash
# Format code
black konfig tests
isort konfig tests

# Lint code
flake8 konfig tests
mypy konfig

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 📊 Monitoring and Observability

Konfig provides comprehensive observability through:

- **Execution Traces**: Detailed logs of every thought, action, and observation
- **Performance Metrics**: Integration success rates, completion times, error frequencies
- **Learning Analytics**: Self-healing effectiveness and procedural memory growth
- **Resource Monitoring**: Database performance, API rate limits, browser resource usage

Access metrics through the health command: `konfig health`

## 🔒 Security

Konfig follows security best practices:

- **Encrypted Secrets**: All credentials stored encrypted at rest
- **API Security**: Rate limiting, authentication, and input validation
- **Network Security**: TLS/SSL for all external communications
- **Audit Trails**: Complete logging of all actions for compliance
- **Privilege Separation**: Least-privilege access for all components

## 🏗️ Development Phases

### Phase 1: Minimum Viable Agent ✅
- Core ReAct loop with basic tool use
- WebInteractor and OktaAPIClient tools
- Hard-coded plan execution for known applications

### Phase 2: Robust Agent (In Progress)
- Complete database schema and state persistence
- Human-in-the-Loop workflow implementation
- Comprehensive execution tracing and logging
- Document parsing and perception capabilities

### Phase 3: Autonomous Agent (Planned)
- Self-healing and error recovery mechanisms
- Learning module with pattern extraction
- Procedural memory and continuous improvement
- General-purpose documentation understanding

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/konfig/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/konfig/discussions)
- **Security**: security@konfig.ai

## 🙏 Acknowledgments

- LangChain and LangGraph teams for the agentic framework
- Okta for the comprehensive Management API
- Playwright team for robust web automation
- PostgreSQL and pgvector communities for vector database capabilities

---

**Built with ❤️ by the Konfig Team**

*Konfig: Where AI meets SSO automation*