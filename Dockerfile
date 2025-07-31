# Multi-stage Docker build for Konfig
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 konfig && \
    useradd --uid 1000 --gid konfig --shell /bin/bash --create-home konfig

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

WORKDIR /app

# Install Node.js for Playwright
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -e .

# Install Playwright browsers
RUN playwright install --with-deps chromium firefox webkit

# Copy source code
COPY . .

# Set proper ownership
RUN chown -R konfig:konfig /app

USER konfig

# Expose ports
EXPOSE 8000

# Development command
CMD ["python", "-m", "konfig.cli", "web", "start", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

WORKDIR /app

# Install Node.js for Playwright (minimal)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install Python dependencies (no dev dependencies)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-dev -r requirements.txt && \
    pip install .

# Install only Chromium for production
RUN playwright install --with-deps chromium

# Copy source code
COPY konfig/ ./konfig/
COPY scripts/ ./scripts/
COPY deployment/ ./deployment/

# Create necessary directories
RUN mkdir -p logs data && \
    chown -R konfig:konfig /app

USER konfig

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "konfig.cli", "web", "start", "--host", "0.0.0.0", "--port", "8000"]

# Testing stage
FROM development as testing

# Install additional test dependencies
RUN pip install pytest-xdist pytest-benchmark

# Copy test files
COPY tests/ ./tests/

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=konfig", "--cov-report=term-missing"]