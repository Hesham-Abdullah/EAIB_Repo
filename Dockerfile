FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js for Playwright
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install  -r requirements.txt

# Install Playwright and Chromium
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/training data/raw data/processed

# Install NLP models
RUN python scripts/setup_nlp.py

# Expose ports
EXPOSE 8000 8501

# Environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the application
CMD ["python", "app.py"] 