#!/bin/bash
# EAIB Docker Startup Script

set -e

echo "🐳 EAIB Docker Deployment"
echo "========================="

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Determine Docker Compose command
DOCKER_COMPOSE_CMD="docker-compose"
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
fi

# Setup environment file
if [ ! -f .env ]; then
    if [ -f env.template ]; then
        cp env.template .env
        echo "⚠️  Edit .env with your API keys, then run again"
        exit 0
    else
        echo "❌ No environment file found"
        exit 1
    fi
fi

# Basic API key validation
if grep -q "your_.*_api_key" .env; then
    echo "⚠️  Replace placeholder API keys in .env"
    exit 1
fi

echo "🚀 Starting EAIB..."

# Cleanup function
cleanup() {
    echo "🛑 Stopping..."
    $DOCKER_COMPOSE_CMD down
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start services
$DOCKER_COMPOSE_CMD up --build -d

# Wait and check status
sleep 10
echo "📊 Status:"
$DOCKER_COMPOSE_CMD ps

# Health check
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ API running: http://localhost:8000"
else
    echo "❌ API not responding"
fi

echo ""
echo "🎉 EAIB Started!"
echo "🌐 Interface: http://localhost:8501"
echo "📚 API Docs:  http://localhost:8000/docs"
echo ""
echo "Commands:"
echo "  Stop:    $DOCKER_COMPOSE_CMD down"
echo "  Logs:    $DOCKER_COMPOSE_CMD logs -f"
echo "  Status:  $DOCKER_COMPOSE_CMD ps"
echo ""
echo "Press Ctrl+C to stop"

# Show logs
$DOCKER_COMPOSE_CMD logs -f 