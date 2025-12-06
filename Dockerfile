# Multi-stage build for smaller image
FROM python:3.12-slim as builder

WORKDIR /app

# Install system requirements for CA certificates
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install CA certificates
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy entire project directory
COPY . .

# Note: No local data directory needed for PostgreSQL

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH
ENV TZ=UTC

# Make start.sh executable
RUN chmod +x start.sh

# Run as non-root user (security best practice)
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app
USER botuser

# Health check - verify PostgreSQL database connection
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psycopg2; import os; psycopg2.connect(os.getenv('DATABASE_URL')).close()" || exit 1

CMD ["./start.sh"]
