# Multi-stage build for smaller image
FROM python:3.12-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY *.py .
COPY .env.example .

# Create data directory
RUN mkdir -p data

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Run as non-root user (security best practice)
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app
USER botuser

# Health check - verify database file is accessible
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sqlite3; sqlite3.connect('data/messages.db').close()" || exit 1

CMD ["python", "-u", "bot.py"]
