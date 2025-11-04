# Multi-stage build for kbbridge MCP server
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir build setuptools wheel

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY kbbridge/ ./kbbridge/

# Build the package
RUN python -m build --wheel --outdir /dist

# Final stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies and the package
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl && \
    rm -rf /root/.cache/pip

# Create non-root user for security
RUN useradd -m -u 1000 kbbridge && \
    chown -R kbbridge:kbbridge /app

# Switch to non-root user
USER kbbridge

# Expose the default port
EXPOSE 5210

# Health check (check if port is listening)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.connect(('localhost', 5210)); s.close()" || exit 1

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=5210

# Run the server
# Use PORT env var if provided (for platforms like Smithery), otherwise default to 5210
CMD ["sh", "-c", "python -m kbbridge.server --host 0.0.0.0 --port ${PORT:-5210} --transport streamable-http"]
