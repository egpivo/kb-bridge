# KB-Bridge MCP Server Deployment Guide

## Understanding MCP Server Deployment

Model Context Protocol (MCP) servers are typically deployed as HTTP services that AI agents and clients connect to. Your server exposes an endpoint at `http://host:port/mcp` using the `streamable-http` transport.

## Common Deployment Scenarios

### 1. **Local Development & Testing**
**Environment**: Developer's machine, CI/CD pipelines
**Tools**: Docker Compose, Docker, or direct Python execution
**Use Case**: Quick iteration, testing, development

```bash
# Direct Python (fastest for development)
python -m kbbridge.server --host 0.0.0.0 --port 5210

# Docker Compose (convenient for local testing)
docker-compose up -d
```

**Why Docker Compose?**
- Easy environment variable management (`.env` file)
- Quick restart/stop
- Consistent with production-like setup
- No Kubernetes overhead for local dev

### 2. **Cloud Platforms**

#### AWS (Amazon Bedrock AgentCore)
- **Best for**: Enterprise AWS deployments
- **Deployment**: AWS Bedrock AgentCore Runtime
- **Supports**: Stateless streamable-HTTP servers (matches your server)
- **Link**: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-mcp.html

#### Azure (App Service)
- **Best for**: Azure-based deployments
- **Deployment**: Azure App Service
- **Integration**: Azure AI Foundry agents
- **Link**: Azure App Service documentation

#### CloudMCP
- **Best for**: Quick cloud deployment
- **Deployment**: Deploy from PyPI (your package is already there!)
- **Features**: Real-time validation, smart detection
- **Link**: https://cloudmcp.run

### 3. **Kubernetes (Production)**

#### Managed Kubernetes Services
- **AWS EKS**: Enterprise-grade Kubernetes
- **Google GKE**: Google Cloud Kubernetes
- **Azure AKS**: Azure Kubernetes Service
- **DigitalOcean Kubernetes**: Simple managed K8s

#### Self-Managed Kubernetes
- **On-premise clusters**
- **Bare metal servers**
- **VM-based clusters**

**Why Kubernetes?**
- Horizontal scaling (multiple replicas)
- High availability (auto-restart, health checks)
- Resource management (CPU/memory limits)
- Service discovery and load balancing
- Rolling updates without downtime
- Secret management for credentials

### 4. **Container Platforms**

#### Docker Swarm
- Simpler than Kubernetes
- Good for small to medium deployments
- Native Docker orchestration

#### Nomad (HashiCorp)
- Alternative to Kubernetes
- Supports Docker containers
- Simpler configuration

## Decision Matrix

| Deployment Scenario | Recommended Solution | Docker Compose Needed? | Helm Needed? |
|---------------------|---------------------|----------------------|--------------|
| **Local Development** | Docker Compose or direct Python | ✅ Yes | ❌ No |
| **CI/CD Testing** | Docker Compose | ✅ Yes | ❌ No |
| **Single Server** | Docker or systemd service | ✅ Optional | ❌ No |
| **Small Team (2-10 users)** | Docker Compose or Docker Swarm | ✅ Yes | ❌ Optional |
| **Medium Production** | Kubernetes (managed) | ❌ No | ✅ Yes |
| **Large Production** | Kubernetes (managed) | ❌ No | ✅ Yes |
| **Enterprise** | Kubernetes (EKS/GKE/AKS) | ❌ No | ✅ Yes |
| **Cloud Platform** | Platform-specific (Bedrock, Azure) | ❌ No | ❌ No |

## Recommendations by Use Case

### Scenario A: Solo Developer / Small Project
**Recommendation**: **Docker Compose only**
- Simple setup
- Easy to manage
- No Kubernetes complexity
- Perfect for < 100 concurrent users

**Keep**: Dockerfile + docker-compose.yml
**Remove**: Helm (optional, but can keep for future)

### Scenario B: Production with Auto-scaling Needs
**Recommendation**: **Helm + Kubernetes**
- Need multiple replicas
- Need auto-scaling
- Need high availability
- Enterprise requirements

**Keep**: Helm chart
**Keep Docker Compose**: For local development/testing

### Scenario C: Cloud Platform Deployment
**Recommendation**: **Platform-specific** (AWS Bedrock, Azure, CloudMCP)
- Use platform-native deployment
- Follow platform best practices
- May not need Helm if platform handles orchestration

**Keep**: Dockerfile (for building image)
**Keep**: Helm (optional, for hybrid deployments)

### Scenario D: Hybrid (Dev + Prod)
**Recommendation**: **Keep both**
- Docker Compose for local dev
- Helm for production Kubernetes
- Best of both worlds

## Your Current Setup Analysis

Based on your codebase:

1. **Server Type**: HTTP-based MCP server (`streamable-http` transport)
2. **Clients**: AI agents connecting via HTTP to `/mcp` endpoint
3. **Port**: 5210 (default)
4. **Authentication**: Header-based credentials
5. **Transport**: HTTP (not stdio)

## Final Recommendation

**Keep both Docker Compose and Helm** because:

1. **Docker Compose** is valuable for:
   - Local development (most developers don't run K8s locally)
   - CI/CD pipelines (simpler than spinning up K8s)
   - Quick testing
   - Documentation examples

2. **Helm** is essential for:
   - Production Kubernetes deployments
   - Enterprise customers
   - Scalable deployments
   - Professional deployment practices

3. **Both are complementary**:
   - Developers use Docker Compose locally
   - Ops teams use Helm in production
   - Different use cases, both needed

## If You Must Choose One

**If you only target Kubernetes production:**
- Keep Helm
- Remove docker-compose.yml (but keep Dockerfile)
- Developers can use `docker run` directly for local testing

**If you only target local/simple deployments:**
- Keep Docker Compose
- Remove Helm (but keep Dockerfile)
- Users can deploy manually if needed

## Next Steps

1. **Identify your primary deployment target**:
   - [ ] Local development only?
   - [ ] Small production deployment?
   - [ ] Enterprise Kubernetes?
   - [ ] Cloud platform (AWS/Azure)?

2. **Choose based on your answer**:
   - Mostly local → Keep Docker Compose, Helm optional
   - Production → Keep Helm, Docker Compose for dev
   - Both → Keep both (recommended)

3. **Update README** based on your decision

## Questions to Consider

1. **Who will deploy this?**
   - Developers → Docker Compose helpful
   - DevOps/SRE → Helm essential
   - Cloud platform users → Platform-specific docs

2. **What's your scale?**
   - < 10 users → Docker Compose sufficient
   - 10-100 users → Either works
   - 100+ users → Kubernetes/Helm recommended

3. **Do you have Kubernetes?**
   - Yes → Helm is valuable
   - No → Docker Compose is simpler

4. **Is this for production?**
   - Yes → Helm recommended
   - No → Docker Compose sufficient
