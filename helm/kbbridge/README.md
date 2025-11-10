# KB-Bridge Helm Chart

This Helm chart deploys the KB-Bridge MCP server on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Docker image of kbbridge (build from Dockerfile or use from registry)

## Installation

### Build and push Docker image

```bash
# Build the image
docker build -t kbbridge:latest .

# Tag for your registry (replace with your registry)
docker tag kbbridge:latest your-registry/kbbridge:0.1.0

# Push to registry
docker push your-registry/kbbridge:0.1.0
```

### Install the chart

```bash
# Install with default values
helm install kbbridge ./helm/kbbridge

# Install with custom values
helm install kbbridge ./helm/kbbridge -f my-values.yaml

# Install with custom image
helm install kbbridge ./helm/kbbridge \
  --set image.repository=your-registry/kbbridge \
  --set image.tag=0.1.0
```

### Install with secrets

```bash
# Create secrets first
kubectl create secret generic kbbridge-secrets \
  --from-literal=RETRIEVAL_API_KEY=your-key \
  --from-literal=LLM_API_TOKEN=your-token

# Install with secret references
helm install kbbridge ./helm/kbbridge \
  --set envFromSecrets[0]=RETRIEVAL_API_KEY \
  --set envFromSecrets[1]=LLM_API_TOKEN
```

## Configuration

Key configuration values in `values.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `kbbridge` |
| `image.tag` | Image tag | `latest` |
| `service.type` | Service type | `ClusterIP` |
| `service.port` | Service port | `5210` |
| `env.RETRIEVAL_ENDPOINT` | Retrieval backend endpoint | `https://api.dify.ai/v1` |
| `env.RETRIEVAL_API_KEY` | Retrieval API key | `""` |
| `env.LLM_API_URL` | LLM API URL | `""` |
| `env.LLM_MODEL` | LLM model | `gpt-4o` |
| `autoscaling.enabled` | Enable HPA | `false` |
| `autoscaling.minReplicas` | Minimum replicas | `1` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |

## Upgrading

```bash
helm upgrade kbbridge ./helm/kbbridge -f my-values.yaml
```

## Uninstalling

```bash
helm uninstall kbbridge
```

## Production Recommendations

1. **Use Secrets**: Store sensitive values in Kubernetes secrets instead of plain env vars
2. **Enable Autoscaling**: Set `autoscaling.enabled=true` for production workloads
3. **Resource Limits**: Adjust `resources` based on your workload
4. **Ingress**: Enable ingress for external access
5. **Image Tag**: Use specific version tags instead of `latest`

## Examples

### Production deployment with secrets and autoscaling

```yaml
# production-values.yaml
replicaCount: 2
image:
  repository: your-registry/kbbridge
  tag: "0.1.0"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

envFromSecrets:
  - RETRIEVAL_API_KEY
  - LLM_API_TOKEN

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: kbbridge.example.com
      paths:
        - path: /
          pathType: Prefix
```

Install with:
```bash
helm install kbbridge ./helm/kbbridge -f production-values.yaml
```
