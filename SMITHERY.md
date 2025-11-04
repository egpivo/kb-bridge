# Deploying KB-Bridge to Smithery

Smithery is a platform for deploying MCP servers. This guide will help you deploy KB-Bridge to Smithery.

## Prerequisites

1. **Smithery Account**: Sign up at [smithery.ai](https://smithery.ai)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Environment Variables**: Prepare your API keys and endpoints

## Configuration

### 1. Smithery Configuration File

The `smithery.yaml` file is already configured with:
- Runtime: Python
- Port: 5210
- Transport: streamable-http
- Health check: `/mcp`

### 2. Required Environment Variables

Set these in the Smithery dashboard:

**Required:**
- `RETRIEVAL_ENDPOINT` - Your retrieval backend endpoint (e.g., `https://api.dify.ai/v1`)
- `RETRIEVAL_API_KEY` - Your retrieval API key
- `LLM_API_URL` - Your LLM service endpoint
- `LLM_MODEL` - LLM model name (e.g., `gpt-4o`)
- `LLM_API_TOKEN` - Your LLM API token

**Optional:**
- `RERANK_URL` - Reranking service URL
- `RERANK_MODEL` - Reranking model name
- `LOG_LEVEL` - Logging level (default: `INFO`)

## Deployment Steps

### Step 1: Push to GitHub

```bash
# Make sure you're on the deployment branch
git checkout deployment

# Add and commit deployment files
git add Dockerfile .dockerignore smithery.yaml
git commit -m "Add deployment configuration for Smithery"

# Push to GitHub
git push origin deployment
```

### Step 2: Connect to Smithery

1. Go to [smithery.ai](https://smithery.ai)
2. Navigate to "Deploy" section
3. Select "From GitHub"
4. Connect your GitHub repository
5. Select the `deployment` branch (or `main` if you merge)

### Step 3: Configure Environment Variables

In the Smithery dashboard:

1. Go to your deployment settings
2. Add environment variables:
   - `RETRIEVAL_ENDPOINT`
   - `RETRIEVAL_API_KEY`
   - `LLM_API_URL`
   - `LLM_MODEL`
   - `LLM_API_TOKEN`
   - (Optional) `RERANK_URL`
   - (Optional) `RERANK_MODEL`

### Step 4: Deploy

1. Smithery will automatically:
   - Build the Docker image from `Dockerfile`
   - Deploy the service
   - Provide a stable URL

2. Your MCP server will be available at:
   ```
   https://your-service.smithery.ai/mcp
   ```

## Testing the Deployment

Once deployed, test your MCP server:

```python
import asyncio
from mcp import ClientSession

async def test():
    async with ClientSession("https://your-service.smithery.ai/mcp") as session:
        result = await session.call_tool("assistant", {
            "dataset_info": '[{"id": "test", "name": "Test"}]',
            "query": "Hello, world!"
        })
        print(result.content[0].text)

asyncio.run(test())
```

## Troubleshooting

### Server Not Starting

1. **Check Logs**: View logs in Smithery dashboard
2. **Verify Environment Variables**: Ensure all required variables are set
3. **Check Port**: Make sure port 5210 is configured correctly

### Connection Issues

1. **Verify URL**: Check that the MCP endpoint is `/mcp`
2. **Check Transport**: Ensure `streamable-http` transport is used
3. **Health Check**: Test the health endpoint at `/mcp`

### Environment Variable Issues

If the server starts but tools fail:
- Verify `RETRIEVAL_API_KEY` is correct
- Check `LLM_API_TOKEN` is valid
- Ensure `RETRIEVAL_ENDPOINT` is accessible

## Updating Deployment

To update your deployment:

```bash
# Make changes to your code
git add .
git commit -m "Update deployment"
git push origin deployment

# Smithery will automatically rebuild and redeploy
```

## Alternative: Using PyPI Package

Since your package is on PyPI, you could also install it directly:

```dockerfile
# In Dockerfile (alternative approach)
FROM python:3.12-slim
RUN pip install kbbridge
CMD ["python", "-m", "kbbridge.server", "--host", "0.0.0.0", "--port", "5210"]
```

However, building from source (current approach) gives you more control.

## Resources

- [Smithery Documentation](https://smithery.ai/docs)
- [MCP Protocol](https://modelcontextprotocol.io)
- [KB-Bridge GitHub](https://github.com/egpivo/kb-bridge)
