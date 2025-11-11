# Integrations

Backend abstraction for retrieval systems (Dify, OpenSearch, Pinecone, etc.)

## Usage

```python
from kbbridge.integrations import BackendAdapterFactory, RetrievalCredentials

# Create resource-bound adapter
credentials = RetrievalCredentials.from_env()
adapter = BackendAdapterFactory.create(
    resource_id="my-resource",
    credentials=credentials
)

# Search (no resource_id needed - adapter is resource-bound)
results = adapter.search(
    query="What is the vacation policy?",
    top_k=20
)

# List files (no resource_id needed - adapter is resource-bound)
files = adapter.list_files()
```

## Architecture

```
integrations/
├── backend_adapter.py        # BackendAdapter (ABC) and BackendAdapterFactory
├── credentials.py            # RetrievalCredentials
├── retriever_base.py         # Abstract Retriever, ChunkHit, FileHit
├── retriever_router.py       # Factory for creating retrievers
├── dify/
│   ├── dify_credentials.py   # DifyCredentials (backward compatibility)
│   ├── dify_backend_adapter.py  # DifyBackendAdapter implementation
│   └── dify_retriever.py     # Low-level DifyRetriever implementation
└── README.md
```

## Adding New Backends

1. Implement `BackendAdapter` interface from `backend_adapter.py`
2. Register with `BackendAdapterFactory`

```python
class OpenSearchBackendAdapter(BackendAdapter):
    def search(self, query: str, **options) -> Dict[str, Any]: ...
    def list_files(self, timeout: int = 30) -> List[str]: ...
    def build_metadata_filter(self, document_name: str = "") -> Optional[Dict]: ...
```
