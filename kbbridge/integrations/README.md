# Integrations

Backend abstraction for retrieval systems (Dify, OpenSearch, Pinecone, etc.)

## Usage

```python
from kbbridge.integrations import DifyAdapter, DifyCredentials

# Create adapter
credentials = DifyCredentials.from_env()
adapter = DifyAdapter(credentials=credentials)

# Search
results = adapter.search(
    dataset_id="my-dataset",
    query="What is the vacation policy?",
    top_k=20
)

# List files
files = adapter.list_files(dataset_id="my-dataset")
```

## Architecture

```
integrations/
├── retriever_base.py        # Abstract Retriever, ChunkHit, FileHit
├── retriever_router.py       # Factory for creating retrievers
├── dify/
│   ├── dify_credentials.py   # Credential management
│   ├── dify_adapter.py       # High-level API
│   └── dify_retriever.py     # Low-level implementation
└── README.md
```

## Adding New Backends

1. Implement `Retriever` interface from `retriever_base.py`
2. Register with `RetrieverRouter`

```python
class OpenSearchRetriever(Retriever):
    def call(self, *, query, method, top_k, **kw): ...
    def normalize_chunks(self, resp): ...
    def group_files(self, chunks): ...
    def build_metadata_filter(self, **kwargs): ...
    def list_files(self, **kwargs): ...
```
