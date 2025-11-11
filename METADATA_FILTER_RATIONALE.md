## Why Option 2 is Perfect for Metadata Filters

### Current Situation
Different backends use completely different metadata filter formats:

**Dify:**
```python
{
    "conditions": [
        {
            "name": "document_name",
            "comparison_operator": "contains",  # or "is", "is_not", etc.
            "value": "file.pdf"
        }
    ],
    "logical_operator": "and"  # or "or"
}
```

**OpenSearch (hypothetical):**
```python
{
    "query": {
        "term": {"document_name": "file.pdf"}
    }
}
# OR
{
    "query": {
        "match": {"document_name": {"query": "file.pdf", "fuzziness": "AUTO"}}
    }
}
```

**Other backends:** Will have their own formats

### How Option 2 Solves This

With **Resource-Bound Adapters**, each adapter encapsulates ALL backend-specific knowledge:

```python
# Service layer - completely generic
def retriever_service(resource_id: str, query: str, document_name: str = ""):
    adapter = BackendAdapterFactory.create(resource_id, credentials)

    # Service works with generic concepts
    # Adapter handles backend-specific translation
    return adapter.search(
        query=query,
        document_name=document_name  # Generic concept
    )

# Adapter layer - backend-specific
class DifyBackendAdapter(BackendAdapter):
    def search(self, query: str, document_name: str = "", **options):
        # Translate generic document_name to Dify-specific format
        if document_name:
            metadata_filter = {
                "conditions": [{
                    "name": "document_name",
                    "comparison_operator": "contains",
                    "value": document_name
                }]
            }
        else:
            metadata_filter = None

        # Use Dify-specific filter format
        return self._dify_retriever.call(
            query=query,
            metadata_filter=metadata_filter
        )

class OpenSearchBackendAdapter(BackendAdapter):
    def search(self, query: str, document_name: str = "", **options):
        # Translate generic document_name to OpenSearch Query DSL
        if document_name:
            metadata_filter = {
                "query": {
                    "term": {"document_name": document_name}
                }
            }
        else:
            metadata_filter = None

        # Use OpenSearch-specific filter format
        return self._opensearch_client.search(
            query=query,
            filter=metadata_filter
        )
```

### Benefits

1. **Services stay generic**: They work with concepts like `document_name`, not backend-specific formats
2. **Adapters handle complexity**: Each adapter knows how to translate generic filters to its backend's format
3. **Easy to extend**: Adding a new backend with different filter syntax? Just implement a new adapter
4. **No leakage**: Backend-specific knowledge (like `comparison_operator`, Query DSL, etc.) never reaches the service layer
5. **Future-proof**: When backends add new filter capabilities, only the adapter needs to change

### Example: Future Filter Extensions

When backends add more filter capabilities:

```python
# Service layer - still generic
adapter.search(
    query=query,
    document_name="file.pdf",
    author="John Doe",  # New generic filter
    date_range={"start": "2024-01-01", "end": "2024-12-31"}  # New generic filter
)

# Dify adapter - translates to Dify format
class DifyBackendAdapter:
    def search(self, query: str, document_name: str = "", author: str = "", date_range: dict = None):
        conditions = []
        if document_name:
            conditions.append({
                "name": "document_name",
                "comparison_operator": "contains",
                "value": document_name
            })
        if author:
            conditions.append({
                "name": "author",
                "comparison_operator": "is",
                "value": author
            })
        if date_range:
            conditions.append({
                "name": "created_at",
                "comparison_operator": "between",
                "value": [date_range["start"], date_range["end"]]
            })

        return {"conditions": conditions} if conditions else None

# OpenSearch adapter - translates to Query DSL
class OpenSearchBackendAdapter:
    def search(self, query: str, document_name: str = "", author: str = "", date_range: dict = None):
        must_clauses = []
        if document_name:
            must_clauses.append({"term": {"document_name": document_name}})
        if author:
            must_clauses.append({"term": {"author": author}})
        if date_range:
            must_clauses.append({
                "range": {
                    "created_at": {
                        "gte": date_range["start"],
                        "lte": date_range["end"]
                    }
                }
            })

        return {"query": {"bool": {"must": must_clauses}}} if must_clauses else None
```

**Services never need to know about the differences!**
