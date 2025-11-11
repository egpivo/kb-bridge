# Design Proposal: Move Backend-Specific Identifiers to Integrations Layer

## Problem
Currently, services expose `dataset_id` which is Dify-specific. Other backends use different identifiers:
- Dify: `dataset_id`
- OpenSearch: `index_name`
- n8n: No identifier needed (webhook-based)

This makes the service layer backend-specific, which is not ideal for a generic solution.

## Proposed Solution

### Option 1: Generic Resource Identifier + Adapter Translation (Recommended)
- Services use a generic `resource_id` parameter
- Adapter/router translates to backend-specific identifiers during initialization
- Retriever instances are created with backend-specific identifiers already set

**Pros:**
- Services remain backend-agnostic
- Clear separation of concerns
- Easy to add new backends

**Cons:**
- Requires refactoring existing services
- Need to update Retriever interface

### Option 2: Resource-Bound Adapters ⭐ **RECOMMENDED**
- Create adapter instances bound to a resource identifier
- Services don't need to pass identifiers - they're encapsulated in the adapter
- Adapter methods don't require identifier parameters
- **Each adapter handles ALL backend-specific concerns** (identifiers, metadata filters, API formats)

**Pros:**
- Clean API - no identifier parameters in service methods
- Strong encapsulation of backend-specific logic
- **Perfect for metadata filters**: Each adapter translates generic filter concepts to backend-specific syntax
  - Dify: `{"conditions": [{"name": "...", "comparison_operator": "...", "value": "..."}]}`
  - OpenSearch: Query DSL format (e.g., `{"term": {...}}` or `{"match": {...}}`)
  - Other backends: Their own formats
- Services work with generic concepts (e.g., `document_name="file.pdf"`), adapter handles translation
- Easy to extend: New backend = new adapter with its own filter logic
- No backend-specific knowledge leaks into service layer

**Cons:**
- Adapter instances are less reusable (but this is actually a feature - they're resource-bound)
- May need adapter factory per resource (but this is manageable and provides clear separation)

### Option 3: Backend-Specific Adapter Methods
- Keep generic identifier in services
- Each adapter handles translation internally
- Services pass generic identifier, adapter translates

**Pros:**
- Minimal changes to services
- Translation logic centralized in adapters

**Cons:**
- Still exposes backend-specific concept (`dataset_id`) in services
- Less clear what identifier means for different backends

## Recommended Approach: Option 1 + Option 2 Hybrid

1. **Create a generic `BackendAdapter` interface** that encapsulates:
   - Credentials
   - Resource identifier (translated to backend-specific format)
   - Backend type

2. **Services use generic `resource_id`** parameter

3. **Adapter factory creates resource-bound adapters**:
   ```python
   adapter = BackendAdapterFactory.create(
       resource_id="my-kb",
       backend_type="dify",
       credentials=credentials
   )
   # adapter is now bound to "my-kb" (translated to dataset_id for Dify)
   ```

4. **Service methods don't need resource identifier**:
   ```python
   def retriever_service(
       resource_id: str,  # Generic identifier
       query: str,
       ...
   ):
       adapter = BackendAdapterFactory.create(resource_id, ...)
       return adapter.search(query=query, ...)
   ```

5. **Adapter methods are identifier-free**:
   ```python
   class BackendAdapter:
       def search(self, query: str, ...):
           # Uses self.resource_id (already translated)
           pass
   ```

## Implementation Steps

1. Create `BackendAdapter` base class in `integrations/`
2. Create adapter implementations (DifyAdapter, OpenSearchAdapter, etc.)
3. Create `BackendAdapterFactory` that handles resource_id translation
4. Refactor services to use generic `resource_id` and adapters
5. Update Retriever interface to remove `dataset_id` from method signatures
6. Update all call sites

## Questions to Consider

1. Should we keep backward compatibility with `dataset_id` parameter name?
2. How should we handle the transition period?
3. Should `resource_id` be optional (some backends don't need it)?
4. How do we handle list_files which currently requires dataset_id?
