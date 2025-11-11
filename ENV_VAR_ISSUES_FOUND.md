# Environment Variable Issues Found

## Summary
Searched codebase for similar environment variable problems like the reranking issue we fixed.

## Issues Found and Status

### ✅ Fixed Issues

1. **`kbbridge/utils/working_components.py`** - `KnowledgeBaseRetriever.retrieve()`
   - **Issue**: Hardcoded `reranking_provider_name="cohere"` and `reranking_model_name="rerank-multilingual-v2.0"`
   - **Fix**: Use `DifyRetrieverDefaults` with environment variable support
   - **Status**: ✅ Fixed

2. **`kbbridge/utils/working_components.py`** - `_process_advanced_approach()`
   - **Issue**: Hardcoded reranking values and `does_rerank=True`
   - **Fix**: Use `DifyRetrieverDefaults` and check `credentials.is_reranking_available()`
   - **Status**: ✅ Fixed

3. **`kbbridge/integrations/dify/constants.py`** - `DifyRetrieverDefaults`
   - **Issue**: Hardcoded enum values, no environment variable support
   - **Fix**: Added `DIFY_RERANKING_PROVIDER_NAME` and `DIFY_RERANKING_MODEL_NAME` env var support
   - **Status**: ✅ Fixed

4. **`kbbridge/core/discovery/file_discover.py`** - `_retrieve()`
   - **Issue**: Empty string defaults `reranking_provider_name=""` and `reranking_model_name=""`
   - **Fix**: Use `DifyRetrieverDefaults` instead of empty strings
   - **Status**: ✅ Fixed

### ⚠️ Potential Issues (Not Critical)

1. **Hardcoded "openai/" prefix** (4 locations)
   - **Files**:
     - `kbbridge/core/synthesis/answer_extractor.py` (line 96)
     - `kbbridge/core/query/rewriter.py` (line 128)
     - `kbbridge/core/query/intention_extractor.py` (line 107)
     - `kbbridge/core/extraction/content_cluster.py` (line 119)
   - **Issue**: `model=f"openai/{self.llm_model}"` assumes OpenAI-compatible API format
   - **Impact**: Low - This is DSPy-specific and OpenAI-compatible APIs are common
   - **Recommendation**: Consider making configurable if supporting non-OpenAI formats becomes necessary

2. **Hardcoded "Bearer dummy_token"** (8 locations)
   - **Files**: Multiple constants files
   - **Issue**: Placeholder authorization header values
   - **Impact**: None - These appear to be intentional placeholders, not used in actual API calls
   - **Recommendation**: No action needed

3. **Hardcoded `does_rerank=True` defaults**
   - **Files**:
     - `kbbridge/integrations/dify/dify_adapter.py` (line 114)
     - `kbbridge/config/constants.py` (lines 10, 77)
   - **Issue**: Default values assume reranking is enabled
   - **Impact**: Low - These are defaults, actual usage checks credentials availability
   - **Recommendation**: Already handled by credential checks in actual usage

## Environment Variables Added

- `DIFY_RERANKING_PROVIDER_NAME` - Override Dify reranking provider name
- `DIFY_RERANKING_MODEL_NAME` - Override Dify reranking model name

## Testing

All fixes have been tested:
- ✅ All existing tests pass
- ✅ Reranking availability checks work correctly
- ✅ Environment variable overrides work correctly
- ✅ Defensive checks prevent errors when config is missing
