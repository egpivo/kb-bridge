"""
Internal prompts for QA Hub core processing.

NOTE: All prompt functions have been migrated to DSPy Signatures.
The prompts are now co-located with their respective modules:
- AnswerExtractionSignature in answer_extractor.py (replaced qa_hub_system/query)
- FileSearchKeywordSignature and ContentBoostKeywordSignature in keyword_generator.py
- ParagraphLocationSignature in paragraph_locator.py
- StructuredAnswerSignature in structured_answer_formatter.py
- UserIntentionSignature in user_intention_extractor.py
- ContentClusterSignature in content_cluster.py
- QueryAnalysisSignature, QueryExpansionSignature, QueryRelaxationSignature in llm_query_rewriter.py

This file is kept as a module marker. All prompting logic has been moved to DSPy Signatures.
"""

# All prompt functions have been removed.
# Use DSPy Signatures directly in their respective modules.
