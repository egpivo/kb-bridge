"""
Core package

Provides lightweight shims for subpackages to make patch paths resolvable in
tests without importing heavy optional dependencies at import time.
"""

try:  # pragma: no cover - lightweight shim only used for patching
    import types as _types

    class _ReflectionIntegrationStub:
        async def reflect_on_answer(self, *args, **kwargs):
            return "", {}

    def _parse_reflection_params_stub(*args, **kwargs):
        return {
            "enable_reflection": False,
            "quality_threshold": 0.0,
            "max_iterations": 0,
        }

    reflection = _types.SimpleNamespace(
        integration=_types.SimpleNamespace(
            ReflectionIntegration=_ReflectionIntegrationStub,
            parse_reflection_params=_parse_reflection_params_stub,
        )
    )
except Exception:
    pass

__all__ = ["reflection"]
