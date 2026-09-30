"""SIMD Backends: Native C++ vectorised planning (AVX2/AVX-512/NEON + OpenMP)."""

from __future__ import annotations

from ai.planning.backends.cpp_backend import CPPSimdDWABackend

__all__ = ["CPPSimdDWABackend"]
