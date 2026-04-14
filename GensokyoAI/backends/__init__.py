"""后端模块"""

# GensokyoAI/backends/__init__.py

from .base import BaseBackend
from .console import ConsoleBackend, ConsoleBackendBuilder

__all__ = [
    "BaseBackend",
    "ConsoleBackend",
    "ConsoleBackendBuilder",
]
