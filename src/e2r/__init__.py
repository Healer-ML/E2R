"""Core implementation of E2R for farmland image-text retrieval."""

from .loss import CLIPLoss
from .model import ItraModel, get_model

__all__ = ["CLIPLoss", "ItraModel", "get_model"]
