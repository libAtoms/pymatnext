"""Shared Pydantic model configuration for pymatnext parameters."""

from pydantic import BaseModel, ConfigDict


class PymatnextParams(BaseModel):
    """Base class for parameter models with a closed schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
