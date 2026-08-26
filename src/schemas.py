from typing import Dict
from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):
    """
    Define the schema for single function parameter pr return value
    """
    type: str = Field(
        ...,
        description="The expected data "
        "type (e.g., 'string', 'number', 'boolean')"
    )


class FunctionDefinition(BaseModel):
    """
    Represents a single function that the LLM is authorized to call.
    """
    name: str = Field(
        ...,
        description="The exact name of the function to be called"
    )
    description: str = Field(
        ...,
        description="A natural language explanation of what the function does"
    )
    parameters: Dict[str, ParameterSchema] = Field(
        default_factory=dict,
        description="A mapping of parametr Names to their Schema ")
    returns: ParameterSchema = Field(
        ..., description="The expected return type Schema of the function ")
