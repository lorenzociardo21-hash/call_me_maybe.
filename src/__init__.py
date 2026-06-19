from typing import Any
from pydantic import BaseModel


class SchemaTypeInfo(BaseModel):
    type: str


class FunctionsDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, SchemaTypeInfo]
    returns: dict[str, Any]


class FunctionCallingTests(BaseModel):
    prompt: str
