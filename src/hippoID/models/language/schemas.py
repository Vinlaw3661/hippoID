from pydantic import BaseModel, Field

class PersonName(BaseModel):
    """Name of person extracted from provided text"""
    name: str = Field(description="The name of the person")