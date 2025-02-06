from typing import Dict
from pydantic import BaseModel


class Coordinates(BaseModel):
    x: int | float
    y: int | float
    width: int | float
    height: int | float
    confidence: float