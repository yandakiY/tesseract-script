from typing import Dict
from pydantic import BaseModel

from models.coordinates import Coordinates
from models.dimensions import Dimensions

class AnnotationRequest(BaseModel):
    dimensions: Dimensions
    coordonnees: Dict[str, Coordinates]