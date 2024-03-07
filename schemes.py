from pydantic import BaseModel

from typing import List, Dict, Union

class Record(BaseModel):
    id: int
    vector: List[float]
    vector1: List[float]  # Changed from vector to vector1
    payload: Dict[str, Union[str, int, float]]
