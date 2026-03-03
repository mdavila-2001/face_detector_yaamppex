from pydantic import BaseModel
from typing import Optional

class RegisterResponse(BaseModel):
    status: str
    message: str
    embedding_size: int
    liveness_score: float

class VerifyResponse(BaseModel):
    status: str
    message: str
    verified: bool
    similarity_score: float
    liveness_score: float
    action: Optional[str] = None
    user_id: Optional[str] = None