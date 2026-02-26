from pydantic import BaseModel

class RegisterResponse(BaseModel):
    status: str
    message: str
    embedding_size: int

class VerifyResponse(BaseModel):
    status: str
    message: str
    verified: bool
    similarity_score: float