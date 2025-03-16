from pydantic import BaseModel

class RequestData(BaseModel):
    prompt: str
    max_length: int = 100
