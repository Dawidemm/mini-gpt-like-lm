from fastapi import APIRouter
from minigpt.app.models.request_model import RequestData
from minigpt.app.services.service import generate_text

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/generate")
async def generate_response(request: RequestData):
    response = generate_text(request.prompt, request.max_length)
    return {"response": response}