from fastapi import APIRouter
from minigpt.app.models.request_model import RequestData
from minigpt.app.services.service import generate

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/generate")
async def generate_response(request: RequestData):
    response = generate(request.prompt, request.max_length)
    return {"response": response}