from fastapi import FastAPI
from minigpt.app.routers import chat

app = FastAPI(title="MiniGPT Chat API")

app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "MiniGPT API is running!"}