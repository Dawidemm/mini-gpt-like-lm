from fastapi import FastAPI
# from app.routers import chat

app = FastAPI(title="MiniGPT Chat API")

# Dodanie routera dla endpointów czatu
# app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "MiniGPT API is running!"}