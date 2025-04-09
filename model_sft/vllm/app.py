from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
from config import VLLM_ENDPOINT, HEADERS

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat_api(req: ChatRequest):
    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": "你是一个中文助手"},
            {"role": "user", "content": req.user_input}
        ]
    }
    res = requests.post(VLLM_ENDPOINT, headers=HEADERS, json=payload)
    return res.json()
