from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://makcsml21-cloud.github.io",
        "https://makcsml21-cloud.github.io/travel-assistant-ui/",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list
    model: str = "llama-3.2-1b-instruct"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"

@app.post("/chat")
async def chat(request: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY не задан в переменных окружения")

    messages = []
    for item in request.history:
        if "role" in item and "content" in item:
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": request.message})

    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": request.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise HTTPException(status_code=502, detail="Пустой ответ от LLM")

        assistant_text = choices[0].get("message", {}).get("content", "Нет ответа от ассистента.")
        return {"message": assistant_text}

    except requests.exceptions.RequestException as e:
        print(f"LLM request error: {e}")
        raise HTTPException(status_code=502, detail=f"Ошибка связи с LLM: {str(e)}")
    except Exception as e:
        print(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")

@app.get("/")
async def root():
    return {"status": "ok", "service": "travel-assistant-groq"}
