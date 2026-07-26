from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# --- CORS: разрешаем фронтенд GitHub Pages ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://makcsml21-cloud.github.io",
        "https://makcsml21-cloud.github.io/travel-assistant-ui/",
        "*"  # для локальной отладки
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list
    model: str = "deepseek-r1:8b"  # модель по умолчанию

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@app.post("/chat")
async def chat(request: ChatRequest):
    # Формируем сообщения для Ollama
    messages = []
    for item in request.history:
        # Ожидаем формат: {"role": "...", "content": "..."}
        if "role" in item and "content" in item:
            messages.append({"role": item["role"], "content": item["content"]})
    
    messages.append({"role": "user", "content": request.message})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": request.model,
                "messages": messages,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        assistant_text = data.get("message", {}).get("content", "Нет ответа от ассистента.")
        return {"message": assistant_text}

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Не удаётся подключиться к Ollama. Проверьте, запущен ли сервер.")
    except Exception as e:
        print(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

@app.get("/")
async def root():
    return {"status": "ok", "service": "travel-assistant-ollama"}
