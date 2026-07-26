from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# --- CORS: разрешаем запросы с фронтенда и локально ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://makcsml21-cloud.github.io/travel-assistant-ui/",
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YA_API_KEY = os.getenv("YA_API_KEY")
YA_FOLDER_ID = os.getenv("YA_FOLDER_ID")

if not YA_API_KEY or not YA_FOLDER_ID:
    raise RuntimeError("YA_API_KEY и/или YA_FOLDER_ID не заданы в переменных окружения!")

YA_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundation-models/v1/foundationModels/textCompletion"

def validate_user_query(query: str) -> str:
    """Простая семантическая валидация: не пустой, не слишком длинный, без явного мусора."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым.")
    if len(query.strip()) > 2000:
        raise HTTPException(status_code=400, detail="Слишком длинный запрос (максимум 2000 символов).")
    return query.strip()

@app.post("/chat")
async def chat(payload: dict):
    user_message = payload.get("message")
    history = payload.get("history", [])  # список {"role": "user"/"assistant", "text": "..."}

    # Валидация
    user_message = validate_user_query(user_message)

    # Формируем контекст (история + новый запрос)
    messages = [
        {"role": item["role"], "text": item["text"]}
        for item in history
    ]
    messages.append({"role": "user", "text": user_message})

    # Запрос к YandexGPT
    body = {
        "modelUri": f"gpt://{YA_FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "temperature": 0.7,
            "maxTokens": 1000
        },
        "messages": messages
    }

    headers = {
        "Authorization": f"Api-Key {YA_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(YA_COMPLETION_URL, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        assistant_text = data["result"]["alternatives"][0]["message"]["text"]
    except Exception as e:
        # В реальном MVP тут логирование и более детальные ошибки
        raise HTTPException(status_code=502, detail=f"Ошибка при обращении к LLM: {str(e)}")

    return {
        "message": assistant_text,
        "history": messages + [{"role": "assistant", "text": assistant_text}]
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "travel-assistant-backend"}
