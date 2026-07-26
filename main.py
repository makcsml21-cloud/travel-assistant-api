from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from yandex_cloud_ai import YandexGPT

app = FastAPI()

# --- CORS: критически важно для GitHub Pages ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://makcsml21-cloud.github.io",
        "https://makcsml21-cloud.github.io/travel-assistant-ui/",
        "*"  # запасной вариант
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list

YA_API_KEY = os.getenv("YA_API_KEY")
YA_FOLDER_ID = os.getenv("YA_FOLDER_ID")

if not YA_API_KEY or not YA_FOLDER_ID:
    raise ValueError("YA_API_KEY и YA_FOLDER_ID должны быть заданы в переменных окружения")

llm = YandexGPT(api_key=YA_API_KEY, folder_id=YA_FOLDER_ID)

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = []
    for item in request.history:
        if "role" in item and "text" in item:
            messages.append({"role": item["role"], "text": item["text"]})
    messages.append({"role": "user", "text": request.message})

    try:
        response = llm.generate(messages=messages)
        alternatives = response.result.alternatives
        if not alternatives or len(alternatives) == 0:
            return {"message": "Не удалось получить ответ от LLM."}
        assistant_text = alternatives[0].message.text
    except Exception as e:
        print(f"LLM error: {e}")
        return {"message": f"Ошибка при генерации ответа: {str(e)}"}

    return {"message": assistant_text}

@app.get("/")
async def root():
    return {"status": "ok"}
