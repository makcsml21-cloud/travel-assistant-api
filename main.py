import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

# --- CORS: обязательно для фронтенда из index.html ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                 # для локальной разработки
    allow_credentials=True,
    allow_methods=["*"],                # разрешает POST, OPTIONS и т.д.
    allow_headers=["*"],
)

# --- Конфигурация ---
YA_API_KEY = os.getenv("YA_API_KEY")
YA_FOLDER_ID = os.getenv("YA_FOLDER_ID")

if not YA_API_KEY or not YA_FOLDER_ID:
    print("⚠️ Внимание: YA_API_KEY или YA_FOLDER_ID не заданы в .env")

CACHE: Dict[str, str] = {}
MAX_HISTORY_MESSAGES = 10

# --- Системный промпт (v3.1) ---
SYSTEM_PROMPT = """
Ты — тревел‑консультант для MVP v2.2. Твоя задача — предложить 2 реалистичных варианта маршрута на Валаам из Санкт‑Петербурга: «Эконом» и «Средний». Каждый вариант должен описывать поездку по дням (понедельник–воскресенье), опираясь только на общедоступные и проверяемые факты.

Для каждого дня в каждом варианте обязательно должны быть:
- день недели и дата (шаблон: «Понедельник, 4 августа»);
- основное действие (1–2 пункта);
- транспорт: откуда, куда, тип транспорта, примерное время в пути, ориентировочная стоимость;
- размещение: тип (хостел/отель/гостевой дом), ориентир по цене за ночь;
- питание: 2–3 варианта по бюджету (эконом/средний).

Если точных данных по какой‑то локации или стоимости нет — не выдумывай. Напиши явно: «Нет подтверждённых данных по этому пункту».

После того как напишешь черновик ответа, проверь себя отдельно для каждого варианта: для каждого дня есть ли упоминание транспорта (тип, время, цена) и бюджета (размещение, питание, общая оценка). Если чего‑то нет — допиши.

Не использу списки, маркеры, заголовки, жирный шрифт и Markdown. Пиши сплошным связным текстом. Сначала опиши вариант «Эконом», затем вариант «Средний». Стиль — нейтральный, практичный, пригодный для планирования поездки.
"""

# --- Модели данных ---
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, Any]] = []

# --- Вспомогательные функции ---
def truncate_history(history: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(history) <= limit:
        return history
    return history[-limit:]

def make_cache_key(question: str, history: List[Dict[str, Any]]) -> str:
    truncated = truncate_history(history, MAX_HISTORY_MESSAGES)
    key_parts = [m["role"] + "|" + m["text"] for m in truncated]
    key = "|".join(key_parts) + "|" + question
    return key

def call_yandex_gpt(question: str, history: List[Dict[str, Any]]) -> str:
    messages = []
    messages.append({"role": "system", "text": SYSTEM_PROMPT})
    for msg in history:
        messages.append({"role": msg["role"], "text": msg["text"]})
    messages.append({"role": "user", "text": question})

    payload = {
        "modelUri": f"gpt://{YA_FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "temperature": 0.7,
            "maxTokens": 2000
        },
        "messages": messages
    }

    headers = {
        "Authorization": f"Api-Key {YA_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        json=payload,
        headers=headers,
        timeout=60
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    candidates = data.get("result", {}).get("candidates", [])
    if not candidates:
        raise HTTPException(status_code=500, detail="Пустой ответ от YandexGPT")

    return candidates[0]["text"]

# --- Эндпоинт ---
@app.post("/chat")
async def chat(req: ChatRequest):
    if not YA_API_KEY or not YA_FOLDER_ID:
        return JSONResponse(
            content={"answer": "", "fromCache": False, "error": "Missing YA_API_KEY or YA_FOLDER_ID in .env"},
            status_code=500
        )

    cache_key = make_cache_key(req.question, req.history)
    if cache_key in CACHE:
        answer = CACHE[cache_key]
        return JSONResponse(
            content={"answer": answer, "fromCache": True},
            media_type="application/json; charset=utf-8"
        )

    try:
        answer = call_yandex_gpt(req.question, req.history)
        CACHE[cache_key] = answer
        return JSONResponse(
            content={"answer": answer, "fromCache": False},
            media_type="application/json; charset=utf-8"
        )
    except Exception as e:
        return JSONResponse(
            content={"answer": "", "fromCache": False, "error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
